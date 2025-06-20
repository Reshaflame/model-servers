# src/pipeline/gru_hybrid_pipeline.py
from glob import glob
import os, pandas as pd, torch
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from utils.constants import CHUNKS_LABELED_PATH
from utils.tuning import manual_gru_search
from utils.evaluator import evaluate_and_export
from utils.model_exporter import export_model
from models.gru_hybrid import train_gru, train_hybrid

def run_pipeline():
    chunk_dir = CHUNKS_LABELED_PATH
    first = glob(os.path.join(chunk_dir, "*.csv"))[0]
    numeric = pd.read_csv(first).drop(columns=['label']).select_dtypes('number').columns
    input_size = len(numeric)

    # ─── helpers ─────────────────────────────────────────────────────────
    def count_positives(loader_fn):
        """Iterate ONCE over the *entire* val generator and count labels."""
        tot = pos = 0
        for _, y in loader_fn():            # iterate lazily over every chunk
            tot += y.numel()
            pos += (y == 1).sum().item()
        return pos, tot
    
    dataset = SequenceChunkedDataset(chunk_dir,
                                     label_column='label',
                                     batch_size=64,
                                     shuffle_files=True,
                                     binary_labels=True,
                                     sequence_length=1,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')
    def val_once():
        """Yield *exactly one* pass over every val batch each time it's called."""
        yield from dataset.val_loader()

    pos_tr, tot_tr = count_positives(dataset.train_loader)
    pos_val, tot_val = count_positives(dataset.val_loader)
    
    print(f"🧮  train positives ≈ {pos_tr}/{tot_tr} "
        f"({100*pos_tr/max(1,tot_tr):.4f} %)")
    print(f"🧮  𝐯𝐚𝐥  positives ≈ {pos_val}/{tot_val} "
        f"({100*pos_val/max(1,tot_val):.4f} %)")
    
    param_grid = [
        {"lr":1e-3,  "hidden_size":48, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4,  "hidden_size":64, "num_layers":1, "epochs":6, "early_stop_patience":2},
        {"lr":1e-3,  "hidden_size":64, "num_layers":2, "epochs":6, "early_stop_patience":2},
        {"lr":5e-4,  "hidden_size":48, "num_layers":2, "epochs":6, "early_stop_patience":2},
    ]

    def train_func(cfg):
        best_f1, _ = train_gru(           # unpack tuple!
            cfg,
            loaders=(dataset.train_loader, val_once),
            input_size=input_size,
            tag=f"gru_h{cfg['hidden_size']}_l{cfg['num_layers']}",
            resume=True,
            eval_every_epoch=True
        )
        return best_f1

    best_cfg = manual_gru_search(train_func, param_grid)
    print("🏅 Best GRU config:", best_cfg)

    # Final GRU training using best config (resume = False to start fresh)
    best_tag = f"gru_h{best_cfg['hidden_size']}_l{best_cfg['num_layers']}"
    best_f1, gru_model = train_gru(       # unpack again
        best_cfg,
        loaders=(dataset.train_loader, val_once),
        input_size=input_size,
        tag=best_tag,
        resume=False,
        eval_every_epoch=False
    )
    print(f"👍 Final GRU F1 on val = {best_f1:.4f}")

    export_model(gru_model, "/app/models/gru_trained_model.pth")
    evaluate_and_export(gru_model,
                        dataset.full_loader(),
                        model_name="gru",
                        device=dataset.device,
                        export_ground_truth=True)

    # ───────────  HYBRID STAGE  ───────────
    print("\n🚀  Starting hybrid fine-tune…")
    hybrid_model = train_hybrid("/app/models/gru_trained_model.pth",
                                loaders=(dataset.train_loader, val_once),
                                input_size=input_size,
                                hidden_size=best_cfg["hidden_size"],
                                num_layers=best_cfg["num_layers"],
                                epochs=3,
                                lr=1e-3)
    evaluate_and_export(hybrid_model,
                        dataset.full_loader(),
                        model_name="gru_hybrid",
                        device=dataset.device)

if __name__ == "__main__":
    run_pipeline()
