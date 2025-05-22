import torch
import logging

def inspect_batch(batch_features, batch_labels, feature_names=None, max_print=5):
    if not isinstance(batch_features, torch.Tensor):
        batch_features = torch.tensor(batch_features)
    if not isinstance(batch_labels, torch.Tensor):
        batch_labels = torch.tensor(batch_labels)

    B, T, F = batch_features.shape

    flat_features = batch_features.view(-1, F)
    flat_labels = batch_labels.view(-1)

    for i in range(F):
        feature = flat_features[:, i]
        name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
        stats = {
            "min": feature.min().item(),
            "max": feature.max().item(),
            "mean": feature.mean().item(),
            "std": feature.std().item(),
            "nan": torch.isnan(feature).sum().item(),
            "inf": torch.isinf(feature).sum().item(),
        }

        if stats["nan"] > 0 or stats["inf"] > 0 or abs(stats["max"]) > 1e4:
            logging.warning(f"[Inspect] ⚠️ {name}: {stats}")
        elif i < max_print:
            logging.info(f"[Inspect] ✅ {name}: {stats}")

    label_stats = {
        "min": flat_labels.min().item(),
        "max": flat_labels.max().item(),
        "mean": flat_labels.float().mean().item(),
        "std": flat_labels.float().std().item(),
        "nan": torch.isnan(flat_labels).sum().item(),
        "inf": torch.isinf(flat_labels).sum().item(),
    }
    logging.info(f"[Inspect] Labels: {label_stats}")
