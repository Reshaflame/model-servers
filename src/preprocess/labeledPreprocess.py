# src/preprocess/labeledPreprocess.py
import pandas as pd, gzip, os, shelve
from utils.meta_builder import dump_freq_dict, save_category_maps, save_feature_list
from utils.chunk_archiver import zip_chunks
from collections import defaultdict, deque
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHUNK_SIZE = 250_000          # keeps RAM < 1.5 GB on Runpod 8 GB node
ROLL_WINDOW = 3600            # seconds
COUNTS_DB   = "tmp/freq.db"   # on-disk key-value store

def preprocess_labeled_data_chunked(auth_gz=os.path.join(DATA_DIR, "auth.txt.gz"),
                                    red_gz=os.path.join(DATA_DIR, "redteam.txt.gz"),
                                    out_dir=os.path.join(DATA_DIR, "chunks_labeled")):
    
    os.makedirs(out_dir, exist_ok=True)
    # ---------- build redteam lookup ----------
    red_cols = ["time","user","src_comp","dst_comp"]
    # Quick check: print first row to confirm column count
    # with gzip.open(auth_gz, 'rt') as f:
    #     sample_line = f.readline().strip()
    #     num_fields = len(sample_line.split(','))
    #     print(f"[Schema Check] First row has {num_fields} fields: {sample_line}")
    with gzip.open(red_gz,'rt') as f:
        red = pd.read_csv(f, names=red_cols)
    red_set = set(zip(red.time, red.user, red.src_comp, red.dst_comp))
    os.makedirs("tmp", exist_ok=True)

    # ---------- freq counters ----------
    freq_user  = shelve.open(COUNTS_DB+"_user",   writeback=True)
    freq_pc    = shelve.open(COUNTS_DB+"_pc",     writeback=True)
    freq_pair  = shelve.open(COUNTS_DB+"_pair",   writeback=True)
    freq_dom   = shelve.open(COUNTS_DB+"_dom",    writeback=True)

    # rolling stats: {user: deque[(timestamp, is_fail)]}
    windows = defaultdict(deque)

    col_names = ["time","src_user","dst_user","src_comp","dst_comp","auth_type","logon_type","auth_orientation","success"]
    seen_auth, seen_logon, seen_orient = set(), set(), set()
    chunk_id = -1
    num_chunks_saved = 0
    with gzip.open(auth_gz,'rt') as f:
        for df in pd.read_csv(f, names=col_names, chunksize=CHUNK_SIZE):
            try:
                chunk_id += 1
                df = df.dropna(subset=["time", "src_user", "src_comp"])   # ← NEW
                df["time"] = pd.to_numeric(df["time"], errors="coerce")
                df["time"] = df["time"].astype(int)                       # ← NEW, safe cast
                if df.empty:
                    print(f"[Chunk {chunk_id}] ⚠️ Empty after dropna; skipping.")
                    continue
                df["label"] = df.apply(lambda r: (r.time,r.src_user,
                                                r.src_comp,r.dst_comp) in red_set,
                                        axis=1).astype(np.float32)

                # ---------- derive features ----------
                df["utc_hour"] = (df.time // 3600) % 24

                domains = df.src_user.str.split("@").str[-1]
                df["src_domain"] = domains

                # frequencies BEFORE current row
                df["user_freq"]  = domains.values  # placeholder; will fill below
                df["pc_freq"]    = 0
                df["pair_freq"]  = 0
                df["domain_freq"] = 0
                df["logins_1h_user"] = 0
                df["fails_1h_user"]  = 0
                df["fail_ratio_1h"]  = 0.0
                
                # ⬇️  collect categorical values for the meta maps
                seen_auth.update(df.auth_type.unique())
                seen_logon.update(df.logon_type.unique())
                seen_orient.update(df.auth_orientation.unique())
                
                # debug prints:
                print(f"[Chunk {chunk_id}] 🧪 Rows before dropna: {len(df)}")
                df = df.drop(columns=["datetime"])
                df = df.dropna(subset=["time", "src_user", "src_comp"])
                print(f"[Chunk {chunk_id}] 🧼 Rows after dropna: {len(df)}")

                # Ensure time is castable
                try:
                    df["time"] = df["time"].astype(int)
                except Exception as e:
                    print(f"[Chunk {chunk_id}] ❌ Casting time to int failed: {e}")
                    print(df["time"].head())
                    continue

                if df.empty:
                    print(f"[Chunk {chunk_id}] ⚠️ Empty after dropna/casting; skipping.")
                    continue

                print(f"[Chunk {chunk_id}] 🏷️ Columns: {df.columns.tolist()}")


                for row_idx, row in df.iterrows():
                    u, pc, t = row.src_user, row.src_comp, row.time

                    if row_idx >= len(domains):
                        print(f"[Chunk {chunk_id}] ❌ Invalid index: {row_idx} >= {len(domains)}")
                        continue

                    dom = domains.iloc[row_idx]

                    df.at[row_idx, "user_freq"]   = freq_user.get(u, 0)
                    df.at[row_idx, "pc_freq"]     = freq_pc.get(pc, 0)
                    df.at[row_idx, "pair_freq"]   = freq_pair.get(f"{u}|{pc}", 0)
                    df.at[row_idx, "domain_freq"] = freq_dom.get(dom, 0)

                    # rolling window update
                    q = windows[u]
                    # drop events older than 1h
                    while q and t - q[0][0] > ROLL_WINDOW: q.popleft()
                    logins = len(q)
                    fails  = sum(1 for ts,fail in q if fail)
                    df.at[row_idx, "logins_1h_user"] = logins
                    df.at[row_idx, "fails_1h_user"]  = fails
                    df.at[row_idx, "fail_ratio_1h"]  = fails / (logins + 1)

                    # append current event
                    q.append((t, 1-int(row.success)))  # success==1→fail flag 0

                    # increment global counters *after* feature capture
                    freq_user[u] = freq_user.get(u,0)+1
                    freq_pc[pc]  = freq_pc.get(pc,0)+1
                    freq_pair[f"{u}|{pc}"] = freq_pair.get(f"{u}|{pc}",0)+1
                    freq_dom[dom] = freq_dom.get(dom,0)+1

                # ---------- persist to CSV ----------
                out = os.path.join(out_dir, f"chunk_{chunk_id:04d}_labeled_feat.csv")
                df.drop(columns=["src_domain"]).to_csv(out, index=False)
                num_chunks_saved += 1
                print(f"[Chunk {chunk_id}] ➜ {out}")
            except Exception as e:
                print(f"[Chunk {chunk_id}] ❌ Failed: {e}")
   
    # ---------- persist metadata ----------
    print("[Meta] 🔧 Saving category maps and feature list...")
    save_category_maps(seen_auth, seen_logon, seen_orient)

    numeric_cols = [col for col in df.columns if col not in {"label", "src_domain"}]
    if "label" in df.columns:
        numeric_cols.append("label")  # include label if available
    save_feature_list(numeric_cols)

    print("[Meta] 🧮 Saving frequency tables...")
    dump_freq_dict(os.path.join(DATA_DIR, "meta", "user_freq.csv.gz"), dict(freq_user))
    dump_freq_dict(os.path.join(DATA_DIR, "meta", "pc_freq.csv.gz"), dict(freq_pc))
    dump_freq_dict(os.path.join(DATA_DIR, "meta", "domain_freq.csv.gz"), dict(freq_dom))
    
    freq_user.close(); freq_pc.close(); freq_pair.close(); freq_dom.close()

    for ext in [".bak", ".dat", ".dir", ".db"]:
        for base in [COUNTS_DB+"_user", COUNTS_DB+"_pc", COUNTS_DB+"_pair", COUNTS_DB+"_dom"]:
            try: os.remove(base + ext)
            except FileNotFoundError: pass

    print("[Meta] 📦 Zipping chunked CSVs into manageable files...")
    zip_chunks(out_dir, kind=os.path.basename(out_dir))
    print(f"[Preprocess] ✅ Saved {num_chunks_saved} enriched chunks to {out_dir}")

if __name__ == "__main__":
    print("[Preprocess] 🚀 Starting labeled preprocessing...")
    preprocess_labeled_data_chunked()
    print("[Preprocess] 🏁 Done.")
