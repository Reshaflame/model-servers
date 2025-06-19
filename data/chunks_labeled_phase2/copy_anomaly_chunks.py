#!/usr/bin/env python
# copy_anomaly_chunks.py
"""
Copy a small, anomaly-rich subset of Phase-2 chunks into a new directory
(e.g. data/chunks_labeled_phase2/slim_train) for faster emergency
training and zipping.
"""
import os, shutil, random, argparse, glob

# ----- configuration defaults ---------------------------------------
DEFAULT_SRC  = "."
DEFAULT_DST  = "slim_train"
DEFAULT_AUTO_FILE = "anomaly_chunks.txt"   # produced by scan_anomaly_chunks.py
KEEP_TOP_N_ANOM = 40                       # top N anomaly chunks
KEEP_RANDOM_N   = 50                       # random normal/quiet chunks
# --------------------------------------------------------------------

def load_anomaly_list(auto_file, manual_list):
    # if os.path.isfile(auto_file):
    #     with open(auto_file) as fh:
    #         lines = [ln.split(",")[0].strip() for ln in fh if ln.strip()]
    #     if lines:
    #         return lines                # already full filenames
    return manual_list

MANUAL_LIST = [
    "chunk_0600_labeled_enc.csv",
    "chunk_0602_labeled_enc.csv",
    "chunk_0603_labeled_enc.csv",
    "chunk_0830_labeled_enc.csv",
    "chunk_0603_labeled_enc.csv",
    "chunk_0827_labeled_enc.csv",
    "chunk_0605_labeled_enc.csv",
    "chunk_0601_labeled_enc.csv",
    "chunk_0833_labeled_enc.csv",
    "chunk_0821_labeled_enc.csv",
    "chunk_0828_labeled_enc.csv",
    "chunk_0585_labeled_enc.csv",
    "chunk_0597_labeled_enc.csv",
    "chunk_0606_labeled_enc.csv",
    "chunk_0824_labeled_enc.csv",
    "chunk_0826_labeled_enc.csv",
    "chunk_0831_labeled_enc.csv",
    "chunk_0574_labeled_enc.csv",
    "chunk_0581_labeled_enc.csv",
    "chunk_0851_labeled_enc.csv",
    "chunk_0852_labeled_enc.csv",
    "chunk_0586_labeled_enc.csv","chunk_0830_labeled_enc.csv",
    "chunk_0825_labeled_enc.csv","chunk_0827_labeled_enc.csv",
    "chunk_0835_labeled_enc.csv","chunk_0601_labeled_enc.csv",
    "chunk_0856_labeled_enc.csv","chunk_0602_labeled_enc.csv",
    "chunk_0917_labeled_enc.csv","chunk_0830_labeled_enc.csv",
    "chunk_0931_labeled_enc.csv","chunk_0827_labeled_enc.csv",
    "chunk_0562_labeled_enc.csv","chunk_0601_labeled_enc.csv",
    "chunk_0563_labeled_enc.csv","chunk_0602_labeled_enc.csv",
    "chunk_0584_labeled_enc.csv","chunk_0830_labeled_enc.csv",
    "chunk_0598_labeled_enc.csv","chunk_0827_labeled_enc.csv",
    "chunk_0829_labeled_enc.csv","chunk_0601_labeled_enc.csv",
    "chunk_0587_labeled_enc.csv","chunk_0602_labeled_enc.csv",
    "chunk_0607_labeled_enc.csv","chunk_0830_labeled_enc.csv",
    "chunk_0819_labeled_enc.csv","chunk_0827_labeled_enc.csv",
    "chunk_0841_labeled_enc.csv","chunk_0601_labeled_enc.csv",
    "chunk_0844_labeled_enc.csv","chunk_0602_labeled_enc.csv",
    "chunk_0175_labeled_enc.csv","chunk_0830_labeled_enc.csv",
    "chunk_0576_labeled_enc.csv","chunk_0827_labeled_enc.csv",
    "chunk_0658_labeled_enc.csv","chunk_0601_labeled_enc.csv",
]

# --------------------------------------------------------------------
def main(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    # 1️⃣  load anomaly-bearing files
    anomaly_files = load_anomaly_list(
        os.path.join(src_dir, DEFAULT_AUTO_FILE),
        MANUAL_LIST
    )
    # guard: only files that truly exist
    anomaly_files = [f for f in anomaly_files if os.path.exists(os.path.join(src_dir, f))]
    anomaly_top   = anomaly_files[:KEEP_TOP_N_ANOM]

    # 2️⃣  pick random quiet chunks
    all_chunks    = glob.glob(os.path.join(src_dir, "chunk_*_labeled_enc.csv"))
    quiet_pool    = [os.path.basename(f) for f in all_chunks if os.path.basename(f) not in anomaly_files]
    random_quiet  = random.sample(quiet_pool, KEEP_RANDOM_N) if len(quiet_pool) >= KEEP_RANDOM_N else quiet_pool

    selected = sorted(set(anomaly_top + random_quiet))
    print(f"📦 Copying {len(selected)} chunks → {dst_dir}")

    copied, missing = 0, []
    for fname in selected:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"✅ Copied {copied} chunks.")
    if missing:
        print("⚠️  Missing:", ", ".join(missing))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy anomaly-rich Phase-2 chunks.")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source directory with *_labeled_enc.csv files")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Destination directory for slim set")
    args = parser.parse_args()
    main(args.src, args.dst)
