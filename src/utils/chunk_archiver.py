import os, zipfile, math, glob

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def zip_chunks(chunk_dir, out_dir="data/archives",
               kind="labeled", max_bytes=2_000_000_000):
    """
    Bundle CSV chunk files into <2 GB zips so Runpod's Jupyter 'Download'
    button works reliably.
    """
    archive_dir = os.path.join(ROOT_DIR, out_dir)
    os.makedirs(archive_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(chunk_dir,"*.csv")))
    part, current_zip, current_size = 1, None, 0

    for f in files:
        f_size = os.path.getsize(f)
        if not current_zip or current_size + f_size > max_bytes:
            if current_zip: current_zip.close()
            zip_path = os.path.join(archive_dir,
                         f"{kind}_part{part}.zip")
            current_zip = zipfile.ZipFile(zip_path,"w",
                                          compression=zipfile.ZIP_DEFLATED)
            current_size, part = 0, part+1
        current_zip.write(f, arcname=os.path.basename(f))
        current_size += f_size
    if current_zip: current_zip.close()
    print(f"[Archiver] ✓ {part-1} archives written to {archive_dir}")
