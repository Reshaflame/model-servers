import os

class CleanupUtility:

    @staticmethod
    def delete_chunks(chunk_dir):
        if not os.path.exists(chunk_dir):
            print(f"[Cleanup] Directory '{chunk_dir}' does not exist.")
            return

        files_removed = 0
        for file in os.listdir(chunk_dir):
            file_path = os.path.join(chunk_dir, file)
            if file.endswith(".csv"):
                os.remove(file_path)
                files_removed += 1

        if files_removed > 0:
            print(f"[Cleanup] Deleted {files_removed} CSV files from {chunk_dir}")
        else:
            print(f"[Cleanup] No CSV files found in {chunk_dir}")

    @staticmethod
    def delete_directory_if_empty(dir_path):
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"[Cleanup] Removed empty directory {dir_path}")

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[Cleanup] Deleted file: {file_path}")

    # --------------------------
    # ✅ Step 1: Clean raw auth
    # --------------------------
    @staticmethod
    def cleanup_raw_auth():
        CleanupUtility.delete_file("data/auth_quarter_01.txt.gz")
        print("[Cleanup] ✅ Removed raw auth_quarter_01.txt.gz after chunking.")

    # ---------------------------------------------
    # ✅ Step 2: Clean raw chunks after preprocessing
    # ---------------------------------------------
    @staticmethod
    def cleanup_raw_chunks():
        shared_chunks = "data/shared_chunks"
        CleanupUtility.delete_chunks(shared_chunks)
        CleanupUtility.delete_directory_if_empty(shared_chunks)
        print("[Cleanup] ✅ Removed raw shared chunks after labeled/unlabeled preprocessing.")

    # ------------------------------------------
    # 🧼 Optional: Wipe everything (for dev)
    # ------------------------------------------
    @staticmethod
    def cleanup_all():
        CleanupUtility.cleanup_raw_auth()
        CleanupUtility.cleanup_raw_chunks()

        unlabeled_chunks = 'data/preprocessed_unlabeled/chunks'
        CleanupUtility.delete_chunks(unlabeled_chunks)
        CleanupUtility.delete_directory_if_empty(unlabeled_chunks)

        print("[Cleanup] ✅ All raw + preprocessed chunks removed.")

if __name__ == "__main__":
    CleanupUtility.cleanup_all()
