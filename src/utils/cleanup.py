import os
import shutil

class CleanupUtility:
    @staticmethod
    def delete_chunks(chunk_dir):
        """
        Deletes all CSV chunk files from the specified directory.
        """
        if not os.path.exists(chunk_dir):
            print(f"[Cleanup] Directory '{chunk_dir}' does not exist.")
            return

        files_removed = 0
        for file in os.listdir(chunk_dir):
            file_path = os.path.join(chunk_dir, file)
            if file.endswith(".csv") and "chunk_" in file:
                os.remove(file_path)
                files_removed += 1
        
        if files_removed > 0:
            print(f"[Cleanup] Deleted {files_removed} chunk files from {chunk_dir}")
        else:
            print(f"[Cleanup] No chunk files found in {chunk_dir}")

    @staticmethod
    def delete_directory_if_empty(dir_path):
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"[Cleanup] Removed empty directory {dir_path}")

    @staticmethod
    def cleanup_all():
        CleanupUtility.delete_chunks('data/labeled_data/chunks')
        CleanupUtility.delete_chunks('data/sampled_data/chunks')
        CleanupUtility.delete_directory_if_empty('data/labeled_data/chunks')
        CleanupUtility.delete_directory_if_empty('data/sampled_data/chunks')


if __name__ == "__main__":
    CleanupUtility.cleanup_all()
