import os
import hashlib
import requests
from typing import Optional
import gdown
import logging
from cryptography.fernet import Fernet
import json

class SecureDownloader:
    def __init__(self, key_file: str = "data/encryption.key"):
        self.key_file = key_file
        self._setup_encryption()
        self._setup_logging()
        
    def _setup_encryption(self):
        """Initialize or load encryption key"""
        if not os.path.exists(self.key_file):
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            with open(self.key_file, 'wb') as f:
                f.write(key)
        else:
            with open(self.key_file, 'rb') as f:
                key = f.read()
        self.cipher_suite = Fernet(key)

    def _setup_logging(self):
        """Setup secure logging"""
        logging.basicConfig(
            filename='data/secure_download.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SecureDownloader')

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_file_integrity(self, file_path: str, expected_hash: str) -> bool:
        """Verify file integrity using hash"""
        actual_hash = self.calculate_file_hash(file_path)
        return actual_hash == expected_hash

    def secure_download(self, file_id: str, output_path: str, expected_hash: Optional[str] = None) -> bool:
        """Securely download file from Google Drive with integrity check"""
        try:
            # Create temporary path for download
            temp_path = f"{output_path}.tmp"
            
            # Download file
            self.logger.info(f"Starting secure download of {output_path}")
            gdown.download(id=file_id, output=temp_path, quiet=False)
            
            # Verify file integrity if hash provided
            if expected_hash and not self.verify_file_integrity(temp_path, expected_hash):
                self.logger.error(f"Hash verification failed for {output_path}")
                os.remove(temp_path)
                return False
            
            # Encrypt the file
            with open(temp_path, 'rb') as f:
                file_data = f.read()
            encrypted_data = self.cipher_suite.encrypt(file_data)
            
            # Save encrypted file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            self.logger.info(f"Successfully downloaded and encrypted {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during secure download: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def decrypt_file(self, encrypted_path: str, decrypted_path: str) -> bool:
        """Decrypt a previously encrypted file"""
        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"Successfully decrypted {encrypted_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during decryption: {str(e)}")
            return False 