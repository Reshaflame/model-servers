import os
import json
from auth_manager import AuthManager
from input_validator import InputValidator
from secure_downloader import SecureDownloader

def test_auth_system():
    print("\n=== Testing Authentication System ===")
    auth = AuthManager()
    
    # Test user creation
    print("\n1. Testing user creation...")
    success = auth.create_user("testuser", "testpass123")
    print(f"User creation {'successful' if success else 'failed'}")
    
    # Test user verification
    print("\n2. Testing user verification...")
    valid = auth.verify_user("testuser", "testpass123")
    print(f"User verification {'successful' if valid else 'failed'}")
    
    # Test token generation and verification
    print("\n3. Testing JWT token system...")
    token = auth.generate_token("testuser")
    print(f"Generated token: {token[:20]}...")
    
    payload = auth.verify_token(token)
    print(f"Token verification {'successful' if payload else 'failed'}")
    if payload:
        print(f"Token payload: {payload}")

def test_input_validation():
    print("\n=== Testing Input Validation ===")
    validator = InputValidator()
    
    # Test valid input
    print("\n1. Testing valid input...")
    valid_data = {
        'time': '1234567890',
        'username': 'test_user@domain.com',
        'computer': 'COMPUTER-01',
        'auth_type': 'NTLM',
        'logon_type': 'INTERACTIVE',
        'auth_orientation': 'INBOUND',
        'success': 'true'
    }
    is_valid = validator.validate_row(valid_data)
    print(f"Valid data validation: {'passed' if is_valid else 'failed'}")
    
    # Test invalid input
    print("\n2. Testing invalid input...")
    invalid_data = {
        'time': 'invalid_time',
        'username': 'test_user<script>alert("xss")</script>',
        'computer': '../../../etc/passwd',
        'auth_type': 'INVALID_TYPE',
        'logon_type': 'INVALID_LOGON',
        'auth_orientation': 'INVALID_ORIENTATION',
        'success': 'invalid'
    }
    is_valid = validator.validate_row(invalid_data)
    print(f"Invalid data validation: {'passed' if is_valid else 'failed'}")

def test_secure_download():
    print("\n=== Testing Secure Download System ===")
    downloader = SecureDownloader()
    
    # Create a test file
    print("\n1. Creating test file...")
    test_content = "This is a test file for security verification"
    test_file = "data/test_file.txt"
    os.makedirs("data", exist_ok=True)
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Test file encryption
    print("\n2. Testing file encryption...")
    encrypted_file = "data/test_file.enc"
    with open(test_file, "rb") as f:
        file_data = f.read()
    encrypted_data = downloader.cipher_suite.encrypt(file_data)
    with open(encrypted_file, "wb") as f:
        f.write(encrypted_data)
    print(f"File encrypted and saved to {encrypted_file}")
    
    # Test file decryption
    print("\n3. Testing file decryption...")
    decrypted_file = "data/test_file.dec"
    success = downloader.decrypt_file(encrypted_file, decrypted_file)
    print(f"File decryption {'successful' if success else 'failed'}")
    
    # Clean up test files
    print("\n4. Cleaning up test files...")
    for file in [test_file, encrypted_file, decrypted_file]:
        if os.path.exists(file):
            os.remove(file)
    print("Test files cleaned up")

def main():
    print("Starting security feature tests...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    
    # Run tests
    test_auth_system()
    test_input_validation()
    test_secure_download()
    
    print("\n=== Security Tests Completed ===")
    print("Check the following log files for detailed information:")
    print("- data/auth.log")
    print("- data/validation.log")
    print("- data/secure_download.log")

if __name__ == "__main__":
    main() 