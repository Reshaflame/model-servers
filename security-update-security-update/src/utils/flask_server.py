from flask import Flask, send_from_directory, render_template_string, request, jsonify
import os
from .auth_manager import AuthManager
from .input_validator import InputValidator
from .secure_downloader import SecureDownloader
import logging
from functools import wraps

app = Flask(__name__)
auth_manager = AuthManager()
input_validator = InputValidator()
secure_downloader = SecureDownloader()

# Setup secure logging
logging.basicConfig(
    filename='data/server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SecureServer')

MODELS_DIR = "/app/models"

def log_request(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
        return f(*args, **kwargs)
    return decorated

@app.route("/")
@log_request
def index():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    html = """
    <h2>📦 Available Models for Download:</h2>
    {% if files %}
    <ul>
    {% for file in files %}
      <li><a href="/download/{{ file }}">{{ file }}</a></li>
    {% endfor %}
    </ul>
    {% else %}
    <p>No models found yet. Check back later!</p>
    {% endif %}
    """
    return render_template_string(html, files=files)

@app.route("/download/<path:filename>")
@auth_manager.require_auth
@log_request
def download_file(filename):
    # Validate filename
    if not input_validator.validate_file_path(filename):
        return jsonify({"error": "Invalid filename"}), 400
        
    try:
        return send_from_directory(MODELS_DIR, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": "File not found"}), 404

@app.route("/login", methods=["POST"])
@log_request
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing credentials"}), 400
        
    username = input_validator.sanitize_string(data['username'])
    password = data['password']
    
    if auth_manager.verify_user(username, password):
        token = auth_manager.generate_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/register", methods=["POST"])
@log_request
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing registration data"}), 400
        
    username = input_validator.sanitize_string(data['username'])
    password = data['password']
    
    if auth_manager.create_user(username, password):
        return jsonify({"message": "User created successfully"})
    else:
        return jsonify({"error": "Username already exists"}), 400

@app.route("/upload", methods=["POST"])
@auth_manager.require_auth
@log_request
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not input_validator.validate_file_path(file.filename):
        return jsonify({"error": "Invalid filename"}), 400
        
    try:
        file_path = os.path.join(MODELS_DIR, file.filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully"})
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "Upload failed"}), 500

if __name__ == "__main__":
    # Create initial admin user if none exists
    if not os.path.exists('data/users.json'):
        auth_manager.create_user('admin', 'admin123', role='admin')
        logger.info("Created initial admin user")
    
    app.run(host='0.0.0.0', port=8888, ssl_context='adhoc')
