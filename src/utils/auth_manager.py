import os
import jwt
import time
import hashlib
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

class AuthManager:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.urandom(32).hex()
        self._setup_logging()
        self._load_users()
        self.role_permissions = {
            'api_client': ['read', 'write', 'api_access'],
            'frontend_client': ['read', 'frontend_access'],
            'investigator': ['read', 'write', 'investigate', 'admin_access'],
            'user': ['read']
        }
        
    def _setup_logging(self):
        """Setup secure logging"""
        logging.basicConfig(
            filename='data/auth.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AuthManager')

    def _load_users(self):
        """Load or create users database"""
        self.users_file = 'data/users.json'
        if not os.path.exists(self.users_file):
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            self.users = {}
            self._save_users()
        else:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)

    def _save_users(self):
        """Save users database"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, role: str = 'user', permissions: List[str] = None) -> bool:
        """Create a new user with specified role and permissions"""
        if username in self.users:
            self.logger.warning(f"User creation failed: {username} already exists")
            return False
        
        if role not in self.role_permissions:
            self.logger.warning(f"Invalid role: {role}")
            return False
            
        self.users[username] = {
            'password': self.hash_password(password),
            'role': role,
            'permissions': permissions or self.role_permissions[role],
            'created_at': datetime.now().isoformat()
        }
        self._save_users()
        self.logger.info(f"Created new user: {username} with role: {role}")
        return True

    def verify_user(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        if username not in self.users:
            return False
        return self.users[username]['password'] == self.hash_password(password)

    def generate_token(self, username: str) -> str:
        """Generate JWT token for authenticated user"""
        user = self.users[username]
        payload = {
            'username': username,
            'role': user['role'],
            'permissions': user.get('permissions', self.role_permissions[user['role']]),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token verification failed: Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Token verification failed: Invalid token")
            return None

    def require_auth(self, f):
        """Decorator for requiring authentication"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'message': 'Missing token'}), 401
            
            token = token.split(' ')[-1]  # Remove 'Bearer ' prefix
            payload = self.verify_token(token)
            if not payload:
                return jsonify({'message': 'Invalid token'}), 401
            
            return f(*args, **kwargs)
        return decorated

    def require_role(self, role: str):
        """Decorator for requiring specific role"""
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                token = request.headers.get('Authorization')
                if not token:
                    return jsonify({'message': 'Missing token'}), 401
                
                token = token.split(' ')[-1]
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'message': 'Invalid token'}), 401
                
                if payload['role'] != role:
                    return jsonify({'message': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            return decorated
        return decorator

    def require_permission(self, permission: str):
        """Decorator for requiring specific permission"""
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                token = request.headers.get('Authorization')
                if not token:
                    return jsonify({'message': 'Missing token'}), 401
                
                token = token.split(' ')[-1]
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'message': 'Invalid token'}), 401
                
                if permission not in payload.get('permissions', []):
                    return jsonify({'message': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            return decorated
        return decorator 