#!/usr/bin/env python3
"""
Security Middleware and Authentication
OAuth2/JWT authentication, encryption, and audit logging for MCP Server
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import json
import hashlib
import secrets
from enum import Enum

from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os
import redis
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# MODELS AND ENUMS
# ================================

class UserRole(str, Enum):
    """User roles with different permissions"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    READ_ONLY = "read_only"


class Permission(str, Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SCRAPE = "scrape"
    MAINTAIN = "maintain"


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[Permission] = []


class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str
    salt: str


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = []


class AuditLogEntry(BaseModel):
    """Audit log entry"""
    id: str
    user_id: Optional[str]
    username: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


# ================================
# SECURITY CONFIGURATION
# ================================

class SecurityConfig:
    """Security configuration manager"""
    
    def __init__(self, config_path: str = "config/security.yaml"):
        """Initialize security configuration"""
        self.config = self.load_config(config_path)
        self.secret_key = self.config['authentication']['secret_key']
        self.algorithm = self.config['authentication']['algorithm']
        self.access_token_expire_minutes = self.config['authentication']['access_token_expire_minutes']
        self.refresh_token_expire_days = self.config['authentication']['refresh_token_expire_days']
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN, Permission.SCRAPE, Permission.MAINTAIN],
            UserRole.DEVELOPER: [Permission.READ, Permission.WRITE, Permission.SCRAPE],
            UserRole.USER: [Permission.READ, Permission.WRITE],
            UserRole.READ_ONLY: [Permission.READ]
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load security configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Security config {config_path} not found, using defaults")
            return {
                'authentication': {
                    'secret_key': secrets.token_urlsafe(32),
                    'algorithm': 'HS256',
                    'access_token_expire_minutes': 30,
                    'refresh_token_expire_days': 7
                },
                'authorization': {
                    'require_approval': ['write', 'delete', 'admin'],
                    'roles': {
                        'admin': ['read', 'write', 'delete', 'admin'],
                        'developer': ['read', 'write'],
                        'user': ['read']
                    }
                },
                'encryption': {
                    'algorithm': 'AES-256-GCM',
                    'key_rotation_days': 90
                },
                'rate_limiting': {
                    'requests_per_minute': 100,
                    'burst_size': 20
                },
                'audit': {
                    'enabled': True,
                    'log_file': 'logs/audit.log',
                    'retention_days': 365
                }
            }
    
    def get_user_permissions(self, role: UserRole) -> List[Permission]:
        """Get permissions for user role"""
        return self.role_permissions.get(role, [])


# ================================
# PASSWORD HANDLING
# ================================

class PasswordManager:
    """Secure password handling"""
    
    def __init__(self):
        """Initialize password manager"""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> Tuple[str, str]:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        hashed = self.pwd_context.hash(password + salt)
        return hashed, salt
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate secure random password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))


# ================================
# JWT TOKEN HANDLING
# ================================

class TokenManager:
    """JWT token management"""
    
    def __init__(self, security_config: SecurityConfig):
        """Initialize token manager"""
        self.config = security_config
        self.redis_client = None
        self.connect_redis()
    
    def connect_redis(self):
        """Connect to Redis for token storage"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for token storage")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Tokens will not be cached.")
            self.redis_client = None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        
        # Store token in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"access_token:{encoded_jwt}",
                    int(expires_delta.total_seconds() if expires_delta else self.config.access_token_expire_minutes * 60),
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Failed to cache token in Redis: {e}")
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        
        # Store refresh token in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"refresh_token:{encoded_jwt}",
                    int(timedelta(days=self.config.refresh_token_expire_days).total_seconds()),
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Failed to cache refresh token in Redis: {e}")
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise JWTError("Invalid token type")
            
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")
            permissions: List[str] = payload.get("permissions", [])
            
            if username is None or user_id is None:
                raise JWTError("Invalid token payload")
            
            # Check if token is blacklisted (if Redis is available)
            if self.redis_client:
                try:
                    blacklisted = self.redis_client.get(f"blacklist:{token}")
                    if blacklisted:
                        raise JWTError("Token has been revoked")
                except Exception as e:
                    logger.warning(f"Failed to check token blacklist: {e}")
            
            return TokenData(
                username=username,
                user_id=user_id,
                role=role,
                permissions=permissions
            )
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding it to blacklist"""
        if not self.redis_client:
            return False
        
        try:
            # Get token expiration
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            exp = payload.get("exp")
            
            if exp:
                ttl = exp - datetime.utcnow().timestamp()
                if ttl > 0:
                    self.redis_client.setex(f"blacklist:{token}", int(ttl), "revoked")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False


# ================================
# USER MANAGEMENT
# ================================

class UserManager:
    """User management system"""
    
    def __init__(self, security_config: SecurityConfig):
        """Initialize user manager"""
        self.config = security_config
        self.password_manager = PasswordManager()
        self.users_file = Path("data/users.json")
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create default admin user if no users exist
        if not self.users_file.exists():
            self.create_default_admin()
    
    def create_default_admin(self):
        """Create default admin user"""
        default_password = self.password_manager.generate_secure_password()
        hashed_password, salt = self.password_manager.get_password_hash(default_password)
        
        admin_user = UserInDB(
            id="admin-001",
            username="admin",
            email="admin@mcp-server.local",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.utcnow(),
            hashed_password=hashed_password,
            salt=salt,
            permissions=self.config.get_user_permissions(UserRole.ADMIN)
        )
        
        users_data = {"admin": admin_user.dict()}
        
        with open(self.users_file, 'w') as f:
            json.dump(users_data, f, indent=2, default=str)
        
        # Save password to secure file for initial setup
        password_file = Path("data/admin_password.txt")
        with open(password_file, 'w') as f:
            f.write(f"Default admin password: {default_password}\n")
            f.write("Please change this password after first login!\n")
        
        logger.info(f"Created default admin user. Password saved to {password_file}")
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        try:
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            user_data = users_data.get(username)
            if user_data:
                # Convert datetime strings back to datetime objects
                if isinstance(user_data['created_at'], str):
                    user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                if user_data.get('last_login') and isinstance(user_data['last_login'], str):
                    user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                
                return UserInDB(**user_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user credentials"""
        user = self.get_user(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.password_manager.verify_password(password + user.salt, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.update_user(user)
        
        return user
    
    def create_user(self, username: str, email: str, password: str, role: UserRole, full_name: Optional[str] = None) -> UserInDB:
        """Create new user"""
        if self.get_user(username):
            raise ValueError("User already exists")
        
        hashed_password, salt = self.password_manager.get_password_hash(password)
        
        user = UserInDB(
            id=f"user-{secrets.token_hex(8)}",
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            is_active=True,
            created_at=datetime.utcnow(),
            hashed_password=hashed_password,
            salt=salt,
            permissions=self.config.get_user_permissions(role)
        )
        
        # Save user
        try:
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
        except FileNotFoundError:
            users_data = {}
        
        users_data[username] = user.dict()
        
        with open(self.users_file, 'w') as f:
            json.dump(users_data, f, indent=2, default=str)
        
        logger.info(f"Created user: {username}")
        return user
    
    def update_user(self, user: UserInDB) -> bool:
        """Update user information"""
        try:
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            users_data[user.username] = user.dict()
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = self.authenticate_user(username, old_password)
        if not user:
            return False
        
        hashed_password, salt = self.password_manager.get_password_hash(new_password)
        user.hashed_password = hashed_password
        user.salt = salt
        
        return self.update_user(user)


# ================================
# ENCRYPTION
# ================================

class EncryptionManager:
    """Data encryption management"""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption manager"""
        self.master_key = master_key or os.environ.get('MASTER_KEY')
        if not self.master_key:
            self.master_key = self.generate_master_key()
        
        self.fernet = self.create_fernet_cipher(self.master_key)
    
    def generate_master_key(self) -> str:
        """Generate new master key"""
        key = Fernet.generate_key()
        
        # Save to secure location
        key_file = Path("data/master.key")
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
        
        logger.info(f"Generated new master key: {key_file}")
        return key.decode()
    
    def create_fernet_cipher(self, key: str) -> Fernet:
        """Create Fernet cipher from key"""
        if isinstance(key, str):
            key = key.encode()
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'mcp_server_salt',  # In production, use random salt
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key = base64.urlsafe_b64encode(kdf.derive(key))
        return Fernet(derived_key)
    
    def encrypt_data(self, data: Union[str, Dict, List]) -> str:
        """Encrypt data"""
        if not isinstance(data, str):
            data = json.dumps(data)
        
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            data_str = decrypted.decode()
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return data_str
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt file and return encrypted file path"""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        encrypted_path = file_path.with_suffix(file_path.suffix + '.enc')
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt file and return decrypted file path"""
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        if output_path is None:
            output_path = encrypted_file_path.with_suffix('')
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path


# ================================
# AUDIT LOGGING
# ================================

class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        """Initialize audit logger"""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def log_action(
        self,
        user_id: Optional[str],
        username: Optional[str],
        action: str,
        resource: str,
        details: Dict[str, Any],
        ip_address: str,
        user_agent: str,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log user action"""
        entry = AuditLogEntry(
            id=secrets.token_hex(16),
            user_id=user_id,
            username=username,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            success=success,
            error_message=error_message
        )
        
        # Log to file
        log_message = json.dumps(entry.dict(), default=str)
        self.logger.info(log_message)
        
        # Store in database for querying (optional)
        await self.store_audit_entry(entry)
    
    async def store_audit_entry(self, entry: AuditLogEntry):
        """Store audit entry in database"""
        try:
            # Could store in Neo4j, PostgreSQL, or other database
            # For now, just file logging
            pass
        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")
    
    def search_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        username: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Search audit logs"""
        entries = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.split(' - INFO - ', 1)[1])
                        entry = AuditLogEntry(**data)
                        
                        # Apply filters
                        if start_date and entry.timestamp < start_date:
                            continue
                        if end_date and entry.timestamp > end_date:
                            continue
                        if username and entry.username != username:
                            continue
                        if action and entry.action != action:
                            continue
                        
                        entries.append(entry)
                        
                        if len(entries) >= limit:
                            break
                            
                    except (json.JSONDecodeError, ValueError):
                        continue
                        
        except FileNotFoundError:
            pass
        
        return list(reversed(entries))  # Most recent first


# ================================
# MIDDLEWARE
# ================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request processing"""
    
    def __init__(self, app, audit_logger: AuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware"""
        start_time = datetime.utcnow()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Process request
        response = await call_next(request)
        
        # Log request
        if self.should_log_request(request):
            await self.audit_logger.log_action(
                user_id=getattr(request.state, 'user_id', None),
                username=getattr(request.state, 'username', None),
                action=f"{request.method} {request.url.path}",
                resource=request.url.path,
                details={
                    "method": request.method,
                    "query_params": dict(request.query_params),
                    "status_code": response.status_code,
                    "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
                },
                ip_address=client_ip,
                user_agent=user_agent,
                success=response.status_code < 400
            )
        
        return response
    
    def should_log_request(self, request: Request) -> bool:
        """Determine if request should be logged"""
        # Skip health checks and static files
        skip_paths = ["/health", "/docs", "/openapi.json", "/favicon.ico"]
        return not any(request.url.path.startswith(path) for path in skip_paths)


# ================================
# AUTHENTICATION DEPENDENCIES
# ================================

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
bearer_scheme = HTTPBearer()

# Global instances (would be initialized in main.py)
security_config = SecurityConfig()
token_manager = TokenManager(security_config)
user_manager = UserManager(security_config)
audit_logger = AuditLogger()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token"""
    token_data = token_manager.verify_token(token)
    user = user_manager.get_user(token_data.username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(**user.dict())


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    
    return permission_checker


def require_role(role: UserRole):
    """Dependency to require specific role"""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return current_user
    
    return role_checker


# Example usage in routes:
# @app.post("/admin/users")
# async def create_user(
#     user_data: UserCreate,
#     current_user: User = Depends(require_permission(Permission.ADMIN))
# ):
#     ...


if __name__ == "__main__":
    # Test security components
    print("Testing security components...")
    
    # Test password hashing
    pm = PasswordManager()
    password = "test_password"
    hashed, salt = pm.get_password_hash(password)
    print(f"Password verification: {pm.verify_password(password + salt, hashed)}")
    
    # Test encryption
    em = EncryptionManager()
    test_data = {"sensitive": "data", "user_id": 123}
    encrypted = em.encrypt_data(test_data)
    decrypted = em.decrypt_data(encrypted)
    print(f"Encryption test: {test_data == decrypted}")
    
    print("Security components initialized successfully!")
