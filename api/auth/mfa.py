"""
Multi-Factor Authentication (MFA) implementation for MCP Yggdrasil.
Supports TOTP (Time-based One-Time Password) with QR code generation.
"""
import pyotp
import qrcode
import base64
from io import BytesIO
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import secrets
import logging
from dataclasses import dataclass
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class MFASetup:
    """MFA setup information for a user."""
    user_id: str
    secret: str
    qr_code: str  # Base64 encoded QR code image
    backup_codes: list[str]
    created_at: datetime


class MultiFactorAuth:
    """
    Multi-Factor Authentication manager using TOTP.
    Provides secure 2FA with QR codes and backup codes.
    """
    
    def __init__(self):
        self.issuer_name = "MCP Yggdrasil"
        self.backup_code_count = 10
        self.backup_code_length = 8
        self.secret_length = 32  # Base32 encoded length
    
    def generate_user_secret(self) -> str:
        """Generate a unique TOTP secret for a user."""
        return pyotp.random_base32(length=self.secret_length)
    
    def generate_backup_codes(self) -> list[str]:
        """Generate backup codes for account recovery."""
        codes = []
        for _ in range(self.backup_code_count):
            # Generate alphanumeric backup codes
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
                          for _ in range(self.backup_code_length))
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        return codes
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """
        Generate QR code for authenticator app setup.
        Returns base64 encoded PNG image.
        """
        try:
            # Create TOTP URI for authenticator apps
            totp = pyotp.TOTP(secret)
            totp_uri = totp.provisioning_uri(
                name=user_email,
                issuer_name=self.issuer_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            qr_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{qr_base64}"
            
        except Exception as e:
            logger.error(f"Error generating QR code: {e}")
            raise ValueError("Failed to generate QR code") from e
    
    def setup_mfa(self, user_id: str, user_email: str) -> MFASetup:
        """
        Set up MFA for a user.
        Returns setup information including secret, QR code, and backup codes.
        """
        # Generate secret
        secret = self.generate_user_secret()
        
        # Generate QR code
        qr_code = self.generate_qr_code(user_email, secret)
        
        # Generate backup codes
        backup_codes = self.generate_backup_codes()
        
        # Create setup info
        setup = MFASetup(
            user_id=user_id,
            secret=secret,
            qr_code=qr_code,
            backup_codes=backup_codes,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"MFA setup completed for user {user_id}")
        return setup
    
    def verify_token(self, secret: str, token: str, window: int = 1) -> bool:
        """
        Verify a TOTP token.
        
        Args:
            secret: User's TOTP secret
            token: 6-digit token from authenticator app
            window: Time window tolerance (default 1 = ±30 seconds)
        
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Remove any spaces or dashes from token
            clean_token = token.replace(' ', '').replace('-', '')
            
            # Verify token
            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(clean_token, valid_window=window)
            
            if is_valid:
                logger.info("TOTP token verified successfully")
            else:
                logger.warning("Invalid TOTP token provided")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying TOTP token: {e}")
            return False
    
    def verify_backup_code(self, stored_codes: list[str], provided_code: str) -> Tuple[bool, Optional[str]]:
        """
        Verify a backup code.
        
        Args:
            stored_codes: List of valid backup codes
            provided_code: Code provided by user
        
        Returns:
            Tuple of (is_valid, used_code)
        """
        # Clean the provided code
        clean_code = provided_code.upper().replace(' ', '')
        
        # Check against stored codes
        for code in stored_codes:
            if code.replace('-', '') == clean_code.replace('-', ''):
                logger.info("Backup code verified successfully")
                return True, code
        
        logger.warning("Invalid backup code provided")
        return False, None
    
    def generate_recovery_token(self, user_id: str, expires_in: int = 3600) -> str:
        """
        Generate a temporary recovery token for account recovery.
        
        Args:
            user_id: User ID
            expires_in: Token expiry in seconds (default 1 hour)
        
        Returns:
            Recovery token
        """
        # In production, this would create a JWT or similar secure token
        # For now, we'll create a simple token
        token_data = {
            'user_id': user_id,
            'purpose': 'mfa_recovery',
            'expires': (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
        }
        
        # This is a placeholder - in production use proper JWT
        token = base64.urlsafe_b64encode(
            f"{user_id}:{datetime.utcnow().timestamp()}:{secrets.token_urlsafe(32)}".encode()
        ).decode()
        
        logger.info(f"Recovery token generated for user {user_id}")
        return token
    
    def get_current_totp(self, secret: str) -> str:
        """Get current TOTP code (for testing/debugging only)."""
        totp = pyotp.TOTP(secret)
        return totp.now()
    
    def is_mfa_required(self, user_role: str) -> bool:
        """
        Determine if MFA is required for a user role.
        
        Args:
            user_role: User's role
        
        Returns:
            True if MFA is required
        """
        # Admin and curator roles require MFA
        mfa_required_roles = ['admin', 'curator']
        
        # Check if MFA is globally enabled
        if not settings.enable_monitoring:  # Using monitoring as proxy for security features
            return False
        
        return user_role.lower() in mfa_required_roles


class MFAValidator:
    """Validator for MFA operations."""
    
    @staticmethod
    def validate_token_format(token: str) -> bool:
        """Validate TOTP token format (6 digits)."""
        clean_token = token.replace(' ', '').replace('-', '')
        return len(clean_token) == 6 and clean_token.isdigit()
    
    @staticmethod
    def validate_backup_code_format(code: str) -> bool:
        """Validate backup code format (XXXX-XXXX)."""
        clean_code = code.replace(' ', '').upper()
        
        # Check format
        if len(clean_code) == 9 and clean_code[4] == '-':
            parts = clean_code.split('-')
            return all(part.isalnum() and len(part) == 4 for part in parts)
        
        # Also accept without dash
        return len(clean_code.replace('-', '')) == 8 and clean_code.replace('-', '').isalnum()
    
    @staticmethod
    def validate_secret(secret: str) -> bool:
        """Validate TOTP secret format."""
        try:
            # Check if it's valid base32
            pyotp.TOTP(secret).now()
            return True
        except:
            return False


# Global MFA instance
mfa_manager = MultiFactorAuth()
mfa_validator = MFAValidator()