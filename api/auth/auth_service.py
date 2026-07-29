import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from api.auth.password import hash_password, verify_password
from api.core.logger import logger
from api.models.auth_schemas import UserResponse
from api.services.db_service import db_service


class AuthService:
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[dict]:
        """
        Retrieves a user dictionary by their username.
        """
        return db_service.get_user_by_username(db, username)

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[dict]:
        """
        Authenticates a user and returns their user dict if successful.
        """
        user = AuthService.get_user_by_username(db, username)
        if not user:
            logger.warning(f"Authentication failed: User '{username}' not found.")
            return None
        
        if not verify_password(password, user["hashed_password"]):
            logger.warning(f"Authentication failed: Incorrect password for '{username}'.")
            return None
            
        return user

    @staticmethod
    def create_default_admin(db: Session) -> None:
        """
        Creates a default administrator account if one does not already exist.
        """
        if not AuthService.get_user_by_username(db, "admin"):
            admin_user = {
                "id": str(uuid.uuid4()),
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": hash_password("Admin@123"),
                "role": "Admin",
                "is_active": True,
                "created_at": datetime.now().isoformat()
            }
            db_service.create_user(db, admin_user)
            logger.info("Default admin user created.")


auth_service = AuthService()
