from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from api.auth.auth_service import auth_service
from api.auth.jwt_handler import verify_access_token
from api.models.auth_schemas import UserResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """
    Dependency to validate the JWT token and return the current user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_access_token(token)
    if payload is None:
        raise credentials_exception
        
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
        
    user_dict = auth_service.get_user_by_username(username)
    if user_dict is None or not user_dict.get("is_active"):
        raise credentials_exception
        
    return UserResponse(**user_dict)
