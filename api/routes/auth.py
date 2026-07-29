from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.auth.auth_service import auth_service
from api.auth.dependencies import get_current_user
from api.auth.jwt_handler import create_access_token
from api.database.session import get_db
from api.core.logger import logger
from api.models.auth_schemas import TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> TokenResponse:
    """
    Authenticate user and return a JWT access token.
    """
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    logger.info(f"Successfully generated JWT token for user: {user['username']}")

    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Returns the profile information of the currently authenticated user.
    """
    return current_user
