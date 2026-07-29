from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class LoginRequest(BaseModel):
    """Schema for user login request."""
    username: str = Field(..., description="The username.")
    password: str = Field(..., description="The user's password.")

class TokenResponse(BaseModel):
    """Schema for access token response."""
    access_token: str = Field(..., description="The JWT access token.")
    token_type: str = Field("bearer", description="The token type (bearer).")

class UserResponse(BaseModel):
    """Schema for returning user data."""
    id: str = Field(..., description="Unique user ID.")
    username: str = Field(..., description="The username.")
    email: Optional[EmailStr] = Field(None, description="The user's email.")
    role: str = Field(..., description="The user's role (e.g., Admin, HR_Manager, Viewer).")
    is_active: bool = Field(True, description="Whether the user account is active.")
    created_at: str = Field(..., description="Timestamp of when the user was created.")
