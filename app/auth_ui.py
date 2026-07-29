import requests
import streamlit as st
from api_client import BACKEND_URL

def login(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/auth/login",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            # Fetch profile
            profile_response = requests.get(
                f"{BACKEND_URL}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if profile_response.status_code == 200:
                profile = profile_response.json()
                st.session_state["access_token"] = token
                st.session_state["username"] = profile.get("username")
                st.session_state["role"] = profile.get("role")
                return True
            else:
                st.error("Failed to fetch user profile.")
                return False
        else:
            st.error("Invalid username or password.")
            return False
    except requests.RequestException as e:
        st.error(f"Authentication service unavailable: {e}")
        return False


def render_login_page():
    st.markdown("<h2 style='text-align: center;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter your credentials to access the platform.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if login(username, password):
                        st.success("Login successful!")
                        st.rerun()
                else:
                    st.warning("Please enter both username and password.")

        with st.expander("Forgot Password?"):
            st.info("""
For this local HR Analytics platform, password self-reset is not available.

Please contact the System Administrator to reset your password.

**Default Local Administrator:**
Username: admin

If this is a fresh local installation, refer to the project README or the database initialization script for the default credentials.

Future versions will support secure email-based password recovery.
            """)

def logout():
    st.session_state.pop("access_token", None)
    st.session_state.pop("username", None)
    st.session_state.pop("role", None)
    st.rerun()

def render_logout_button():
    if "access_token" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.write(f"Logged in as **{st.session_state.get('username')}** ({st.session_state.get('role')})")
        if st.sidebar.button("Logout", use_container_width=True):
            logout()
