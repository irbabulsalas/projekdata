import streamlit as st
import hashlib
import bcrypt
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime, timedelta

class AuthManager:
    _instance = None
    _current_user = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuthManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def init_session(cls):
        """Initialize authentication session"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
    
    @classmethod
    def is_authenticated(cls) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    @classmethod
    def get_current_user(cls) -> Optional[Dict]:
        """Get current user information"""
        if cls.is_authenticated():
            return {
                'id': st.session_state.get('user_id'),
                'username': st.session_state.get('username'),
                'email': st.session_state.get('email', '')
            }
        return None
    
    @classmethod
    def login(cls, username: str, password: str) -> bool:
        """Login user"""
        try:
            # For demo purposes, accept any login
            # In production, implement proper authentication
            if username and password:
                st.session_state.authenticated = True
                st.session_state.user_id = hash(username) % 10000
                st.session_state.username = username
                st.session_state.email = f"{username}@example.com"
                return True
            return False
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            return False
    
    @classmethod
    def logout(cls):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.email = None
        st.rerun()
    
    @classmethod
    def register(cls, username: str, email: str, password: str) -> bool:
        """Register new user"""
        try:
            # For demo purposes, always succeed
            # In production, implement proper user registration
            if username and email and password:
                return True
            return False
        except Exception as e:
            st.error(f"Registration failed: {str(e)}")
            return False
    
    @classmethod
    def render_auth_sidebar(cls):
        """Render authentication sidebar"""
        if not cls.is_authenticated():
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔐 Authentication")
            
            auth_tab = st.sidebar.tabs(["Login", "Register"])
            
            with auth_tab[0]:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("🔑 Login", type="primary"):
                    if cls.login(username, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            
            with auth_tab[1]:
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
                
                if st.button("📝 Register", type="primary"):
                    if new_password != confirm_password:
                        st.error("❌ Passwords don't match")
                    elif cls.register(new_username, new_email, new_password):
                        st.success("✅ Registration successful! Please login.")
                    else:
                        st.error("❌ Registration failed")
        else:
            user = cls.get_current_user()
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"👤 {user['username']}")
            st.sidebar.write(f"📧 {user['email']}")
            
            if st.sidebar.button("🚪 Logout"):
                cls.logout()

# Create global instance
auth_manager = AuthManager()