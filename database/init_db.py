"""
Database initialization script for AI Data Analysis Platform
"""

import os
import sys
sys.path.append('..')

from database.db_manager import db_manager

def initialize_database():
    """Initialize the database with all required tables"""
    try:
        print("🔧 Initializing database...")
        db_manager.create_tables()
        print("✅ Database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        return False

def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 Created data directory")

if __name__ == "__main__":
    create_data_directory()
    initialize_database()