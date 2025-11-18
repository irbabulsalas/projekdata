import sqlite3
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.db_path = "data/app.db"
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        if not os.path.exists("data"):
            os.makedirs("data")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Create models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                model_type TEXT,
                algorithm TEXT,
                metrics TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def create_tables(self):
        """Create all tables (called during initialization)"""
        self.ensure_database_exists()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a query and return results"""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return results
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def execute_single(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """Execute a query and return single result"""
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def insert_user(self, username: str, email: str, password_hash: str) -> Optional[int]:
        """Insert new user"""
        query = '''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        '''
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (username, email, password_hash))
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return user_id
        except Exception as e:
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        query = 'SELECT * FROM users WHERE username = ?'
        return self.execute_single(query, (username,))
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        query = 'SELECT * FROM users WHERE email = ?'
        return self.execute_single(query, (email,))
    
    def insert_project(self, project_id: str, name: str, description: str, is_public: bool, user_id: int) -> bool:
        """Insert new project"""
        query = '''
            INSERT INTO projects (id, name, description, is_public, user_id)
            VALUES (?, ?, ?, ?, ?)
        '''
        try:
            self.execute_query(query, (project_id, name, description, is_public, user_id))
            return True
        except Exception:
            return False
    
    def get_user_projects(self, user_id: int) -> List[Dict]:
        """Get all projects for a user"""
        query = '''
            SELECT * FROM projects 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        '''
        return self.execute_query(query, (user_id,))
    
    def insert_dataset(self, dataset_id: str, project_id: str, name: str, rows: int, columns: int, file_path: str) -> bool:
        """Insert new dataset"""
        query = '''
            INSERT INTO datasets (id, project_id, name, rows, columns, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        try:
            self.execute_query(query, (dataset_id, project_id, name, rows, columns, file_path))
            return True
        except Exception:
            return False
    
    def get_project_datasets(self, project_id: str) -> List[Dict]:
        """Get all datasets for a project"""
        query = '''
            SELECT * FROM datasets 
            WHERE project_id = ? 
            ORDER BY created_at DESC
        '''
        return self.execute_query(query, (project_id,))
    
    def insert_model(self, model_id: str, project_id: str, name: str, model_type: str, algorithm: str, metrics: str, file_path: str) -> bool:
        """Insert new model"""
        query = '''
            INSERT INTO models (id, project_id, name, model_type, algorithm, metrics, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        try:
            self.execute_query(query, (model_id, project_id, name, model_type, algorithm, metrics, file_path))
            return True
        except Exception:
            return False
    
    def get_project_models(self, project_id: str) -> List[Dict]:
        """Get all models for a project"""
        query = '''
            SELECT * FROM models 
            WHERE project_id = ? 
            ORDER BY created_at DESC
        '''
        return self.execute_query(query, (project_id,))

# Create global instance
db_manager = DatabaseManager()