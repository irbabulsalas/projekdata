import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import joblib

class SessionManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_project(self, name: str, description: str = "", is_public: bool = False) -> Optional[str]:
        """Save project metadata"""
        try:
            project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            project_data = {
                'id': project_id,
                'name': name,
                'description': description,
                'is_public': is_public,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'user_id': st.session_state.get('user_id', 'demo_user')
            }
            
            # Save to file (in production, use database)
            projects_file = os.path.join(self.data_dir, "projects.json")
            projects = []
            
            if os.path.exists(projects_file):
                with open(projects_file, 'r') as f:
                    projects = json.load(f)
            
            projects.append(project_data)
            
            with open(projects_file, 'w') as f:
                json.dump(projects, f, indent=2)
            
            return project_id
        except Exception as e:
            st.error(f"Failed to save project: {str(e)}")
            return None
    
    def save_dataset(self, project_id: str, name: str, df: pd.DataFrame) -> Optional[str]:
        """Save dataset"""
        try:
            dataset_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save dataframe
            dataset_path = os.path.join(self.data_dir, f"{dataset_id}.csv")
            df.to_csv(dataset_path, index=False)
            
            # Save metadata
            dataset_data = {
                'id': dataset_id,
                'project_id': project_id,
                'name': name,
                'rows': len(df),
                'columns': len(df.columns),
                'file_path': dataset_path,
                'created_at': datetime.now().isoformat()
            }
            
            datasets_file = os.path.join(self.data_dir, "datasets.json")
            datasets = []
            
            if os.path.exists(datasets_file):
                with open(datasets_file, 'r') as f:
                    datasets = json.load(f)
            
            datasets.append(dataset_data)
            
            with open(datasets_file, 'w') as f:
                json.dump(datasets, f, indent=2)
            
            return dataset_id
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            return None
    
    def save_model(self, project_id: str, name: str, model: Any, model_type: str, algorithm: str, metrics: Dict) -> Optional[str]:
        """Save trained model"""
        try:
            model_id = f"mdl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model
            model_path = os.path.join(self.data_dir, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            
            # Save metadata
            model_data = {
                'id': model_id,
                'project_id': project_id,
                'name': name,
                'model_type': model_type,
                'algorithm': algorithm,
                'metrics': metrics,
                'file_path': model_path,
                'created_at': datetime.now().isoformat()
            }
            
            models_file = os.path.join(self.data_dir, "models.json")
            models = []
            
            if os.path.exists(models_file):
                with open(models_file, 'r') as f:
                    models = json.load(f)
            
            models.append(model_data)
            
            with open(models_file, 'w') as f:
                json.dump(models, f, indent=2)
            
            return model_id
        except Exception as e:
            st.error(f"Failed to save model: {str(e)}")
            return None
    
    def list_user_projects(self) -> List[Dict]:
        """List user's projects"""
        try:
            projects_file = os.path.join(self.data_dir, "projects.json")
            
            if not os.path.exists(projects_file):
                return []
            
            with open(projects_file, 'r') as f:
                projects = json.load(f)
            
            user_id = st.session_state.get('user_id', 'demo_user')
            user_projects = [p for p in projects if p.get('user_id') == user_id]
            
            return sorted(user_projects, key=lambda x: x.get('created_at', ''), reverse=True)
        except Exception as e:
            st.error(f"Failed to list projects: {str(e)}")
            return []
    
    def list_user_datasets(self) -> List[Dict]:
        """List user's datasets"""
        try:
            datasets_file = os.path.join(self.data_dir, "datasets.json")
            
            if not os.path.exists(datasets_file):
                return []
            
            with open(datasets_file, 'r') as f:
                datasets = json.load(f)
            
            # Get user's project IDs
            user_projects = self.list_user_projects()
            user_project_ids = {p['id'] for p in user_projects}
            
            user_datasets = [d for d in datasets if d.get('project_id') in user_project_ids]
            
            return sorted(user_datasets, key=lambda x: x.get('created_at', ''), reverse=True)
        except Exception as e:
            st.error(f"Failed to list datasets: {str(e)}")
            return []
    
    def list_user_models(self) -> List[Dict]:
        """List user's models"""
        try:
            models_file = os.path.join(self.data_dir, "models.json")
            
            if not os.path.exists(models_file):
                return []
            
            with open(models_file, 'r') as f:
                models = json.load(f)
            
            # Get user's project IDs
            user_projects = self.list_user_projects()
            user_project_ids = {p['id'] for p in user_projects}
            
            user_models = [m for m in models if m.get('project_id') in user_project_ids]
            
            return sorted(user_models, key=lambda x: x.get('created_at', ''), reverse=True)
        except Exception as e:
            st.error(f"Failed to list models: {str(e)}")
            return []
    
    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset by ID"""
        try:
            datasets_file = os.path.join(self.data_dir, "datasets.json")
            
            if not os.path.exists(datasets_file):
                return None
            
            with open(datasets_file, 'r') as f:
                datasets = json.load(f)
            
            dataset = next((d for d in datasets if d.get('id') == dataset_id), None)
            
            if dataset and os.path.exists(dataset['file_path']):
                return pd.read_csv(dataset['file_path'])
            
            return None
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")
            return None
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load model by ID"""
        try:
            models_file = os.path.join(self.data_dir, "models.json")
            
            if not os.path.exists(models_file):
                return None
            
            with open(models_file, 'r') as f:
                models = json.load(f)
            
            model = next((m for m in models if m.get('id') == model_id), None)
            
            if model and os.path.exists(model['file_path']):
                return joblib.load(model['file_path'])
            
            return None
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

# Create global instance
session_manager = SessionManager()