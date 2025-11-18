import os
import json
import requests
from typing import Dict, List, Any, Optional
import pandas as pd

# Load environment variables explicitly
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("DEBUG: GLM - .env file loaded successfully")
except ImportError:
    print("DEBUG: GLM - dotenv not available, using os.environ only")

class GLMIntegration:
    """Integration with GLM-4.6 API via Agent Router for AI-powered data analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('GLM_API_KEY', '')
        self.base_url = "https://api.agentrouter.ai/v1/chat/completions"
        self.model = "glm-4.6"  # GLM-4.6 model
        
        print(f"DEBUG: GLM API Key found: {bool(self.api_key)}")
        print(f"DEBUG: GLM API Key length: {len(self.api_key) if self.api_key else 0}")
        
    def is_configured(self) -> bool:
        """Check if GLM API is properly configured"""
        return bool(self.api_key)
    
    def get_headers(self) -> Dict[str, str]:
        """Get API headers for GLM request via Agent Router"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_with_glm(self, 
                     messages: List[Dict[str, str]], 
                     data_context: Optional[Dict] = None,
                     max_tokens: int = 1000) -> str:
        """
        Chat with GLM model
        
        Args:
            messages: List of chat messages
            data_context: Optional data context for analysis
            max_tokens: Maximum tokens in response
            
        Returns:
            AI response string
        """
        if not self.is_configured():
            return "⚠️ GLM API key not configured. Please set GLM_API_KEY in your environment variables."
        
        try:
            # Prepare system message with data context
            system_message = self._prepare_system_message(data_context)
            
            # Prepare messages for API
            api_messages = [{"role": "system", "content": system_message}]
            api_messages.extend(messages)
            
            # Prepare request payload for GLM-4.6
            payload = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"GLM API Error: {response.status_code}"
                if response.text:
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('error', {}).get('message', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                return error_msg
                
        except requests.exceptions.Timeout:
            return "⏰ GLM API request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "🔌 Cannot connect to GLM API. Please check your internet connection."
        except Exception as e:
            return f"❌ GLM API Error: {str(e)}"
    
    def _prepare_system_message(self, data_context: Optional[Dict] = None) -> str:
        """Prepare system message with data context"""
        base_system = """You are GLM-4.6, an expert data analyst assistant with advanced capabilities in:
- Data analysis and visualization with deep insights
- Machine learning and statistical modeling
- Python programming with pandas, numpy, scikit-learn
- Creating comprehensive insights and recommendations from data
- Communicating complex concepts in simple, clear terms
- Advanced reasoning and pattern recognition

You are helping analyze data in an Indonesian AI Data Analysis Platform. Please respond primarily in Indonesian when appropriate.

When analyzing data:
1. Provide clear, actionable insights with deep understanding
2. Suggest specific visualizations or analyses with reasoning
3. Recommend next steps for deeper analysis
4. Be concise but thorough in your explanations
5. Use examples when helpful
6. Leverage your advanced reasoning capabilities for complex patterns"""
        
        if data_context:
            context_info = f"""
            
DATA CONTEXT:
- Dataset Shape: {data_context.get('shape', 'Unknown')}
- Columns: {data_context.get('columns', [])}
- Data Types: {data_context.get('dtypes', {})}
- Sample Data: {data_context.get('sample', 'No sample available')}
- Missing Values: {data_context.get('missing_info', 'Unknown')}
- Numeric Columns: {data_context.get('numeric_columns', [])}
- Categorical Columns: {data_context.get('categorical_columns', [])}

Please use this context to provide specific, relevant insights about the user's data."""
            
            return base_system + context_info
        
        return base_system
    
    def generate_insights(self, df: pd.DataFrame, analysis_type: str = "general") -> str:
        """
        Generate automatic insights about the dataset
        
        Args:
            df: DataFrame to analyze
            analysis_type: Type of analysis (general, patterns, recommendations)
            
        Returns:
            Generated insights string
        """
        if not self.is_configured():
            return "⚠️ GLM API key not configured. Please set GLM_API_KEY in your environment variables."
        
        try:
            # Prepare data context
            context = self._get_data_context(df)
            
            # Prepare analysis prompt
            analysis_prompts = {
                "general": f"Berdasarkan data ini, berikan insight umum tentang dataset ini. Fokus pada: 1) Karakteristik utama data, 2) Potensi masalah kualitas data, 3) Rekomendasi analisis awal.",
                "patterns": f"Identifikasi pola dan tren menarik dalam data ini. Carikan: 1) Korelasi penting, 2) Distribusi yang tidak biasa, 3) Outlier atau anomali, 4) Insight bisnis yang bisa diambil.",
                "recommendations": f"Berdasarkan data ini, berikan rekomendasi konkret untuk: 1) Analisis lanjutan yang sebaiknya dilakukan, 2) Visualisasi yang paling informatif, 3) Model machine learning yang cocok, 4) Aksi bisnis yang bisa diambil."
            }
            
            prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            
            messages = [{"role": "user", "content": prompt}]
            
            return self.chat_with_glm(messages, context, max_tokens=1500)
            
        except Exception as e:
            return f"❌ Error generating insights: {str(e)}"
    
    def _get_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract relevant context from DataFrame"""
        try:
            context = {
                'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
                'columns': df.columns.tolist()[:20],  # Limit to first 20 columns
                'dtypes': dict(list(df.dtypes.astype(str).items())[:10]),  # Limit to first 10
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()[:10],
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()[:10],
                'missing_info': dict(list(df.isnull().sum().items())[:10]),  # Limit to first 10
            }
            
            # Add sample data (first few rows)
            if len(df) > 0:
                sample_rows = min(5, len(df))
                sample_cols = min(5, len(df.columns))
                sample_data = df.iloc[:sample_rows, :sample_cols].to_string()
                context['sample'] = sample_data
            
            return context
            
        except Exception as e:
            return {'error': f'Error extracting context: {str(e)}'}

# Global GLM instance
glm_integration = GLMIntegration()

def chat_with_glm(messages: List[Dict[str, str]], 
                 data_context: Optional[Dict] = None,
                 max_tokens: int = 1000) -> str:
    """Convenience function for GLM chat"""
    return glm_integration.chat_with_glm(messages, data_context, max_tokens)

def generate_glm_insights(df: pd.DataFrame, analysis_type: str = "general") -> str:
    """Convenience function for generating GLM insights"""
    return glm_integration.generate_insights(df, analysis_type)

def is_glm_available() -> bool:
    """Check if GLM is properly configured"""
    return glm_integration.is_configured()

def get_glm_status() -> Dict[str, Any]:
    """Get GLM configuration status"""
    return {
        'configured': glm_integration.is_configured(),
        'model': glm_integration.model,
        'api_key_set': bool(glm_integration.api_key)
    }