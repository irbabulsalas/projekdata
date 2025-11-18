import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from openai import OpenAI

# Load environment variables explicitly
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("DEBUG: .env file loaded successfully")
except ImportError:
    print("DEBUG: dotenv not available, using os.environ only")

class OpenAIIntegration:
    """Integration with OpenAI API for AI-powered data analysis"""
    
    def __init__(self):
        # Try multiple methods to get API key
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Debug: Print API key status (remove in production)
        print(f"DEBUG: OpenAI API Key found: {bool(self.api_key)}")
        print(f"DEBUG: OpenAI API Key length: {len(self.api_key) if self.api_key else 0}")
        print(f"DEBUG: Environment variables: {list(os.environ.keys())}")
        
        try:
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            self.model = "gpt-4o-mini"  # Default OpenAI model
        except Exception as e:
            print(f"DEBUG: OpenAI client initialization error: {str(e)}")
            self.client = None
        
    def is_configured(self) -> bool:
        """Check if OpenAI API is properly configured"""
        configured = bool(self.api_key) and self.client is not None
        print(f"DEBUG: OpenAI configured status: {configured}")
        return configured
    
    def chat_with_openai(self, 
                       messages: List[Dict[str, str]], 
                       data_context: Optional[Dict] = None,
                       max_tokens: int = 1000) -> str:
        """
        Chat with OpenAI model
        
        Args:
            messages: List of chat messages
            data_context: Optional data context for analysis
            max_tokens: Maximum tokens in response
            
        Returns:
            AI response string
        """
        if not self.is_configured():
            return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your environment variables."
        
        try:
            # Prepare system message with data context
            system_message = self._prepare_system_message(data_context)
            
            # Prepare messages for API
            api_messages = [{"role": "system", "content": system_message}]
            api_messages.extend(messages)
            
            # Make API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            error_msg = f"❌ OpenAI API Error: {str(e)}"
            if "rate_limit" in str(e).lower():
                error_msg += " - Rate limit exceeded. Please try again later."
            elif "insufficient_quota" in str(e).lower():
                error_msg += " - Insufficient quota. Please check your OpenAI account."
            elif "invalid_api_key" in str(e).lower():
                error_msg += " - Invalid API key. Please check your OPENAI_API_KEY."
            return error_msg
    
    def _prepare_system_message(self, data_context: Optional[Dict] = None) -> str:
        """Prepare system message with data context"""
        base_system = """You are an expert data analyst assistant with advanced capabilities in:
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
6. Leverage your advanced reasoning capabilities for complex patterns
7. Consider business implications and practical applications"""
        
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

Please use this context to provide specific, relevant insights about user's data. Focus on practical applications and business value."""
            
            return base_system + context_info
        
        return base_system
    
    def generate_insights(self, df: pd.DataFrame, analysis_type: str = "general") -> str:
        """
        Generate automatic insights about dataset
        
        Args:
            df: DataFrame to analyze
            analysis_type: Type of analysis (general, patterns, recommendations)
            
        Returns:
            Generated insights string
        """
        if not self.is_configured():
            return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your environment variables."
        
        try:
            # Prepare data context
            context = self._get_data_context(df)
            
            # Prepare analysis prompt
            analysis_prompts = {
                "general": f"Berdasarkan data ini, berikan insight umum tentang dataset. Fokus pada: 1) Karakteristik utama data, 2) Potensi masalah kualitas data, 3) Rekomendasi analisis awal, 4) Implikasi bisnis yang bisa diambil.",
                "patterns": f"Identifikasi pola dan tren menarik dalam data ini. Carikan: 1) Korelasi penting dan signifikansinya, 2) Distribusi yang tidak biasa, 3) Outlier atau anomali, 4) Insight bisnis yang actionable, 5) Rekomendasi visualisasi yang efektif.",
                "recommendations": f"Berdasarkan data ini, berikan rekomendasi konkret untuk: 1) Analisis lanjutan yang sebaiknya dilakukan, 2) Visualisasi yang paling informatif, 3) Model machine learning yang cocok, 4) Aksi bisnis yang bisa diambil, 5) Metrik KPI yang sebaiknya dipantau."
            }
            
            prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            
            messages = [{"role": "user", "content": prompt}]
            
            return self.chat_with_openai(messages, context, max_tokens=1500)
            
        except Exception as e:
            return f"❌ Error generating insights: {str(e)}"
    
    def _get_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract relevant context from DataFrame"""
        try:
            context = {
                'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
                'columns': df.columns.tolist()[:20],  # Limit to first 20 columns
                'dtypes': df.dtypes.astype(str).to_dict()[:10],  # Limit to first 10
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()[:10],
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()[:10],
                'missing_info': df.isnull().sum().to_dict()[:10],  # Limit to first 10
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

# Global OpenAI instance
openai_integration = OpenAIIntegration()

def chat_with_openai(messages: List[Dict[str, str]], 
                   data_context: Optional[Dict] = None,
                   max_tokens: int = 1000) -> str:
    """Convenience function for OpenAI chat"""
    return openai_integration.chat_with_openai(messages, data_context, max_tokens)

def generate_openai_insights(df: pd.DataFrame, analysis_type: str = "general") -> str:
    """Convenience function for generating OpenAI insights"""
    return openai_integration.generate_insights(df, analysis_type)

def is_openai_available() -> bool:
    """Check if OpenAI is properly configured"""
    return openai_integration.is_configured()

def get_openai_status() -> Dict[str, Any]:
    """Get OpenAI configuration status"""
    return {
        'configured': openai_integration.is_configured(),
        'model': openai_integration.model,
        'api_key_set': bool(openai_integration.api_key)
    }