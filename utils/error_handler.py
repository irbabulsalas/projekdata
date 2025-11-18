"""
Error Handler Module for AI Data Analysis Platform
Comprehensive error handling and logging system.
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
from streamlit import exception, error, warning, info, success
import pandas as pd


class DataAnalysisError(Exception):
    """Custom exception for data analysis errors."""
    pass


class DataProcessingError(DataAnalysisError):
    """Exception for data processing errors."""
    pass


class MLModelError(DataAnalysisError):
    """Exception for machine learning model errors."""
    pass


class TextAnalyticsError(DataAnalysisError):
    """Exception for text analytics errors."""
    pass


class GeminiAPIError(DataAnalysisError):
    """Exception for Gemini API errors."""
    pass


class CollaborationError(DataAnalysisError):
    """Exception for collaboration errors."""
    pass


class ValidationError(DataAnalysisError):
    """Exception for data validation errors."""
    pass


class ErrorHandler:
    """Centralized error handling and logging system."""
    
    def __init__(self, log_file: str = "logs/app.log"):
        """Initialize the error handler with logging configuration."""
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with context information."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.logger.error(f"Error occurred: {error_info}")
        return error_info
    
    def handle_data_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle data-related errors with user-friendly messages."""
        self.log_error(error, context)
        
        if isinstance(error, pd.errors.EmptyDataError):
            error("📊 **Data Error**: The uploaded file is empty. Please check your data and try again.")
        elif isinstance(error, pd.errors.ParserError):
            error("📊 **Parsing Error**: Unable to parse the file. Please ensure it's a valid CSV/Excel file.")
        elif isinstance(error, FileNotFoundError):
            error("📊 **File Error**: File not found. Please upload the file again.")
        elif isinstance(error, MemoryError):
            error("📊 **Memory Error**: File is too large. Please try with a smaller dataset.")
        else:
            error(f"📊 **Data Error**: {str(error)}")
    
    def handle_ml_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle machine learning errors."""
        self.log_error(error, context)
        
        if "insufficient data" in str(error).lower():
            error("🤖 **ML Error**: Insufficient data for training. Please provide more samples.")
        elif "feature" in str(error).lower():
            error("🤖 **ML Error**: Feature mismatch. Please check your feature selection.")
        elif "convergence" in str(error).lower():
            error("🤖 **ML Error**: Model failed to converge. Try adjusting hyperparameters.")
        else:
            error(f"🤖 **ML Error**: {str(error)}")
    
    def handle_text_analytics_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle text analytics errors."""
        self.log_error(error, context)
        
        if "language" in str(error).lower():
            error("📝 **Text Analytics Error**: Language detection failed. Please check your text data.")
        elif "encoding" in str(error).lower():
            error("📝 **Text Analytics Error**: Encoding issue detected. Please check file encoding.")
        else:
            error(f"📝 **Text Analytics Error**: {str(error)}")
    
    def handle_gemini_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle Gemini API errors."""
        self.log_error(error, context)
        
        if "rate limit" in str(error).lower():
            warning("⏰ **API Rate Limit**: Please wait before making another request.")
        elif "api key" in str(error).lower():
            error("🔑 **API Error**: Invalid API key. Please check your configuration.")
        elif "quota" in str(error).lower():
            warning("📊 **API Quota**: API quota exceeded. Please try again later.")
        else:
            error(f"🤖 **Gemini API Error**: {str(error)}")
    
    def handle_validation_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle validation errors."""
        self.log_error(error, context)
        
        if "required" in str(error).lower():
            error("✅ **Validation Error**: Required field is missing.")
        elif "format" in str(error).lower():
            error("✅ **Validation Error**: Invalid format. Please check your input.")
        else:
            error(f"✅ **Validation Error**: {str(error)}")


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(error_type: str = "general"):
    """Decorator for handling errors in functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit length
                    'kwargs': str(kwargs)[:200]  # Limit length
                }
                
                if error_type == "data":
                    error_handler.handle_data_error(e, context)
                elif error_type == "ml":
                    error_handler.handle_ml_error(e, context)
                elif error_type == "text":
                    error_handler.handle_text_analytics_error(e, context)
                elif error_type == "gemini":
                    error_handler.handle_gemini_error(e, context)
                elif error_type == "validation":
                    error_handler.handle_validation_error(e, context)
                elif error_type == "collaboration":
                    error_handler.log_error(e, context)
                    error(f"👥 **Collaboration Error**: {str(e)}")
                else:
                    error_handler.log_error(e, context)
                    error(f"❌ **Error**: {str(e)}")
                
                return None
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> bool:
    """Validate DataFrame requirements."""
    if df is None:
        return False
    
    if df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if len(df.columns) < min_cols:
        return False
    
    return True


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """Validate file extension."""
    if not filename:
        raise ValidationError("No filename provided")
    
    file_ext = filename.lower().split('.')[-1]
    if file_ext not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(f"File extension '{file_ext}' not allowed. Allowed: {allowed_extensions}")
    
    return True


def safe_execute(operation_name: str, func, *args, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.log_error(e, {'operation': operation_name, 'function': func.__name__})
        st.error(f"❌ Error in {operation_name}: {str(e)}")
        return None


def safe_execute_legacy(func, default_value=None, error_message: str = "Operation failed"):
    """Legacy safe execute function with error handling."""
    try:
        return func()
    except Exception as e:
        error_handler.log_error(e, {'function': func.__name__})
        warning(f"⚠️ {error_message}: {str(e)}")
        return default_value


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step_description: str = ""):
        """Update progress and show status."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        
        elapsed = datetime.now() - self.start_time
        if self.current_step > 0:
            estimated_total = elapsed * (self.total_steps / self.current_step)
            remaining = estimated_total - elapsed
            eta_str = f" (ETA: {str(remaining).split('.')[0]})"
        else:
            eta_str = ""
        
        info(f"🔄 Progress: {progress:.1%} - {step_description}{eta_str}")
    
    def complete(self, message: str = "Operation completed successfully!"):
        """Mark operation as complete."""
        elapsed = datetime.now() - self.start_time
        success(f"✅ {message} (Time: {str(elapsed).split('.')[0]})")


# Streamlit-specific error handling
def streamlit_error_boundary():
    """Context manager for Streamlit error handling."""
    try:
        yield
    except Exception as e:
        error_handler.log_error(e)
        exception(f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs for more details.")