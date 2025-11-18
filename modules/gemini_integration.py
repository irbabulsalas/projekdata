"""
Gemini Integration Module for AI Data Analysis Platform
Advanced AI chat assistant with function calling capabilities powered by Google Gemini 2.5 Flash.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Google AI Libraries
import google.generativeai as genai
from google.ai.generativelanguage import GenerateContentResponse
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Data Analysis Libraries
try:
    from modules.smart_pipeline import SmartDataPipeline
except ImportError:
    SmartDataPipeline = None

try:
    from modules.ml_models import MLModelManager, AutoML
except ImportError:
    MLModelManager = None
    AutoML = None

try:
    from modules.visualizations import AdvancedVisualizer
except ImportError:
    AdvancedVisualizer = None

try:
    from modules.text_analytics import AdvancedTextAnalyzer
except ImportError:
    AdvancedTextAnalyzer = None

try:
    from utils.helpers import DataValidator, PerformanceMonitor
except ImportError:
    DataValidator = None
    PerformanceMonitor = None

try:
    from utils.error_handler import handle_errors, GeminiAPIError, ValidationError
except ImportError:
    def handle_errors(component_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return {"error": f"Error in {component_name}: {str(e)}"}
            return wrapper
        return decorator
    
    class GeminiAPIError(Exception):
        pass
    
    class ValidationError(Exception):
        pass

try:
    from utils.rate_limiter import rate_limit, gemini_rate_limiter, gemini_rate_limit
except ImportError:
    def rate_limit():
        def decorator(func):
            return func
        return decorator
    
    def gemini_rate_limit():
        def decorator(func):
            return func
        return decorator
    
    gemini_rate_limiter = None


class GeminiChatAssistant:
    """Advanced AI chat assistant with function calling for data analysis."""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini chat assistant."""
        self.api_key = api_key or self._load_api_key()
        self.model = None
        self.chat_history = []
        self.context = {}
        self.available_functions = {}
        
        # Initialize components with fallbacks
        self.pipeline = SmartDataPipeline() if SmartDataPipeline else None
        self.ml_manager = MLModelManager() if MLModelManager else None
        self.visualizer = AdvancedVisualizer() if AdvancedVisualizer else None
        self.text_analyzer = AdvancedTextAnalyzer() if AdvancedTextAnalyzer else None
        
        self._initialize_gemini()
        self._register_functions()
    
    def _load_api_key(self) -> str:
        """Load API key from environment or config."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        print("DEBUG: Gemini - .env file loaded successfully")
        
        api_key = os.getenv('GEMINI_API_KEY')
        print(f"DEBUG: Gemini API Key found: {bool(api_key)}")
        print(f"DEBUG: Gemini API Key length: {len(api_key) if api_key else 0}")
        
        if not api_key:
            raise GeminiAPIError("GEMINI_API_KEY not found in environment variables")
        
        return api_key
    
    def _initialize_gemini(self):
        """Initialize Gemini model with safety settings."""
        try:
            genai.configure(api_key=self.api_key)
           
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
           
            # Try to initialize with Gemini 2.5 Flash first, fallback to 1.5 Flash
            try:
                self.model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Simple initialization like user example
                self.model_name = "gemini-2.0-flash-exp"
                print("✅ Gemini 2.5 Flash initialized successfully")
            except Exception as quota_error:
                # Fallback to Gemini 1.5 Flash if quota exceeded
                print("Falling back to Gemini 1.5 Flash due to quota limits")
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.model_name = "gemini-1.5-flash"
           
        except Exception as e:
            raise GeminiAPIError(f"Failed to initialize Gemini: {str(e)}")
    
    def _get_function_declarations(self) -> List[Dict]:
        """Get function declarations for Gemini function calling."""
        return [
            {
                "name": "load_data",
                "description": "Load data from various sources (CSV, Excel, JSON, Parquet)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_source": {
                            "type": "string",
                            "description": "Path to data file or DataFrame object"
                        },
                        "file_type": {
                            "type": "string",
                            "enum": ["csv", "excel", "json", "parquet"],
                            "description": "Type of data file"
                        }
                    },
                    "required": ["data_source"]
                }
            },
            {
                "name": "analyze_data_quality",
                "description": "Analyze data quality and generate comprehensive report",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "perform_eda",
                "description": "Perform exploratory data analysis with visualizations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Target column for analysis (optional)"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "train_ml_models",
                "description": "Train machine learning models with auto-comparison",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Target column for prediction"
                        },
                        "model_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model types to train"
                        },
                        "hyperparameter_tuning": {
                            "type": "boolean",
                            "description": "Whether to perform hyperparameter tuning"
                        }
                    },
                    "required": ["target_column"]
                }
            },
            {
                "name": "create_visualizations",
                "description": "Create various types of data visualizations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "viz_type": {
                            "type": "string",
                            "enum": ["overview", "eda", "ml_results", "text_analysis"],
                            "description": "Type of visualization to create"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Target column for visualization"
                        }
                    },
                    "required": ["viz_type"]
                }
            },
            {
                "name": "analyze_text",
                "description": "Perform comprehensive text analytics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_column": {
                            "type": "string",
                            "description": "Column containing text data"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["sentiment", "keywords", "topics", "entities", "readability"],
                            "description": "Type of text analysis to perform"
                        }
                    },
                    "required": ["text_column"]
                }
            },
            {
                "name": "get_data_summary",
                "description": "Get comprehensive summary of the loaded data",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "export_results",
                "description": "Export analysis results to various formats",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["csv", "excel", "json", "pdf"],
                            "description": "Export format"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Name of the export file"
                        }
                    },
                    "required": ["format", "filename"]
                }
            }
        ]
    
    def _register_functions(self):
        """Register available functions for function calling."""
        self.available_functions = {
            "load_data": self._load_data_function,
            "analyze_data_quality": self._analyze_data_quality_function,
            "perform_eda": self._perform_eda_function,
            "train_ml_models": self._train_ml_models_function,
            "create_visualizations": self._create_visualizations_function,
            "analyze_text": self._analyze_text_function,
            "get_data_summary": self._get_data_summary_function,
            "export_results": self._export_results_function
        }
    
    @gemini_rate_limit
    @handle_errors("gemini")
    def chat(self, message: str, include_context: bool = True) -> Dict[str, Any]:
        """Send message to Gemini and handle function calls."""
        try:
            # Prepare chat history
            chat_session = self.model.start_chat(history=self.chat_history)
            
            # Add context if requested
            if include_context and self.context:
                context_message = self._format_context()
                full_message = f"Context: {context_message}\n\nUser: {message}"
            else:
                full_message = message
            
            # Send message to Gemini
            response = chat_session.send_message(full_message)
            
            # Handle function calls
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content
                
                if content.parts and content.parts[0].function_call:
                    function_call = content.parts[0].function_call
                    function_result = self._execute_function_call(function_call)
                    
                    # Send function result back to Gemini
                    follow_up_response = chat_session.send_message(
                        f"Function result: {json.dumps(function_result)}"
                    )
                    
                    result = {
                        "response": follow_up_response.text,
                        "function_call": {
                            "name": function_call.name,
                            "args": dict(function_call.args),
                            "result": function_result
                        }
                    }
                else:
                    result = {
                        "response": response.text,
                        "function_call": None
                    }
                
                # Update chat history
                self.chat_history.append({
                    "role": "user",
                    "parts": [{"text": full_message}]
                })
                self.chat_history.append({
                    "role": "model",
                    "parts": [content]
                })
                
                return result
            
            return {"response": "No response generated", "function_call": None}
            
        except Exception as e:
            raise GeminiAPIError(f"Chat failed: {str(e)}")
    
    def _execute_function_call(self, function_call) -> Dict[str, Any]:
        """Execute function call and return result."""
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        if function_name not in self.available_functions:
            return {"error": f"Unknown function: {function_name}"}
        
        try:
            function_result = self.available_functions[function_name](**function_args)
            return {"success": True, "result": function_result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_data_function(self, data_source: str, file_type: str = None) -> Dict[str, Any]:
        """Function to load data."""
        try:
            if self.pipeline is None:
                return {"error": "Data pipeline not available"}
            
            if file_type is None:
                # Auto-detect file type
                if data_source.endswith('.csv'):
                    file_type = 'csv'
                elif data_source.endswith(('.xlsx', '.xls')):
                    file_type = 'excel'
                elif data_source.endswith('.json'):
                    file_type = 'json'
                elif data_source.endswith('.parquet'):
                    file_type = 'parquet'
                else:
                    file_type = 'csv'  # Default
            
            data = self.pipeline.load_data(data_source)
            self.context['data_loaded'] = True
            self.context['data_shape'] = data.shape
            self.context['data_columns'] = data.columns.tolist()
            
            return {
                "message": f"Data loaded successfully. Shape: {data.shape}",
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "sample_data": data.head().to_dict()
            }
        except Exception as e:
            return {"error": f"Failed to load data: {str(e)}"}
    
    def _analyze_data_quality_function(self) -> Dict[str, Any]:
        """Function to analyze data quality."""
        try:
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            quality_report = self.pipeline.auto_detect_data_quality()
            self.context['quality_report'] = quality_report
            
            return {
                "message": "Data quality analysis completed",
                "overall_score": quality_report['overall_score'],
                "issues_found": len(quality_report['issues']),
                "recommendations": quality_report['recommendations']
            }
        except Exception as e:
            return {"error": f"Data quality analysis failed: {str(e)}"}
    
    def _perform_eda_function(self, target_column: str = None) -> Dict[str, Any]:
        """Function to perform EDA."""
        try:
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            eda_summary = self.pipeline.generate_eda_summary()
            self.context['eda_summary'] = eda_summary
            
            return {
                "message": "EDA completed successfully",
                "dataset_info": eda_summary['dataset_info'],
                "key_insights": self._extract_key_insights(eda_summary)
            }
        except Exception as e:
            return {"error": f"EDA failed: {str(e)}"}
    
    def _train_ml_models_function(self, target_column: str, model_types: List[str] = None, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Function to train ML models."""
        try:
            if self.ml_manager is None:
                return {"error": "ML manager not available"}
            
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            # Setup data for ML
            setup_info = self.ml_manager.setup_data(self.pipeline.data, target_column)
            
            # Train models
            training_results = self.ml_manager.train_models(model_types, hyperparameter_tuning)
            
            self.context['ml_results'] = training_results
            self.context['best_model'] = training_results.get('best_model')
            
            return {
                "message": f"ML models trained successfully. Best model: {training_results.get('best_model')}",
                "problem_type": setup_info['problem_type'],
                "best_model": training_results.get('best_model'),
                "performance_summary": self._format_ml_performance(training_results)
            }
        except Exception as e:
            return {"error": f"ML training failed: {str(e)}"}
    
    def _create_visualizations_function(self, viz_type: str, target_column: str = None) -> Dict[str, Any]:
        """Function to create visualizations."""
        try:
            if self.visualizer is None:
                return {"error": "Visualizer not available"}
            
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            self.visualizer.set_data(self.pipeline.data)
            
            if viz_type == "overview":
                vizs = self.visualizer.create_data_overview_dashboard()
            elif viz_type == "eda":
                vizs = self.visualizer.create_eda_visualizations(target_column)
            elif viz_type == "ml_results" and self.context.get('ml_results'):
                vizs = self.visualizer.create_ml_visualizations(self.context['ml_results'])
            elif viz_type == "text_analysis":
                # Would need text data
                vizs = {}
            else:
                return {"error": f"Unknown visualization type: {viz_type}"}
            
            self.context['visualizations'] = vizs
            
            return {
                "message": f"{viz_type.title()} visualizations created successfully",
                "visualization_count": len(vizs),
                "available_visualizations": list(vizs.keys())
            }
        except Exception as e:
            return {"error": f"Visualization creation failed: {str(e)}"}
    
    def _analyze_text_function(self, text_column: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
        """Function to analyze text."""
        try:
            if self.text_analyzer is None:
                return {"error": "Text analyzer not available"}
            
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            if text_column not in self.pipeline.data.columns:
                return {"error": f"Column '{text_column}' not found in data"}
            
            self.text_analyzer.set_data(self.pipeline.data[text_column])
            
            if analysis_type == "sentiment":
                result = self.text_analyzer.analyze_sentiment()
            elif analysis_type == "keywords":
                result = self.text_analyzer.extract_keywords()
            elif analysis_type == "topics":
                result = self.text_analyzer.perform_topic_modeling()
            elif analysis_type == "entities":
                result = self.text_analyzer.extract_entities()
            elif analysis_type == "readability":
                result = self.text_analyzer.calculate_readability()
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
            
            self.context['text_analysis'] = result
            
            return {
                "message": f"Text {analysis_type} analysis completed",
                "analysis_type": analysis_type,
                "summary": result
            }
        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def _get_data_summary_function(self) -> Dict[str, Any]:
        """Function to get data summary."""
        try:
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            data = self.pipeline.data
            
            summary = {
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "data_types": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "numeric_summary": data.describe().to_dict(),
                "sample_data": data.head().to_dict()
            }
            
            return {
                "message": "Data summary generated",
                "summary": summary
            }
        except Exception as e:
            return {"error": f"Data summary failed: {str(e)}"}
    
    def _export_results_function(self, format: str, filename: str) -> Dict[str, Any]:
        """Function to export results."""
        try:
            if not self.context.get('data_loaded'):
                return {"error": "No data loaded. Please load data first."}
            
            # Create exports directory
            import os
            os.makedirs('exports', exist_ok=True)
            
            if format in ['csv', 'excel', 'json']:
                # Export processed data
                if format == 'csv':
                    filepath = f"exports/{filename}.csv"
                    self.pipeline.data.to_csv(filepath, index=False)
                elif format == 'excel':
                    filepath = f"exports/{filename}.xlsx"
                    self.pipeline.data.to_excel(filepath, index=False)
                elif format == 'json':
                    filepath = f"exports/{filename}.json"
                    self.pipeline.data.to_json(filepath, orient='records')
                
                return {
                    "message": f"Data exported successfully to {filepath}",
                    "filepath": filepath
                }
            elif format == 'pdf':
                # Export analysis report
                report_content = self._generate_analysis_report()
                filepath = f"exports/{filename}.pdf"
                
                # Create PDF (simplified version)
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, report_content)
                pdf.output(filepath)
                
                return {
                    "message": f"Analysis report exported to {filepath}",
                    "filepath": filepath
                }
            else:
                return {"error": f"Unsupported export format: {format}"}
                
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
    
    def _format_context(self) -> str:
        """Format current context for chat."""
        context_parts = []
        
        if self.context.get('data_loaded'):
            context_parts.append(f"Data loaded with shape: {self.context.get('data_shape')}")
            context_parts.append(f"Columns: {', '.join(self.context.get('data_columns', []))}")
        
        if self.context.get('quality_report'):
            context_parts.append(f"Data quality score: {self.context['quality_report'].get('overall_score', 'N/A')}")
        
        if self.context.get('best_model'):
            context_parts.append(f"Best ML model: {self.context['best_model']}")
        
        return "; ".join(context_parts)
    
    def _extract_key_insights(self, eda_summary: Dict[str, Any]) -> List[str]:
        """Extract key insights from EDA summary."""
        insights = []
        
        dataset_info = eda_summary.get('dataset_info', {})
        if dataset_info:
            shape = dataset_info.get('shape', (0, 0))
            insights.append(f"Dataset contains {shape[0]} rows and {shape[1]} columns")
            
            missing_pct = (dataset_info.get('missing_values', {}).values() / shape[0]).mean() * 100
            if missing_pct > 0:
                insights.append(f"Average missing values: {missing_pct:.1f}%")
        
        correlation_analysis = eda_summary.get('correlation_analysis', {})
        strong_correlations = correlation_analysis.get('strong_correlations', [])
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations between features")
        
        return insights
    
    def _format_ml_performance(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format ML performance results."""
        comparison = training_results.get('comparison_summary', {})
        
        if comparison.empty:
            return {"message": "No performance data available"}
        
        # Get top 3 models
        top_models = comparison.head(3)
        
        performance_summary = {
            "top_models": top_models.to_dict('records'),
            "best_performance": top_models.iloc[0].to_dict() if not top_models.empty else {}
        }
        
        return performance_summary
    
    def _generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        report_parts = []
        report_parts.append("DATA ANALYSIS REPORT")
        report_parts.append("=" * 50)
        report_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("")
        
        # Data summary
        if self.context.get('data_loaded'):
            shape = self.context.get('data_shape', (0, 0))
            report_parts.append(f"Dataset Shape: {shape[0]} rows, {shape[1]} columns")
            report_parts.append("")
        
        # Quality analysis
        if self.context.get('quality_report'):
            quality = self.context['quality_report']
            report_parts.append("DATA QUALITY")
            report_parts.append("-" * 20)
            report_parts.append(f"Overall Score: {quality.get('overall_score', 'N/A')}")
            report_parts.append(f"Issues Found: {len(quality.get('issues', []))}")
            report_parts.append("")
        
        # ML Results
        if self.context.get('ml_results'):
            ml_results = self.context['ml_results']
            report_parts.append("MACHINE LEARNING RESULTS")
            report_parts.append("-" * 30)
            report_parts.append(f"Best Model: {ml_results.get('best_model', 'N/A')}")
            report_parts.append("")
        
        return "\n".join(report_parts)
    
    @rate_limit()
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history."""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear chat history and context."""
        self.chat_history = []
        self.context = {}
    
    def get_available_functions(self) -> List[str]:
        """Get list of available functions."""
        return list(self.available_functions.keys())
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get current context summary."""
        return self.context.copy()


class DataAnalysisAgent:
    """Intelligent data analysis agent with autonomous capabilities."""
    
    def __init__(self, api_key: str = None):
        """Initialize data analysis agent."""
        self.chat_assistant = GeminiChatAssistant(api_key)
        self.analysis_plan = []
        self.current_step = 0
        self.results = {}
    
    @gemini_rate_limit
    def autonomous_analysis(self, data_source: str, analysis_goal: str = "comprehensive") -> Dict[str, Any]:
        """Perform autonomous data analysis."""
        try:
            # Load data
            load_result = self.chat_assistant._load_data_function(data_source)
            if "error" in load_result:
                return {"error": load_result["error"]}
            
            # Generate analysis plan
            plan = self._generate_analysis_plan(analysis_goal)
            self.analysis_plan = plan
            
            # Execute analysis plan
            for i, step in enumerate(plan):
                step_result = self._execute_analysis_step(step)
                self.results[f"step_{i+1}"] = step_result
                
                # Update context
                if step_result.get("success"):
                    self.chat_assistant.context.update(step_result.get("context_updates", {}))
            
            # Generate final report
            final_report = self._generate_final_report()
            
            return {
                "message": "Autonomous analysis completed",
                "analysis_plan": plan,
                "results": self.results,
                "final_report": final_report
            }
            
        except Exception as e:
            return {"error": f"Autonomous analysis failed: {str(e)}"}
    
    def _generate_analysis_plan(self, analysis_goal: str) -> List[Dict[str, Any]]:
        """Generate analysis plan based on goal."""
        if analysis_goal == "comprehensive":
            return [
                {"step": "data_quality", "description": "Analyze data quality"},
                {"step": "eda", "description": "Perform exploratory data analysis"},
                {"step": "visualization", "description": "Create visualizations"},
                {"step": "ml_modeling", "description": "Train ML models if applicable"}
            ]
        elif analysis_goal == "quick_insights":
            return [
                {"step": "data_summary", "description": "Get data summary"},
                {"step": "basic_eda", "description": "Basic EDA"}
            ]
        elif analysis_goal == "ml_focused":
            return [
                {"step": "data_quality", "description": "Analyze data quality"},
                {"step": "feature_engineering", "description": "Prepare features"},
                {"step": "ml_modeling", "description": "Train and evaluate models"}
            ]
        else:
            return [{"step": "data_summary", "description": "Basic data analysis"}]
    
    def _execute_analysis_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single analysis step."""
        step_name = step["step"]
        
        try:
            if step_name == "data_quality":
                result = self.chat_assistant._analyze_data_quality_function()
            elif step_name == "eda":
                result = self.chat_assistant._perform_eda_function()
            elif step_name == "visualization":
                result = self.chat_assistant._create_visualizations_function("overview")
            elif step_name == "ml_modeling":
                # Auto-detect target column for ML
                data = self.chat_assistant.pipeline.data
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    target_col = numeric_cols[-1]  # Use last numeric column as target
                    result = self.chat_assistant._train_ml_models_function(target_col)
                else:
                    result = {"error": "No suitable target column found for ML"}
            elif step_name == "data_summary":
                result = self.chat_assistant._get_data_summary_function()
            else:
                result = {"error": f"Unknown step: {step_name}"}
            
            result["step"] = step_name
            result["description"] = step["description"]
            result["success"] = "error" not in result
            
            return result
            
        except Exception as e:
            return {
                "step": step_name,
                "description": step["description"],
                "success": False,
                "error": str(e)
            }
    
    def _generate_final_report(self) -> str:
        """Generate final analysis report."""
        report_parts = []
        report_parts.append("AUTONOMOUS DATA ANALYSIS REPORT")
        report_parts.append("=" * 60)
        report_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("")
        
        # Analysis plan summary
        report_parts.append("ANALYSIS PLAN EXECUTED:")
        for i, step in enumerate(self.analysis_plan, 1):
            status = "✓" if f"step_{i}" in self.results and self.results[f"step_{i}"].get("success") else "✗"
            report_parts.append(f"{i}. {step['description']} {status}")
        report_parts.append("")
        
        # Key findings
        report_parts.append("KEY FINDINGS:")
        for step_key, result in self.results.items():
            if result.get("success") and "message" in result:
                report_parts.append(f"- {result['message']}")
        report_parts.append("")
        
        return "\n".join(report_parts)
    
    def get_analysis_progress(self) -> Dict[str, Any]:
        """Get current analysis progress."""
        completed_steps = sum(1 for result in self.results.values() if result.get("success"))
        total_steps = len(self.analysis_plan)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "progress_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "current_step": self.current_step,
            "plan": self.analysis_plan
        }


# Utility functions for Gemini integration
def create_custom_function(name: str, description: str, parameters: Dict[str, Any], function: Callable) -> Dict[str, Any]:
    """Create custom function for Gemini integration."""
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "function": function
    }


def register_custom_function(chat_assistant: GeminiChatAssistant, custom_function: Dict[str, Any]):
    """Register custom function with chat assistant."""
    chat_assistant.available_functions[custom_function["name"]] = custom_function["function"]
    
    # Update function declarations (would need to reinitialize model)
    # This is a simplified version - in practice, you'd want to rebuild the function declarations
    pass


def analyze_with_gemini(data: pd.DataFrame, question: str, api_key: str = None) -> str:
    """Quick analysis function using Gemini."""
    try:
        assistant = GeminiChatAssistant(api_key)
        assistant.pipeline.data = data
        assistant.context['data_loaded'] = True
        assistant.context['data_shape'] = data.shape
        assistant.context['data_columns'] = data.columns.tolist()
        
        response = assistant.chat(question, include_context=True)
        return response.get("response", "No response generated")
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"


# Standalone functions for app_adapted.py compatibility
def generate_insights(df: pd.DataFrame, analysis_type: str = "general") -> str:
    """Generate quick insights about the data using Gemini."""
    try:
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            return "❌ Gemini API key not configured"
        
        # Initialize Gemini with fallback
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Gemini 2.5 Flash
        except Exception:
            # Fallback to Gemini 1.5 Flash
            print("Falling back to Gemini 1.5 Flash for insights generation")
            model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare data summary
        data_summary = f"""
        Dataset Analysis Request:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data types: {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        - Sample data: {df.head().to_dict()}
        """
        
        # Generate insights based on analysis type
        if analysis_type == "general":
            prompt = f"""
            Analyze this dataset and provide 3-5 key insights:
            
            {data_summary}
            
            Focus on:
            1. Data quality issues
            2. Interesting patterns or trends
            3. Potential analysis opportunities
            4. Recommendations for further analysis
            
            Keep it concise and actionable.
            """
        elif analysis_type == "ml":
            prompt = f"""
            Analyze this dataset for machine learning opportunities:
            
            {data_summary}
            
            Focus on:
            1. Potential target variables for prediction
            2. Feature engineering opportunities
            3. Suitable ML algorithms
            4. Data preprocessing recommendations
            
            Keep it concise and actionable.
            """
        else:
            prompt = f"""
            Analyze this dataset and provide insights:
            
            {data_summary}
            
            Provide 3-5 key findings and recommendations.
            """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Failed to generate insights: {str(e)}"


def chat_with_gemini(message: str, chat_history: List[Dict[str, str]], context: Dict[str, Any] = None) -> str:
    """Chat with Gemini about data analysis."""
    try:
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            return "❌ Gemini API key not configured"
        
        # Initialize Gemini - simple approach like user example
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Simple initialization
        except Exception:
            # Fallback to Gemini 1.5 Flash
            print("Falling back to Gemini 1.5 Flash for chat")
            model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare context
        context_str = ""
        if context:
            context_str = f"Current Data Context:\n"
            if 'shape' in context:
                context_str += f"- Dataset shape: {context['shape']}\n"
            if 'columns' in context:
                context_str += f"- Columns: {context['columns']}\n"
            if 'missing_values' in context:
                context_str += f"- Missing values: {context['missing_values']}\n"
            context_str += "\n"
        
        # Prepare chat history
        history_text = ""
        if chat_history:
            for msg in chat_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.title()}: {content}\n"
        
        # Combine everything
        full_prompt = f"""
        {context_str}
        Chat History:
        {history_text}
        
        Current Question: {message}
        
        Please provide a helpful response about data analysis. If you need to perform specific operations on the data, explain what you would do step by step.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Chat failed: {str(e)}"


def get_data_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Get data context for Gemini chat."""
    if df is None or df.empty:
        return {}
    
    try:
        context = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Add basic statistics for numeric columns
        if context['numeric_columns']:
            context['numeric_summary'] = df[context['numeric_columns']].describe().to_dict()
        
        # Add sample data
        context['sample_data'] = df.head().to_dict()
        
        return context
        
    except Exception as e:
        return {'error': f"Failed to get data context: {str(e)}"}