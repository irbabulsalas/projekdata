#!/usr/bin/env python3
"""
Simple test script for Gemini 2.5 Flash API
Test different model names to find the working one
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

def test_gemini_models():
    """Test different Gemini model names to find working one"""
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    print(f"🔑 API Key found: {bool(api_key)}")
    print(f"🔑 API Key length: {len(api_key)}")
    
    # Configure API
    genai.configure(api_key=api_key)
    
    # Test different model names
    models_to_test = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash", 
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-pro",
        "gemini-pro-vision"
    ]
    
    for model_name in models_to_test:
        print(f"\n🧪 Testing model: {model_name}")
        try:
            # Try to create model
            model = genai.GenerativeModel(model_name)
            
            # Try to generate content
            response = model.generate_content("Hello, please respond with 'Model working' if you can see this message.")
            
            if response and response.text:
                print(f"✅ {model_name}: {response.text}")
                
                # If this model works, test chat functionality
                if "flash" in model_name.lower():
                    print(f"🎉 {model_name} is working! Testing chat...")
                    try:
                        chat = model.start_chat(history=[])
                        chat_response = chat.send_message("What is 2+2?")
                        print(f"💬 Chat response: {chat_response.text}")
                        return model_name  # Return working model
                    except Exception as chat_error:
                        print(f"❌ Chat failed with {model_name}: {str(chat_error)}")
                else:
                    print(f"✅ {model_name} basic generation works")
            else:
                print(f"❌ {model_name}: No response received")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {model_name}: {error_msg}")
            
            # Check for specific errors
            if "404" in error_msg and "not found" in error_msg:
                print(f"🔍 Model {model_name} not found")
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print(f"🚫 Quota limit exceeded for {model_name}")
            elif "permission" in error_msg.lower():
                print(f"🔒 Permission denied for {model_name}")
    
    return None

def test_simple_chat():
    """Test simple chat with working model"""
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    genai.configure(api_key=api_key)
    
    # Try with the most likely working model
    model_name = "gemini-2.0-flash-exp"
    
    try:
        print(f"\n🚀 Starting chat with {model_name}")
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=[])
        
        print("💬 Chat started. Type 'stop' to exit.")
        print("💬 You can now chat with Gemini 2.5 Flash!")
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ['stop', 'exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                response = chat.send_message(user_input)
                
                if response and response.text:
                    print(f"Gemini: {response.text}")
                else:
                    print("❌ No response received")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💬 You can continue typing...")
                
    except Exception as e:
        print(f"❌ Failed to start chat: {str(e)}")

if __name__ == "__main__":
    print("🧪 Gemini 2.5 Flash Test Script")
    print("=" * 50)
    
    # First, test which model works
    working_model = test_gemini_models()
    
    if working_model:
        print(f"\n🎉 Found working model: {working_model}")
        print("🚀 Starting simple chat...")
        test_simple_chat()
    else:
        print("\n❌ No working model found")
        print("💡 Suggestions:")
        print("1. Check your API key")
        print("2. Check your quota limits")
        print("3. Try a different model name")
        print("4. Check internet connection")