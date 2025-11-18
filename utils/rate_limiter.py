"""
Rate Limiter Module for AI Data Analysis Platform
Comprehensive rate limiting system for API calls and resource usage.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from collections import defaultdict, deque
from functools import wraps
import streamlit as st
from dataclasses import dataclass
import json
import os


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    cooldown_seconds: int = 60
    burst_limit: int = 5


class RateLimiter:
    """Advanced rate limiting system with multiple time windows."""
    
    def __init__(self, config: RateLimitConfig = None):
        """Initialize rate limiter with configuration."""
        self.config = config or RateLimitConfig()
        self.requests = defaultdict(lambda: defaultdict(deque))
        self.lock = threading.Lock()
        self.blocked_until = defaultdict(float)
        self.usage_stats = defaultdict(lambda: defaultdict(int))
        
        # Load persistent data if available
        self.load_persistent_data()
    
    def load_persistent_data(self):
        """Load rate limiting data from persistent storage."""
        try:
            if os.path.exists('data/rate_limits.json'):
                with open('data/rate_limits.json', 'r') as f:
                    data = json.load(f)
                    # Convert timestamps back to datetime objects
                    for user_id, timestamps in data.get('requests', {}).items():
                        for window, ts_list in timestamps.items():
                            self.requests[user_id][window] = deque(
                                [datetime.fromisoformat(ts) for ts in ts_list],
                                maxlen=1000
                            )
        except Exception:
            pass  # Start fresh if loading fails
    
    def save_persistent_data(self):
        """Save rate limiting data to persistent storage."""
        try:
            os.makedirs('data', exist_ok=True)
            data = {
                'requests': {
                    user_id: {
                        window: [ts.isoformat() for ts in timestamps]
                        for window, timestamps in windows.items()
                    }
                    for user_id, windows in self.requests.items()
                }
            }
            with open('data/rate_limits.json', 'w') as f:
                json.dump(data, f)
        except Exception:
            pass  # Continue without saving
    
    def _cleanup_old_requests(self, user_id: str):
        """Clean up old requests outside the time windows."""
        now = datetime.now()
        
        # Clean minute window
        minute_cutoff = now - timedelta(minutes=1)
        while (self.requests[user_id]['minute'] and 
               self.requests[user_id]['minute'][0] < minute_cutoff):
            self.requests[user_id]['minute'].popleft()
        
        # Clean hour window
        hour_cutoff = now - timedelta(hours=1)
        while (self.requests[user_id]['hour'] and 
               self.requests[user_id]['hour'][0] < hour_cutoff):
            self.requests[user_id]['hour'].popleft()
        
        # Clean day window
        day_cutoff = now - timedelta(days=1)
        while (self.requests[user_id]['day'] and 
               self.requests[user_id]['day'][0] < day_cutoff):
            self.requests[user_id]['day'].popleft()
    
    def _is_blocked(self, user_id: str) -> bool:
        """Check if user is currently blocked."""
        return datetime.now().timestamp() < self.blocked_until[user_id]
    
    def _block_user(self, user_id: str, duration_seconds: int = None):
        """Block user for specified duration."""
        duration = duration_seconds or self.config.cooldown_seconds
        self.blocked_until[user_id] = datetime.now().timestamp() + duration
    
    def can_make_request(self, user_id: str = "default") -> tuple:
        """Check if user can make a request."""
        with self.lock:
            # Check if user is blocked
            if self._is_blocked(user_id):
                remaining_time = self.blocked_until[user_id] - datetime.now().timestamp()
                return False, f"Rate limit exceeded. Please wait {int(remaining_time)} seconds."
            
            # Clean up old requests
            self._cleanup_old_requests(user_id)
            
            # Check all rate limits
            minute_count = len(self.requests[user_id]['minute'])
            hour_count = len(self.requests[user_id]['hour'])
            day_count = len(self.requests[user_id]['day'])
            
            if minute_count >= self.config.max_requests_per_minute:
                self._block_user(user_id)
                return False, "Minute rate limit exceeded. Please wait."
            
            if hour_count >= self.config.max_requests_per_hour:
                self._block_user(user_id, 300)  # 5 minute block
                return False, "Hourly rate limit exceeded. Please wait 5 minutes."
            
            if day_count >= self.config.max_requests_per_day:
                self._block_user(user_id, 3600)  # 1 hour block
                return False, "Daily rate limit exceeded. Please wait 1 hour."
            
            return True, "Request allowed."
    
    def record_request(self, user_id: str = "default"):
        """Record a request for rate limiting."""
        with self.lock:
            now = datetime.now()
            self.requests[user_id]['minute'].append(now)
            self.requests[user_id]['hour'].append(now)
            self.requests[user_id]['day'].append(now)
            
            # Update usage stats
            self.usage_stats[user_id]['total_requests'] += 1
            self.usage_stats[user_id]['last_request'] = now.isoformat()
            
            # Save persistent data periodically
            if self.usage_stats[user_id]['total_requests'] % 10 == 0:
                self.save_persistent_data()
    
    def get_usage_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get usage statistics for a user."""
        with self.lock:
            self._cleanup_old_requests(user_id)
            
            return {
                'requests_this_minute': len(self.requests[user_id]['minute']),
                'requests_this_hour': len(self.requests[user_id]['hour']),
                'requests_this_day': len(self.requests[user_id]['day']),
                'total_requests': self.usage_stats[user_id]['total_requests'],
                'last_request': self.usage_stats[user_id]['last_request'],
                'limits': {
                    'per_minute': self.config.max_requests_per_minute,
                    'per_hour': self.config.max_requests_per_hour,
                    'per_day': self.config.max_requests_per_day
                },
                'remaining_minute': max(0, self.config.max_requests_per_minute - len(self.requests[user_id]['minute'])),
                'remaining_hour': max(0, self.config.max_requests_per_hour - len(self.requests[user_id]['hour'])),
                'remaining_day': max(0, self.config.max_requests_per_day - len(self.requests[user_id]['day']))
            }
    
    def reset_user_limits(self, user_id: str = "default"):
        """Reset rate limits for a specific user."""
        with self.lock:
            self.requests[user_id].clear()
            self.blocked_until[user_id] = 0
            self.usage_stats[user_id].clear()
    
    def get_wait_time(self, user_id: str = "default") -> float:
        """Get time to wait before next request."""
        with self.lock:
            if self._is_blocked(user_id):
                return self.blocked_until[user_id] - datetime.now().timestamp()
            
            self._cleanup_old_requests(user_id)
            
            if len(self.requests[user_id]['minute']) >= self.config.max_requests_per_minute:
                # Time until oldest request expires
                oldest = self.requests[user_id]['minute'][0]
                return (oldest + timedelta(minutes=1) - datetime.now()).total_seconds()
            
            return 0


# Global rate limiter instances
gemini_rate_limiter = RateLimiter(RateLimitConfig(
    max_requests_per_minute=1,
    max_requests_per_hour=15,
    max_requests_per_day=150,
    cooldown_seconds=60
))

general_rate_limiter = RateLimiter(RateLimitConfig(
    max_requests_per_minute=30,
    max_requests_per_hour=500,
    max_requests_per_day=5000
))


def rate_limit(limiter: RateLimiter = None, user_id_func: Callable = None):
    """Decorator for rate limiting function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine which rate limiter to use
            target_limiter = limiter or general_rate_limiter
            
            # Get user ID
            if user_id_func:
                user_id = user_id_func(*args, **kwargs)
            elif hasattr(st, 'session_state') and hasattr(st.session_state, 'user_id'):
                user_id = st.session_state.user_id
            else:
                user_id = "default"
            
            # Check if request is allowed
            can_proceed, message = target_limiter.can_make_request(user_id)
            
            if not can_proceed:
                st.warning(f"⏰ {message}")
                return None
            
            # Record the request
            target_limiter.record_request(user_id)
            
            # Execute the function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Don't count failed requests towards rate limit
                with target_limiter.lock:
                    if target_limiter.requests[user_id]['minute']:
                        target_limiter.requests[user_id]['minute'].pop()
                    if target_limiter.requests[user_id]['hour']:
                        target_limiter.requests[user_id]['hour'].pop()
                    if target_limiter.requests[user_id]['day']:
                        target_limiter.requests[user_id]['day'].pop()
                raise e
        
        return wrapper
    return decorator


def gemini_rate_limit(func):
    """Specific decorator for Gemini API rate limiting."""
    return rate_limit(gemini_rate_limiter)(func)


def display_rate_limit_status(limiter: RateLimiter = None, user_id: str = "default"):
    """Display current rate limit status in Streamlit."""
    target_limiter = limiter or general_rate_limiter
    stats = target_limiter.get_usage_stats(user_id)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📊 Minute Usage",
            f"{stats['requests_this_minute']}/{stats['limits']['per_minute']}",
            f"{stats['remaining_minute']} left"
        )
    
    with col2:
        st.metric(
            "📈 Hour Usage", 
            f"{stats['requests_this_hour']}/{stats['limits']['per_hour']}",
            f"{stats['remaining_hour']} left"
        )
    
    with col3:
        st.metric(
            "📅 Day Usage",
            f"{stats['requests_this_day']}/{stats['limits']['per_day']}",
            f"{stats['remaining_day']} left"
        )
    
    # Show wait time if needed
    wait_time = target_limiter.get_wait_time(user_id)
    if wait_time > 0:
        st.info(f"⏱️ Next request available in: {int(wait_time)} seconds")


class RateLimitMiddleware:
    """Middleware for rate limiting in web applications."""
    
    def __init__(self, app, limiter: RateLimiter = None):
        self.app = app
        self.limiter = limiter or general_rate_limiter
    
    def __call__(self, environ, start_response):
        """WSGI application with rate limiting."""
        # Get client IP or user identifier
        user_id = environ.get('REMOTE_ADDR', 'unknown')
        
        # Check rate limit
        can_proceed, message = self.limiter.can_make_request(user_id)
        
        if not can_proceed:
            start_response('429 Too Many Requests', [
                ('Content-Type', 'text/plain'),
                ('Retry-After', str(int(self.limiter.get_wait_time(user_id))))
            ])
            return [message.encode()]
        
        # Record request
        self.limiter.record_request(user_id)
        
        # Continue with application
        return self.app(environ, start_response)


# Utility functions for rate limit management
def get_session_user_id() -> str:
    """Get user ID from session state."""
    if hasattr(st, 'session_state'):
        return st.session_state.get('user_id', 'default')
    return 'default'


def create_custom_rate_limiter(
    max_per_minute: int = 60,
    max_per_hour: int = 1000,
    max_per_day: int = 10000
) -> RateLimiter:
    """Create a custom rate limiter with specified limits."""
    config = RateLimitConfig(
        max_requests_per_minute=max_per_minute,
        max_requests_per_hour=max_per_hour,
        max_requests_per_day=max_per_day
    )
    return RateLimiter(config)


# Batch rate limiting for multiple operations
class BatchRateLimiter:
    """Rate limiter for batch operations."""
    
    def __init__(self, base_limiter: RateLimiter):
        self.base_limiter = base_limiter
        self.batch_queue = []
        self.batch_size = 5
        self.batch_timeout = 60  # seconds
    
    def add_to_batch(self, func: Callable, *args, **kwargs):
        """Add function to batch queue."""
        self.batch_queue.append((func, args, kwargs))
        
        if len(self.batch_queue) >= self.batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self):
        """Process all functions in the batch."""
        if not self.batch_queue:
            return []
        
        results = []
        user_id = get_session_user_id()
        
        # Check if we can process the entire batch
        can_proceed, message = self.base_limiter.can_make_request(user_id)
        
        if not can_proceed:
            st.warning(f"⏰ {message}")
            return []
        
        # Process all functions in the batch
        for func, args, kwargs in self.batch_queue:
            try:
                self.base_limiter.record_request(user_id)
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(None)
                st.error(f"Error in batch operation: {str(e)}")
        
        # Clear the queue
        self.batch_queue.clear()
        
        return results


# Streamlit-specific functions for app_adapted.py
def initialize_rate_limiter():
    """Initialize rate limiter for Streamlit app."""
    if 'rate_limiter_initialized' not in st.session_state:
        st.session_state.rate_limiter_initialized = True


def can_make_request(user_id: str = "default") -> tuple[bool, str]:
    """Check if user can make a request (simplified for app)."""
    return gemini_rate_limiter.can_make_request(user_id)


def show_rate_limit_status():
    """Display rate limit status in Streamlit."""
    display_rate_limit_status(gemini_rate_limiter)


def update_rate_limit():
    """Update rate limit after request."""
    gemini_rate_limiter.record_request()


def get_remaining_requests() -> int:
    """Get remaining requests for current hour."""
    stats = gemini_rate_limiter.get_usage_stats()
    return stats['remaining_hour']