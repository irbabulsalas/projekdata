"""
Collaboration Features Module for AI Data Analysis Platform
Real-time collaboration, session sharing, and team workspace functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import json
import uuid
import hashlib
import base64
from datetime import datetime, timedelta
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Collaboration Libraries
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import websockets
    import asyncio
    WEBSOCKET_AVAILABLE = True
except ImportError:
    try:
        import websocket
        import asyncio
        WEBSOCKET_AVAILABLE = True
    except ImportError:
        WEBSOCKET_AVAILABLE = False

# Security
try:
    import jwt
    from cryptography.fernet import Fernet
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

import secrets

from utils.helpers import ConfigManager, get_timestamp
from utils.error_handler import handle_errors, ValidationError
from utils.rate_limiter import rate_limit


class SessionManager:
    """Manage collaborative analysis sessions."""
    
    def __init__(self, redis_url: str = None):
        """Initialize session manager."""
        self.redis_client = None
        self.sessions = {}
        self.user_sessions = {}
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except:
                print("Warning: Redis connection failed, using in-memory storage")
    
    @handle_errors("collaboration")
    def create_session(self, creator_id: str, session_name: str, 
                     description: str = "", is_public: bool = False) -> Dict[str, Any]:
        """Create a new collaborative session."""
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'creator_id': creator_id,
            'session_name': session_name,
            'description': description,
            'is_public': is_public,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'participants': [creator_id],
            'status': 'active',
            'data_state': {},
            'comments': [],
            'version_history': [],
            'settings': {
                'max_participants': 10,
                'auto_save': True,
                'save_interval': 300,  # 5 minutes
                'allow_anonymous': False
            }
        }
        
        # Store session
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                timedelta(hours=24),  # 24 hour expiry
                json.dumps(session_data)
            )
        else:
            self.sessions[session_id] = session_data
        
        # Add to user sessions
        self._add_user_session(creator_id, session_id)
        
        return {
            'session_id': session_id,
            'message': 'Session created successfully',
            'session_data': session_data
        }
    
    @handle_errors("collaboration")
    def join_session(self, session_id: str, user_id: str, user_name: str = None) -> Dict[str, Any]:
        """Join an existing session."""
        session_data = self._get_session(session_id)
        
        if not session_data:
            return {'error': 'Session not found'}
        
        if session_data['status'] != 'active':
            return {'error': 'Session is not active'}
        
        if len(session_data['participants']) >= session_data['settings']['max_participants']:
            return {'error': 'Session is full'}
        
        if user_id in session_data['participants']:
            return {'error': 'User already in session'}
        
        # Add user to session
        session_data['participants'].append(user_id)
        session_data['last_activity'] = datetime.now().isoformat()
        
        # Add join comment
        join_comment = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'user_name': user_name or f"User_{user_id[:8]}",
            'comment': f"joined the session",
            'timestamp': datetime.now().isoformat(),
            'type': 'system'
        }
        session_data['comments'].append(join_comment)
        
        # Update session
        self._update_session(session_id, session_data)
        self._add_user_session(user_id, session_id)
        
        return {
            'message': 'Joined session successfully',
            'session_data': session_data
        }
    
    @handle_errors("collaboration")
    def leave_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Leave a session."""
        session_data = self._get_session(session_id)
        
        if not session_data:
            return {'error': 'Session not found'}
        
        if user_id not in session_data['participants']:
            return {'error': 'User not in session'}
        
        # Remove user from session
        session_data['participants'].remove(user_id)
        session_data['last_activity'] = datetime.now().isoformat()
        
        # Add leave comment
        leave_comment = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'comment': f"left the session",
            'timestamp': datetime.now().isoformat(),
            'type': 'system'
        }
        session_data['comments'].append(leave_comment)
        
        # Update session
        self._update_session(session_id, session_data)
        self._remove_user_session(user_id, session_id)
        
        # If no participants left, deactivate session
        if len(session_data['participants']) == 0:
            session_data['status'] = 'inactive'
            self._update_session(session_id, session_data)
        
        return {'message': 'Left session successfully'}
    
    @handle_errors("collaboration")
    def update_session_data(self, session_id: str, user_id: str, 
                         data_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update session data with changes."""
        session_data = self._get_session(session_id)
        
        if not session_data:
            return {'error': 'Session not found'}
        
        if user_id not in session_data['participants']:
            return {'error': 'User not in session'}
        
        # Create version snapshot
        version_snapshot = {
            'version': len(session_data['version_history']) + 1,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'changes': data_updates,
            'previous_state': session_data['data_state'].copy()
        }
        
        # Apply updates
        session_data['data_state'].update(data_updates)
        session_data['version_history'].append(version_snapshot)
        session_data['last_activity'] = datetime.now().isoformat()
        
        # Update session
        self._update_session(session_id, session_data)
        
        return {
            'message': 'Session data updated',
            'version': version_snapshot['version']
        }
    
    @handle_errors("collaboration")
    def add_comment(self, session_id: str, user_id: str, user_name: str, 
                   comment: str, comment_type: str = 'user') -> Dict[str, Any]:
        """Add comment to session."""
        session_data = self._get_session(session_id)
        
        if not session_data:
            return {'error': 'Session not found'}
        
        if user_id not in session_data['participants']:
            return {'error': 'User not in session'}
        
        comment_data = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'user_name': user_name or f"User_{user_id[:8]}",
            'comment': comment,
            'timestamp': datetime.now().isoformat(),
            'type': comment_type
        }
        
        session_data['comments'].append(comment_data)
        session_data['last_activity'] = datetime.now().isoformat()
        
        # Update session
        self._update_session(session_id, session_data)
        
        return {'message': 'Comment added', 'comment_id': comment_data['id']}
    
    def _get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        if self.redis_client:
            session_json = self.redis_client.get(f"session:{session_id}")
            if session_json:
                return json.loads(session_json)
        else:
            return self.sessions.get(session_id)
        
        return None
    
    def _update_session(self, session_id: str, session_data: Dict[str, Any]):
        """Update session data."""
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                timedelta(hours=24),
                json.dumps(session_data)
            )
        else:
            self.sessions[session_id] = session_data
    
    def _add_user_session(self, user_id: str, session_id: str):
        """Add session to user's session list."""
        if self.redis_client:
            user_sessions_json = self.redis_client.get(f"user_sessions:{user_id}")
            user_sessions = json.loads(user_sessions_json) if user_sessions_json else []
            if session_id not in user_sessions:
                user_sessions.append(session_id)
            self.redis_client.setex(
                f"user_sessions:{user_id}",
                timedelta(hours=24),
                json.dumps(user_sessions)
            )
        else:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            if session_id not in self.user_sessions[user_id]:
                self.user_sessions[user_id].append(session_id)
    
    def _remove_user_session(self, user_id: str, session_id: str):
        """Remove session from user's session list."""
        if self.redis_client:
            user_sessions_json = self.redis_client.get(f"user_sessions:{user_id}")
            user_sessions = json.loads(user_sessions_json) if user_sessions_json else []
            if session_id in user_sessions:
                user_sessions.remove(session_id)
            self.redis_client.setex(
                f"user_sessions:{user_id}",
                timedelta(hours=24),
                json.dumps(user_sessions)
            )
        else:
            if user_id in self.user_sessions and session_id in self.user_sessions[user_id]:
                self.user_sessions[user_id].remove(session_id)
    
    @handle_errors("collaboration")
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        if self.redis_client:
            user_sessions_json = self.redis_client.get(f"user_sessions:{user_id}")
            user_session_ids = json.loads(user_sessions_json) if user_sessions_json else []
            
            sessions = []
            for session_id in user_session_ids:
                session_data = self._get_session(session_id)
                if session_data:
                    sessions.append(session_data)
            
            return sessions
        else:
            if user_id in self.user_sessions:
                sessions = []
                for session_id in self.user_sessions[user_id]:
                    session_data = self.sessions.get(session_id)
                    if session_data:
                        sessions.append(session_data)
                return sessions
        
        return []
    
    @handle_errors("collaboration")
    def get_public_sessions(self) -> List[Dict[str, Any]]:
        """Get all public sessions."""
        if self.redis_client:
            # This would require Redis SCAN in production
            # Simplified version for demo
            return []
        else:
            return [session for session in self.sessions.values() if session.get('is_public', False)]


class RealTimeCollaboration:
    """Real-time collaboration using WebSockets."""
    
    def __init__(self, session_manager: SessionManager):
        """Initialize real-time collaboration."""
        self.session_manager = session_manager
        self.connections = {}
        self.rooms = {}
        
        if not WEBSOCKET_AVAILABLE:
            print("Warning: WebSocket libraries not available")
    
    @handle_errors("collaboration")
    async def handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connection."""
        try:
            # Extract session and user info from path or initial message
            connection_info = await websocket.recv()
            auth_data = json.loads(connection_info)
            
            session_id = auth_data.get('session_id')
            user_id = auth_data.get('user_id')
            user_name = auth_data.get('user_name')
            
            if not session_id or not user_id:
                await websocket.send(json.dumps({'error': 'Missing authentication'}))
                return
            
            # Validate session and user
            session_data = self.session_manager._get_session(session_id)
            if not session_data or user_id not in session_data['participants']:
                await websocket.send(json.dumps({'error': 'Invalid session or user'}))
                return
            
            # Add connection
            connection_id = str(uuid.uuid4())
            self.connections[connection_id] = {
                'websocket': websocket,
                'session_id': session_id,
                'user_id': user_id,
                'user_name': user_name,
                'connected_at': datetime.now().isoformat()
            }
            
            # Add to room
            if session_id not in self.rooms:
                self.rooms[session_id] = []
            self.rooms[session_id].append(connection_id)
            
            # Send confirmation
            await websocket.send(json.dumps({
                'type': 'connection_confirmed',
                'connection_id': connection_id,
                'session_data': session_data
            }))
            
            # Notify other users
            await self._broadcast_to_room(session_id, connection_id, {
                'type': 'user_joined',
                'user_id': user_id,
                'user_name': user_name,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep connection alive
            await self._handle_connection_loop(websocket, connection_id, session_id)
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
    
    async def _handle_connection_loop(self, websocket, connection_id: str, session_id: str):
        """Handle messages from connected client."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(connection_id, session_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({'error': 'Invalid JSON'}))
        except websocket.exceptions.ConnectionClosed:
            await self._handle_disconnection(connection_id, session_id)
    
    async def _handle_message(self, connection_id: str, session_id: str, message: Dict[str, Any]):
        """Handle incoming message."""
        message_type = message.get('type')
        
        if message_type == 'data_update':
            # Handle data updates
            await self._handle_data_update(connection_id, session_id, message)
        elif message_type == 'comment':
            # Handle comments
            await self._handle_comment(connection_id, session_id, message)
        elif message_type == 'cursor_position':
            # Handle cursor position sharing
            await self._handle_cursor_position(connection_id, session_id, message)
        elif message_type == 'selection':
            # Handle text/data selection sharing
            await self._handle_selection(connection_id, session_id, message)
    
    async def _handle_data_update(self, connection_id: str, session_id: str, message: Dict[str, Any]):
        """Handle data update message."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Update session data
        result = self.session_manager.update_session_data(
            session_id, 
            connection['user_id'], 
            message.get('data', {})
        )
        
        if 'error' not in result:
            # Broadcast to other users in room
            await self._broadcast_to_room(session_id, connection_id, {
                'type': 'data_updated',
                'user_id': connection['user_id'],
                'user_name': connection['user_name'],
                'data': message.get('data'),
                'version': result.get('version'),
                'timestamp': datetime.now().isoformat()
            })
    
    async def _handle_comment(self, connection_id: str, session_id: str, message: Dict[str, Any]):
        """Handle comment message."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Add comment to session
        result = self.session_manager.add_comment(
            session_id,
            connection['user_id'],
            connection['user_name'],
            message.get('comment', ''),
            message.get('comment_type', 'user')
        )
        
        if 'error' not in result:
            # Broadcast comment to room
            await self._broadcast_to_room(session_id, connection_id, {
                'type': 'new_comment',
                'comment_id': result.get('comment_id'),
                'user_id': connection['user_id'],
                'user_name': connection['user_name'],
                'comment': message.get('comment'),
                'comment_type': message.get('comment_type', 'user'),
                'timestamp': datetime.now().isoformat()
            })
    
    async def _handle_cursor_position(self, connection_id: str, session_id: str, message: Dict[str, Any]):
        """Handle cursor position sharing."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Broadcast cursor position to other users
        await self._broadcast_to_room(session_id, connection_id, {
            'type': 'cursor_position',
            'user_id': connection['user_id'],
            'user_name': connection['user_name'],
            'position': message.get('position'),
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_selection(self, connection_id: str, session_id: str, message: Dict[str, Any]):
        """Handle selection sharing."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Broadcast selection to other users
        await self._broadcast_to_room(session_id, connection_id, {
            'type': 'selection',
            'user_id': connection['user_id'],
            'user_name': connection['user_name'],
            'selection': message.get('selection'),
            'timestamp': datetime.now().isoformat()
        })
    
    async def _broadcast_to_room(self, session_id: str, exclude_connection_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in room except sender."""
        if session_id not in self.rooms:
            return
        
        for connection_id in self.rooms[session_id]:
            if connection_id != exclude_connection_id:
                connection = self.connections.get(connection_id)
                if connection:
                    try:
                        await connection['websocket'].send(json.dumps(message))
                    except:
                        # Connection might be closed
                        pass
    
    async def _handle_disconnection(self, connection_id: str, session_id: str):
        """Handle WebSocket disconnection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Remove from connections
        del self.connections[connection_id]
        
        # Remove from room
        if session_id in self.rooms:
            self.rooms[session_id].remove(connection_id)
            if not self.rooms[session_id]:
                del self.rooms[session_id]
        
        # Leave session
        self.session_manager.leave_session(session_id, connection['user_id'])
        
        # Notify other users
        await self._broadcast_to_room(session_id, connection_id, {
            'type': 'user_left',
            'user_id': connection['user_id'],
            'user_name': connection['user_name'],
            'timestamp': datetime.now().isoformat()
        })
    
    def get_room_status(self, session_id: str) -> Dict[str, Any]:
        """Get current room status."""
        if session_id not in self.rooms:
            return {'error': 'Room not found'}
        
        connected_users = []
        for connection_id in self.rooms[session_id]:
            connection = self.connections.get(connection_id)
            if connection:
                connected_users.append({
                    'connection_id': connection_id,
                    'user_id': connection['user_id'],
                    'user_name': connection['user_name'],
                    'connected_at': connection['connected_at']
                })
        
        return {
            'session_id': session_id,
            'connected_users': connected_users,
            'user_count': len(connected_users)
        }


class VersionControl:
    """Version control system for collaborative analysis."""
    
    def __init__(self):
        """Initialize version control."""
        self.versions = {}
        self.branches = {}
        self.tags = {}
    
    @handle_errors("collaboration")
    def create_version(self, session_id: str, user_id: str, changes: Dict[str, Any], 
                     description: str = "") -> str:
        """Create a new version."""
        version_id = str(uuid.uuid4())
        
        version_data = {
            'version_id': version_id,
            'session_id': session_id,
            'user_id': user_id,
            'description': description,
            'changes': changes,
            'timestamp': datetime.now().isoformat(),
            'parent_version': self._get_latest_version(session_id)
        }
        
        if session_id not in self.versions:
            self.versions[session_id] = []
        
        self.versions[session_id].append(version_data)
        
        return version_id
    
    def _get_latest_version(self, session_id: str) -> Optional[str]:
        """Get latest version ID for session."""
        if session_id in self.versions and self.versions[session_id]:
            return self.versions[session_id][-1]['version_id']
        return None
    
    @handle_errors("collaboration")
    def get_version_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get version history for session."""
        return self.versions.get(session_id, [])
    
    @handle_errors("collaboration")
    def create_branch(self, session_id: str, branch_name: str, from_version: str = None) -> str:
        """Create a new branch."""
        branch_id = str(uuid.uuid4())
        
        branch_data = {
            'branch_id': branch_id,
            'session_id': session_id,
            'branch_name': branch_name,
            'from_version': from_version or self._get_latest_version(session_id),
            'created_at': datetime.now().isoformat(),
            'versions': []
        }
        
        if session_id not in self.branches:
            self.branches[session_id] = {}
        
        self.branches[session_id][branch_name] = branch_data
        
        return branch_id
    
    @handle_errors("collaboration")
    def merge_branch(self, session_id: str, source_branch: str, target_branch: str = 'main') -> Dict[str, Any]:
        """Merge branch into target branch."""
        if session_id not in self.branches:
            return {'error': 'No branches found for session'}
        
        branches = self.branches[session_id]
        if source_branch not in branches or target_branch not in branches:
            return {'error': 'Branch not found'}
        
        # Simplified merge logic
        source_versions = branches[source_branch]['versions']
        target_versions = branches[target_branch]['versions']
        
        # Merge versions (simplified)
        merged_versions = target_versions + source_versions
        
        # Update target branch
        branches[target_branch]['versions'] = merged_versions
        branches[target_branch]['last_merged'] = {
            'from_branch': source_branch,
            'merged_at': datetime.now().isoformat()
        }
        
        return {'message': f'Merged {source_branch} into {target_branch}'}


class CollaborationSecurity:
    """Security features for collaboration."""
    
    def __init__(self, secret_key: str = None):
        """Initialize security manager."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        if JWT_AVAILABLE:
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
        self.active_tokens = {}
    
    @handle_errors("collaboration")
    def generate_token(self, user_id: str, session_id: str, expires_in: int = 3600) -> str:
        """Generate JWT token for user."""
        if not JWT_AVAILABLE:
            # Fallback to simple token
            token = secrets.token_urlsafe(32)
            self.active_tokens[token] = {
                'user_id': user_id,
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat()
            }
            return token
        
        payload = {
            'user_id': user_id,
            'session_id': session_id,
            'exp': datetime.now().timestamp() + expires_in,
            'iat': datetime.now().timestamp()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store token
        self.active_tokens[token] = {
            'user_id': user_id,
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat()
        }
        
        return token
    
    @handle_errors("collaboration")
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        if not JWT_AVAILABLE:
            # Fallback verification
            if token in self.active_tokens:
                token_data = self.active_tokens[token]
                expires_at = datetime.fromisoformat(token_data['expires_at'])
                if datetime.now() < expires_at:
                    return {
                        'valid': True,
                        'user_id': token_data['user_id'],
                        'session_id': token_data['session_id']
                    }
                else:
                    return {'valid': False, 'error': 'Token expired'}
            else:
                return {'valid': False, 'error': 'Token not found'}
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is still active
            if token in self.active_tokens:
                return {
                    'valid': True,
                    'user_id': payload['user_id'],
                    'session_id': payload['session_id']
                }
            else:
                return {'valid': False, 'error': 'Token not active'}
                
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
    
    @handle_errors("collaboration")
    def encrypt_data(self, data: Any) -> str:
        """Encrypt sensitive data."""
        if not JWT_AVAILABLE:
            # Fallback - just return base64 encoded data
            data_json = json.dumps(data, default=str)
            return base64.b64encode(data_json.encode()).decode()
        
        data_json = json.dumps(data, default=str)
        encrypted_data = self.cipher.encrypt(data_json.encode())
        return encrypted_data.decode()
    
    @handle_errors("collaboration")
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt sensitive data."""
        if not JWT_AVAILABLE:
            # Fallback - just decode base64 data
            try:
                decoded_data = base64.b64decode(encrypted_data.encode()).decode()
                return json.loads(decoded_data)
            except:
                return None
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data.encode())
            return json.loads(decrypted_data.decode())
        except:
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens."""
        current_time = datetime.now()
        expired_tokens = []
        
        for token, token_data in self.active_tokens.items():
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if current_time > expires_at:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_tokens[token]


class CollaborationAnalytics:
    """Analytics for collaboration features."""
    
    def __init__(self):
        """Initialize analytics."""
        self.events = []
        self.metrics = {}
    
    @handle_errors("collaboration")
    def track_event(self, event_type: str, session_id: str, user_id: str, 
                   data: Dict[str, Any] = None):
        """Track collaboration event."""
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'session_id': session_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        self.events.append(event)
        
        # Update metrics
        self._update_metrics(event_type, session_id, user_id)
    
    def _update_metrics(self, event_type: str, session_id: str, user_id: str):
        """Update collaboration metrics."""
        if event_type not in self.metrics:
            self.metrics[event_type] = {
                'total_count': 0,
                'unique_sessions': set(),
                'unique_users': set(),
                'hourly_distribution': {}
            }
        
        self.metrics[event_type]['total_count'] += 1
        self.metrics[event_type]['unique_sessions'].add(session_id)
        self.metrics[event_type]['unique_users'].add(user_id)
        
        # Hourly distribution
        hour = datetime.now().hour
        if hour not in self.metrics[event_type]['hourly_distribution']:
            self.metrics[event_type]['hourly_distribution'][hour] = 0
        self.metrics[event_type]['hourly_distribution'][hour] += 1
    
    @handle_errors("collaboration")
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for specific session."""
        session_events = [event for event in self.events if event['session_id'] == session_id]
        
        if not session_events:
            return {'message': 'No events found for session'}
        
        # Calculate metrics
        total_events = len(session_events)
        unique_users = len(set(event['user_id'] for event in session_events))
        event_types = {}
        
        for event in session_events:
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Time analysis
        if len(session_events) > 1:
            start_time = datetime.fromisoformat(session_events[0]['timestamp'])
            end_time = datetime.fromisoformat(session_events[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        return {
            'session_id': session_id,
            'total_events': total_events,
            'unique_users': unique_users,
            'event_types': event_types,
            'duration_seconds': duration,
            'events_per_user': total_events / unique_users if unique_users > 0 else 0
        }
    
    @handle_errors("collaboration")
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for specific user."""
        user_events = [event for event in self.events if event['user_id'] == user_id]
        
        if not user_events:
            return {'message': 'No events found for user'}
        
        # Calculate metrics
        total_events = len(user_events)
        unique_sessions = len(set(event['session_id'] for event in user_events))
        event_types = {}
        
        for event in user_events:
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        return {
            'user_id': user_id,
            'total_events': total_events,
            'unique_sessions': unique_sessions,
            'event_types': event_types,
            'avg_events_per_session': total_events / unique_sessions if unique_sessions > 0 else 0
        }
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get global collaboration analytics."""
        if not self.events:
            return {'message': 'No events recorded'}
        
        total_events = len(self.events)
        unique_users = len(set(event['user_id'] for event in self.events))
        unique_sessions = len(set(event['session_id'] for event in self.events))
        
        return {
            'total_events': total_events,
            'unique_users': unique_users,
            'unique_sessions': unique_sessions,
            'event_type_metrics': {
                event_type: {
                    'count': metrics['total_count'],
                    'unique_sessions': len(metrics['unique_sessions']),
                    'unique_users': len(metrics['unique_users'])
                }
                for event_type, metrics in self.metrics.items()
            }
        }


# Utility functions for collaboration
@rate_limit()
def create_shareable_link(session_id: str, base_url: str = "http://localhost:5000") -> str:
    """Create shareable link for session."""
    return f"{base_url}/session/{session_id}"


def generate_session_qr_code(session_id: str, base_url: str = "http://localhost:5000") -> str:
    """Generate QR code for session."""
    try:
        import qrcode
        from io import BytesIO
        import base64
        
        link = create_shareable_link(session_id, base_url)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(link)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except ImportError:
        return link  # Fallback to just return link


def validate_session_permissions(user_id: str, session_id: str, 
                            session_manager: SessionManager) -> Dict[str, Any]:
    """Validate user permissions for session."""
    session_data = session_manager._get_session(session_id)
    
    if not session_data:
        return {'valid': False, 'error': 'Session not found'}
    
    if user_id not in session_data['participants']:
        return {'valid': False, 'error': 'User not in session'}
    
    if session_data['status'] != 'active':
        return {'valid': False, 'error': 'Session not active'}
    
    return {'valid': True, 'permissions': 'read_write'}