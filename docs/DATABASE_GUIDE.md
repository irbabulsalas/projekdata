# 📊 Database & Authentication Guide

## Overview

Aplikasi yang diadaptasi dilengkapi dengan sistem authentication dan database untuk menyimpan projects, datasets, dan models.

## 🏗️ Struktur Database

### Tables

#### Users
- `id` - Primary key
- `username` - Unique username
- `email` - Email address
- `password_hash` - Hashed password
- `created_at` - Registration timestamp

#### Projects
- `id` - Primary key
- `user_id` - Foreign key to users
- `name` - Project name
- `description` - Project description
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp

#### Datasets
- `id` - Primary key
- `project_id` - Foreign key to projects
- `name` - Dataset name
- `file_path` - File location
- `metadata` - JSON metadata
- `created_at` - Upload timestamp

#### Models
- `id` - Primary key
- `project_id` - Foreign key to projects
- `name` - Model name
- `model_type` - Algorithm used
- `file_path` - Model file location
- `metrics` - JSON performance metrics
- `created_at` - Training timestamp

## 🔐 Authentication System

### Registration
1. Klik "Register" di sidebar
2. Masukkan username, email, dan password
3. Password akan di-hash menggunakan bcrypt
4. Akun dibuat dan otomatis login

### Login
1. Klik "Login" di sidebar
2. Masukkan username dan password
3. Sistem verifikasi password hash
4. Session state diset untuk user yang login

### Security Features
- Password hashing dengan bcrypt
- Session management
- Rate limiting untuk login attempts
- Input validation dan sanitization

## 💾 File Storage

### Database Files
- `database/app.db` - SQLite database
- `database/users/` - User data storage
- `database/projects/` - Project data storage
- `database/datasets/` - Dataset files
- `database/models/` - Trained models

### File Organization
```
database/
├── app.db                    # SQLite database
├── users/                    # User-specific data
│   └── {user_id}/
│       ├── projects/         # Project folders
│       ├── datasets/         # Dataset files
│       └── models/           # Model files
└── temp/                     # Temporary files
```

## 🛠️ Database Operations

### Initialization
```python
from database.init_db import initialize_database
initialize_database()  # Create tables and folders
```

### User Management
```python
from database.auth_manager import AuthManager

# Register user
AuthManager.register_user(username, email, password)

# Login user
AuthManager.login_user(username, password)

# Get current user
current_user = AuthManager.get_current_user()
```

### Project Management
```python
from database.session_manager import SessionManager

# Save project
SessionManager.save_project(name, description, data)

# Load project
project_data = SessionManager.load_project(project_id)

# List projects
projects = SessionManager.list_user_projects()
```

## 📝 API Reference

### AuthManager Class

#### Methods
- `register_user(username, email, password)` - Register new user
- `login_user(username, password)` - Authenticate user
- `logout_user()` - Clear session
- `get_current_user()` - Get logged-in user info
- `is_logged_in()` - Check authentication status
- `render_auth_sidebar()` - Render login/register UI

### SessionManager Class

#### Methods
- `save_project(name, description, data)` - Save project to database
- `load_project(project_id)` - Load project data
- `delete_project(project_id)` - Delete project
- `list_user_projects()` - Get user's projects
- `save_dataset(name, file, metadata)` - Save dataset
- `load_dataset(dataset_id)` - Load dataset
- `save_model(name, model, metrics)` - Save trained model
- `load_model(model_id)` - Load model

### DBManager Class

#### Methods
- `execute_query(query, params)` - Execute SQL query
- `fetch_one(query, params)` - Fetch single result
- `fetch_all(query, params)` - Fetch all results
- `insert_data(table, data)` - Insert data into table
- `update_data(table, data, condition)` - Update data
- `delete_data(table, condition)` - Delete data

## 🔧 Configuration

### Environment Variables
```env
DATABASE_URL=sqlite:///database/app.db
SECRET_KEY=your-secret-key-here
SESSION_TIMEOUT=3600  # 1 hour
```

### Database Settings
- Default: SQLite for simplicity
- Can be configured for PostgreSQL/MySQL
- Connection pooling for production
- Automatic backups recommended

## 🚀 Deployment Considerations

### Production Database
1. Use PostgreSQL or MySQL for production
2. Configure connection pooling
3. Set up regular backups
4. Enable SSL connections
5. Configure read replicas for scaling

### Security
1. Use environment variables for credentials
2. Enable database encryption
3. Implement rate limiting
4. Regular security updates
5. Monitor for suspicious activity

### Performance
1. Add database indexes
2. Optimize queries
3. Implement caching
4. Monitor query performance
5. Use connection pooling

## 🐛 Troubleshooting

### Common Issues

**Q: "Database connection failed"**
A: Check database file permissions and path

**Q: "User registration failed"**
A: Verify username uniqueness and password requirements

**Q: "Project save failed"**
A: Check disk space and file permissions

**Q: "Session expired"**
A: Check session timeout configuration

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Monitoring

### Database Metrics
- Connection count
- Query execution time
- Error rates
- Storage usage

### User Metrics
- Registration rate
- Login frequency
- Project creation
- Feature usage

## 🔮 Future Enhancements

### Planned Features
- Multi-factor authentication
- OAuth integration (Google, GitHub)
- Team collaboration
- Advanced permissions
- Audit logging
- Data encryption at rest

### Scalability
- Database sharding
- Read replicas
- Caching layer
- CDN integration
- Load balancing

---

## 📞 Support

For database-related issues:
1. Check logs in `database/logs/`
2. Verify configuration
3. Test with sample data
4. Contact support with error details

---

*Last updated: November 2025*