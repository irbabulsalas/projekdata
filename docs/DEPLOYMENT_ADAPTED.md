# 🚀 Deployment Guide - Adapted Version

## Overview

Guide untuk mendeploy AI Data Analysis Platform versi adaptasi dengan authentication dan database features.

## 🌐 Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env dengan GEMINI_API_KEY

# Initialize database
python -c "from database.init_db import initialize_database; initialize_database()"

# Run application
streamlit run app_adapted.py --server.port 8503
```

### 2. Railway (Recommended)

#### Prerequisites
- GitHub account
- Railway account
- Gemini API Key

#### Steps

1. **Push ke GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/username/projekdata.git
   git push -u origin main
   ```

2. **Setup Railway**
   - Login ke Railway dashboard
   - Click "New Project"
   - Pilih "Deploy from GitHub repo"
   - Connect repository Anda
   - Railway akan otomatis detect Streamlit app

3. **Configure Environment Variables**
   ```env
   GEMINI_API_KEY=your-gemini-api-key
   SECRET_KEY=your-secret-key-for-sessions
   DATABASE_URL=sqlite:///database/app.db
   ```

4. **Update Railway Service**
   - Tambahkan file `railway.toml`:
   ```toml
   [build]
   builder = "NIXPACKS"
   
   [deploy]
   startCommand = "streamlit run app_adapted.py --server.port $PORT --server.address 0.0.0.0"
   healthcheckPath = "/_stcore/health"
   healthcheckTimeout = 100
   restartPolicyType = "ON_FAILURE"
   restartPolicyMaxRetries = 10
   ```

5. **Deploy**
   - Railway akan otomatis build dan deploy
   - Aplikasi akan available di `https://your-app-name.up.railway.app`

#### Railway Configuration

**railway.toml**
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run app_adapted.py --server.port $PORT --server.address 0.0.0.0"
healthcheckPath = "/_stcore/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[[services]]
name = "app"
source = "."
[services.config]
cpu = "1"
memory = "512Mi"
```

### 3. Heroku

#### Setup
```bash
# Install Heroku CLI
# Login ke Heroku
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set GEMINI_API_KEY=your-api-key
heroku config:set SECRET_KEY=your-secret-key

# Deploy
git push heroku main
```

**Procfile**
```
web: streamlit run app_adapted.py --server.port $PORT --server.address 0.0.0.0
```

**runtime.txt**
```
python-3.11.5
```

### 4. Docker

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create database directory
RUN mkdir -p database

# Expose port
EXPOSE 8501

# Initialize database
RUN python -c "from database.init_db import initialize_database; initialize_database()"

# Run the app
CMD ["streamlit", "run", "app_adapted.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./database:/app/database
    restart: unless-stopped
```

#### Run with Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 5. AWS EC2

#### Setup Script
```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/username/projekdata.git
cd projekdata

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your keys

# Initialize database
python -c "from database.init_db import initialize_database; initialize_database()"

# Install and configure nginx as reverse proxy
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure nginx
sudo tee /etc/nginx/sites-available/streamlit << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8503;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Create systemd service
sudo tee /etc/systemd/system/streamlit.service << EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/projekdata
Environment=PATH=/home/ubuntu/projekdata/venv/bin
ExecStart=/home/ubuntu/projekdata/venv/bin/streamlit run app_adapted.py --server.port 8503
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl start streamlit
sudo systemctl enable streamlit
```

## 🔧 Configuration

### Environment Variables

**Required**
```env
GEMINI_API_KEY=your-gemini-api-key
SECRET_KEY=your-secret-key-for-sessions
```

**Optional**
```env
DATABASE_URL=sqlite:///database/app.db
SESSION_TIMEOUT=3600
MAX_FILE_SIZE=200MB
RATE_LIMIT_REQUESTS=15
RATE_LIMIT_WINDOW=3600
```

### Database Configuration

#### SQLite (Default)
```env
DATABASE_URL=sqlite:///database/app.db
```

#### PostgreSQL
```env
DATABASE_URL=postgresql://username:password@host:port/database
```

#### MySQL
```env
DATABASE_URL=mysql://username:password@host:port/database
```

## 🔒 Security Considerations

### Production Security

1. **Environment Variables**
   - Never commit API keys to Git
   - Use platform-specific secret management
   - Rotate keys regularly

2. **Database Security**
   - Use strong passwords
   - Enable SSL connections
   - Regular backups
   - Access control

3. **Application Security**
   - HTTPS only
   - Rate limiting
   - Input validation
   - Error handling

4. **Infrastructure Security**
   - Firewall rules
   - VPN access
   - Monitoring
   - Updates

### SSL/HTTPS Setup

#### Railway
- Automatic SSL provided
- Custom domain support

#### Nginx (Self-hosted)
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8503;
        # ... other proxy settings
    }
}
```

## 📊 Monitoring & Logging

### Application Monitoring

#### Streamlit Monitoring
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

#### Health Checks
```python
# Add to app_adapted.py
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### Database Monitoring

#### Query Logging
```python
import sqlite3

# Enable SQLite logging
conn = sqlite3.connect('database/app.db')
conn.set_trace_callback(print)
```

#### Performance Metrics
- Connection count
- Query execution time
- Error rates
- Storage usage

## 🚀 Performance Optimization

### Application Performance

1. **Caching**
   ```python
   import streamlit as st
   from functools import lru_cache

   @st.cache_data
   def expensive_operation(data):
       # Cache expensive computations
       return result
   ```

2. **Lazy Loading**
   ```python
   # Load modules only when needed
   if page == "ML Models":
       from modules.ml_models import train_models
   ```

3. **Database Optimization**
   ```python
   # Add indexes
   CREATE INDEX idx_projects_user_id ON projects(user_id);
   ```

### Infrastructure Performance

1. **Load Balancing**
   - Multiple app instances
   - Database read replicas
   - CDN for static assets

2. **Resource Scaling**
   - Horizontal scaling
   - Auto-scaling rules
   - Performance monitoring

## 🐛 Troubleshooting

### Common Deployment Issues

**Q: "Application won't start"**
A: Check logs, verify dependencies, confirm environment variables

**Q: "Database connection failed"**
A: Verify database URL, check credentials, confirm network access

**Q: "API rate limits"**
A: Monitor usage, implement caching, upgrade API tier

**Q: "Memory errors"**
A: Optimize code, increase resources, implement streaming

### Debug Mode

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Railway Logs
```bash
railway logs
```

#### Docker Logs
```bash
docker-compose logs -f app
```

## 💰 Cost Optimization

### Railway Costs
- Hobby plan: $5/month
- Pro plan: $20/month
- Additional features: Extra

### AWS EC2 Costs
- t3.micro: Free tier eligible
- t3.small: ~$15/month
- t3.medium: ~$30/month

### Optimization Tips
1. Use free tiers when possible
2. Monitor resource usage
3. Optimize database queries
4. Implement caching
5. Use CDN for static assets

## 📈 Scaling Strategy

### Vertical Scaling
- Increase CPU/RAM
- Better storage
- Faster network

### Horizontal Scaling
- Load balancer
- Multiple app instances
- Database clustering

### Database Scaling
- Read replicas
- Sharding
- Caching layer

## 🔮 Future Enhancements

### Planned Features
- Multi-region deployment
- Advanced monitoring
- Auto-scaling
- CI/CD pipeline
- Blue-green deployment

### Infrastructure Improvements
- Container orchestration
- Microservices architecture
- Event-driven architecture
- Serverless options

---

## 📞 Support

For deployment issues:
1. Check platform documentation
2. Review application logs
3. Verify configuration
4. Contact platform support

---

*Last updated: November 2025*