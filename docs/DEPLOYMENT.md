# 🚀 Deployment Guide - ProjekData

Guide lengkap untuk deployment AI Data Analysis Platform di berbagai platform.

## 📋 Prerequisites

### Requirements
- Python 3.9+
- Git
- Account untuk platform deployment (Heroku/Railway/Docker)
- Gemini API Key

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379
DEBUG=False
```

## 🐳 Docker Deployment

### Local Development
1. Clone repository:
```bash
git clone <repository-url>
cd projekdata
```

2. Setup environment:
```bash
cp .env.example .env
# Edit .env dengan API key Anda
```

3. Run dengan Docker Compose:
```bash
docker-compose up -d
```

4. Akses aplikasi:
```
http://localhost:8501
```

### Production Docker
1. Build image:
```bash
docker build -t projekdata .
```

2. Run container:
```bash
docker run -d \
  --name projekdata \
  -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/exports:/app/exports \
  projekdata
```

## ☁️ Cloud Deployment

### Heroku Deployment

#### 1. Setup Heroku CLI
```bash
# Install Heroku CLI
# Login ke Heroku
heroku login

# Create app
heroku create your-app-name
```

#### 2. Deploy
```bash
# Add buildpack
heroku buildpacks:set heroku/python

# Set environment variables
heroku config:set GEMINI_API_KEY=your_gemini_api_key
heroku config:set REDIS_URL=redis://your-redis-url

# Deploy
git push heroku main
```

#### 3. Scale Dynos
```bash
heroku ps:scale web=1
```

### Railway Deployment

#### 1. Setup Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login
```

#### 2. Deploy
```bash
# Initialize project
railway init

# Link repository
railway link

# Set environment variables
railway variables set GEMINI_API_KEY=your_gemini_api_key

# Deploy
railway up
```

#### 3. Configuration
- `railway.json` sudah dikonfigurasi
- Build otomatis dengan Nixpacks
- Health check terkonfigurasi

### AWS Deployment

#### 1. ECS Deployment
```bash
# Create ECR repository
aws ecr create-repository --repository-name projekdata

# Build dan push image
docker build -t projekdata .
docker tag projekdata:latest <account-id>.dkr.ecr.<region>.amazonaws.com/projekdata:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/projekdata:latest
```

#### 2. ECS Task Definition
```json
{
  "family": "projekdata",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "projekdata",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/projekdata:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GEMINI_API_KEY",
          "value": "your_gemini_api_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/projekdata",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform Deployment

#### 1. Cloud Run
```bash
# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Build dan deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/projekdata
gcloud run deploy --image gcr.io/PROJECT-ID/projekdata --platform managed
```

#### 2. Environment Variables
```bash
gcloud run services update projekdata \
  --set-env-vars GEMINI_API_KEY=your_gemini_api_key
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `REDIS_URL` | Redis connection URL | Optional |
| `DEBUG` | Debug mode | No |
| `STREAMLIT_SERVER_PORT` | Server port | No |
| `STREAMLIT_SERVER_ADDRESS` | Server address | No |

### Resource Requirements

#### Minimum
- CPU: 1 core
- RAM: 2GB
- Storage: 10GB

#### Recommended
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB

#### Production
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB

## 🔒 Security

### SSL/TLS
- Gunakan HTTPS di production
- Configure SSL certificates
- Enable HSTS headers

### API Keys
- Jangan commit API keys ke repository
- Gunakan environment variables
- Rotate keys secara berkala

### Data Protection
- Encrypt sensitive data
- Implement access controls
- Regular security audits

## 📊 Monitoring

### Health Checks
```bash
# Health endpoint
curl https://your-app.com/_stcore/health
```

### Logging
```bash
# View logs
docker logs projekdata
heroku logs --tail
railway logs
```

### Metrics
- CPU usage
- Memory consumption
- Request latency
- Error rates

## 🔄 CI/CD

### GitHub Actions
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Railway
        uses: railway-app/railway-action@v1
        with:
          api-token: ${{ secrets.RAILWAY_TOKEN }}
```

### Auto-deployment
- GitHub integration
- Automatic builds
- Rollback capabilities

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Increase memory limit
docker run --memory=4g projekdata
```

#### 2. API Key Issues
```bash
# Verify API key
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models
```

#### 3. Port Issues
```bash
# Check port availability
netstat -tulpn | grep 8501
```

#### 4. Dependency Issues
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode
```bash
# Enable debug
export DEBUG=True
streamlit run app.py --logger.level=debug
```

## 📈 Performance Optimization

### Caching
- Redis untuk session storage
- File caching untuk datasets
- Model caching

### Load Balancing
- Multiple instances
- Load balancer configuration
- Auto-scaling

### Database Optimization
- Connection pooling
- Query optimization
- Indexing

## 🔄 Updates & Maintenance

### Version Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update models
python -m spacy download en_core_web_sm
```

### Backup Strategy
- Regular database backups
- Model artifact backups
- Configuration backups

### Rollback Plan
- Version control
- Database snapshots
- Quick rollback procedures

## 📞 Support

### Documentation
- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)

### Community
- GitHub Issues
- Discord Server
- Stack Overflow

### Contact
- Email: support@projekdata.com
- Documentation: docs.projekdata.com

---

**Happy Deploying! 🚀**

*Untuk bantuan tambahan, hubungi tim support ProjekData*