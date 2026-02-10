# Production Deployment Guide

## Pre-Deployment Checklist

### 1. Verify All Components
```bash
# Check Python version
python --version  # Should be 3.7+

# Check all modules installed
pip list | grep -E "flask|torch|numpy|pandas|scikit-learn"

# Verify project structure
ls -la
# Should see: app.py, network_config.py, models/, templates/, static/
```

### 2. Train Models for Each Network Type

```bash
# Generate training data
python generate_sample_datasets.py --output data/ --samples 10000

# Train SDN model
python train_network_model.py \
  --network-type sdn \
  --data data/sdn_data.csv \
  --epochs 100 \
  --batch-size 64

# Train Traditional model
python train_network_model.py \
  --network-type traditional \
  --data data/traditional_data.csv \
  --epochs 100

# Train IoT model
python train_network_model.py \
  --network-type iot \
  --data data/iot_data.csv \
  --epochs 100

# Train Hybrid model
python train_network_model.py \
  --network-type hybrid \
  --data data/hybrid_data.csv \
  --epochs 100
```

### 3. Verify Model Files

```bash
ls -la models/
# Should contain:
# - sdn_scaler.pkl
# - sdn_lstm_svm_fed_model.pkl
# - sdn_tcn_bigru_fed_model.pkl
# - traditional_*.pkl
# - iot_*.pkl
# - hybrid_*.pkl
# (or fallback to original lstm_svm_fed_model.pkl, etc.)
```

### 4. Test Application Locally

```bash
# Start development server
python app.py

# In another terminal, test endpoints
curl http://localhost:5000/api/network-features/sdn
curl -X POST http://localhost:5000/predict \
  -F "network_type=sdn" \
  -F "dt=100" \
  -F "switch=1" \
  ... (other features)

# Test all network types in browser
# Visit http://localhost:5000/login
# Create account, login, test each network type
```

## Production Deployment

### Option 1: Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or with production settings
gunicorn \
  -w 8 \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  app:app
```

### Option 2: Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV FLASK_ENV=production
ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t ddos-detection .
docker run -p 5000:5000 -v $(pwd)/models:/app/models ddos-detection
```

### Option 3: Using Nginx + Gunicorn

`nginx.conf`:
```nginx
upstream ddos_app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your_domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your_domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://ddos_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/static;
        expires 30d;
    }
}
```

Run:
```bash
# Start Gunicorn on port 8000
gunicorn -w 4 -b 127.0.0.1:8000 app:app &

# Start Nginx
sudo systemctl start nginx
```

## Security Hardening

### 1. Environment Variables

Create `.env`:
```
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-very-long-random-secret-key-here
DB_PATH=/var/lib/ddos-detection/users.db
MODELS_PATH=/var/lib/ddos-detection/models
```

Update `app.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret')
```

### 2. Database Security

```bash
# Create dedicated directory
sudo mkdir -p /var/lib/ddos-detection
sudo chown www-data:www-data /var/lib/ddos-detection
sudo chmod 700 /var/lib/ddos-detection

# Initialize database
python -c "from app import app; app.app_context().push()"
```

### 3. HTTPS/SSL Certificates

```bash
# Let's Encrypt (free)
certbot certonly --standalone -d your_domain.com

# Or use self-signed for internal deployment
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365
```

### 4. Rate Limiting

Add to `app.py`:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...existing code...
```

### 5. Authentication Enhancement

```python
# Use stronger session settings
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12)
)
```

## Monitoring and Logging

### 1. Application Logging

Add to `app.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    handler = RotatingFileHandler(
        'ddos_detection.log',
        maxBytes=10485760,
        backupCount=10
    )
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.info('DDoS Detection Application Started')
```

### 2. Monitor Models

Create `monitoring.py`:
```python
import os
import json
from datetime import datetime

def check_model_health():
    """Verify all models are loaded"""
    from app import AVAILABLE_MODELS, NETWORK_TYPES
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for net_type in NETWORK_TYPES:
        models = AVAILABLE_MODELS.get(net_type, {})
        status['models'][net_type] = {
            'lstm': models.get('lstm') is not None,
            'svm': models.get('svm') is not None,
            'tcn': models.get('tcn') is not None
        }
    
    return status

if __name__ == '__main__':
    print(json.dumps(check_model_health(), indent=2))
```

Run periodically:
```bash
# Check every 5 minutes
*/5 * * * * cd /path/to/app && python monitoring.py >> monitoring.log
```

### 3. Health Check Endpoint

Add to `app.py`:
```python
@app.route('/health')
def health():
    """Health check endpoint"""
    from app import AVAILABLE_MODELS, NETWORK_TYPES
    
    all_available = all(
        AVAILABLE_MODELS.get(net_type, {}).get('lstm')
        for net_type in NETWORK_TYPES
    )
    
    return jsonify({
        'status': 'healthy' if all_available else 'degraded',
        'models_available': all_available,
        'timestamp': datetime.now().isoformat()
    })
```

## Performance Optimization

### 1. Model Caching

```python
# Models already cached in AVAILABLE_MODELS globally
# No need for per-request loading
```

### 2. Prediction Optimization

```python
# Batch predictions if possible
def batch_predict(network_type, data_list):
    """Batch predict multiple flows"""
    X = np.array(data_list)
    X_scaled = AVAILABLE_SCALERS[network_type].transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    lstm_emb = AVAILABLE_MODELS[network_type]['lstm'](X_tensor)
    predictions = torch.sigmoid(SVM(lstm_emb)).cpu().detach().numpy()
    
    return predictions
```

### 3. Async Predictions (Optional)

```python
# Use Celery for long-running predictions
from celery import Celery

celery = Celery(app.name)

@celery.task
def predict_async(network_type, features):
    # Prediction logic here
    pass
```

## Backup and Recovery

### 1. Database Backup

```bash
# Schedule daily SQLite backup
0 2 * * * cp /var/lib/ddos-detection/users.db \
  /backup/ddos-detection/users.db.$(date +\%Y\%m\%d)
```

### 2. Model Backup

```bash
# Archive trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Store in secure location
scp models_backup_*.tar.gz backup_server:/backups/
```

### 3. Configuration Backup

```bash
# Save configuration and environment
cp .env .env.backup
cp network_config.py network_config.py.backup
```

## Troubleshooting

### Issue: Models not loading

```bash
# Check models exist
ls -la models/ | grep -E "_scaler|_lstm|_tcn"

# Check PyTorch version compatibility
python -c "import torch; print(torch.__version__)"

# Check file permissions
sudo chown www-data:www-data models/*.pkl
```

### Issue: High memory usage

```bash
# Reduce model workers
gunicorn -w 2 app:app  # Fewer workers

# Check for memory leaks
top -p $(pgrep -f gunicorn | head -1)
```

### Issue: Slow predictions

```bash
# Profile prediction function
python -m cProfile -s cumtime app.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Maintenance Schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Check model health | Daily | `python monitoring.py` |
| Database backup | Daily | Cron job |
| Log rotation | Weekly | Automatic (RotatingFileHandler) |
| Model retraining | Monthly | `python train_network_model.py` |
| Security updates | As needed | `pip install --upgrade` |
| Database cleanup | Monthly | Delete old logs |

## Success Indicators

✅ All 4 network types selectable  
✅ Predictions return < 100ms  
✅ No memory leaks  
✅ Models load on startup  
✅ Database persists correctly  
✅ Health check endpoint responds  
✅ HTTPS working  
✅ Rate limiting active  
✅ Logs recorded  
✅ Backups automated  
