# DDoS Detection System - Documentation Index

Welcome to the Multi-Network DDoS Detection System! This documentation index will help you navigate all the resources available.

## 🚀 Getting Started

### New to the Project?
1. Start with **[README.md](README.md)** - Project overview and features
2. Follow **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start examples
3. Try the web interface or command-line examples

### Just Want to Run It?
```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py

# Open http://localhost:5000 in your browser
```

## 📚 Documentation Files

### Core Documentation

| File | Purpose | For Whom |
|------|---------|----------|
| **[README.md](README.md)** | Project overview, features, architecture | Everyone |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Quick examples, common tasks, CLI usage | Users & Developers |
| **[MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md)** | Detailed technical setup, advanced configuration | Developers & DevOps |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Production deployment, security, monitoring | DevOps & Sysadmins |
| **[CHANGES.md](CHANGES.md)** | What's new, migration guide, architecture | Project Managers & Developers |

### Code Files

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| **app.py** | Flask web application | `predict()`, `dashboard()`, model loaders |
| **network_config.py** | Network type definitions | `NETWORK_TYPES`, `get_network_config()` |
| **train_network_model.py** | Model training utility | `LSTMModel`, `TCNBiGRU`, `train_lstm_svm()` |
| **generate_sample_datasets.py** | Synthetic data generation | `generate_network_data()` |

## 🎯 Feature Overview

### Supported Network Types

```
┌─────────────────────────────────────┐
│   4 Network Types Supported         │
├─────────────────────────────────────┤
│ 1. SDN-Assisted Network    (19 ft)  │
│ 2. Traditional Network     (17 ft)  │
│ 3. IoT Network             (16 ft)  │
│ 4. Hybrid Network          (17 ft)  │
└─────────────────────────────────────┘
```

### Dual Model Ensemble

```
LSTM+SVM        TCN+BiGRU
    ↓               ↓
    └───────┬───────┘
            ↓
    Ensemble Prediction
    + Confidence Score
```

## 💡 Common Use Cases

### Use Case 1: Using the Web Dashboard
**Goal:** Make DDoS predictions through web interface

1. Open http://localhost:5000
2. Login with credentials
3. Select network type from dropdown
4. Enter flow metrics
5. Get prediction with confidence score

**Documentation:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md#web-interface-quick-steps)

### Use Case 2: Training Models
**Goal:** Create trained models for a specific network type

1. Prepare training data CSV
2. Run: `python train_network_model.py --network-type traditional --data data.csv`
3. Models automatically made available

**Documentation:** [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#adding-a-new-network-type)

### Use Case 3: Generating Sample Data
**Goal:** Create synthetic network data for testing

1. Run: `python generate_sample_datasets.py --output data/`
2. Use generated CSV files for training

**Documentation:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md#generate-sample-data-for-all-network-types)

### Use Case 4: Production Deployment
**Goal:** Deploy system to production environment

1. Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Train models for each network type
3. Configure Nginx/Gunicorn
4. Set up monitoring and backups

**Documentation:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Use Case 5: Adding New Network Type
**Goal:** Support a custom network architecture (e.g., Satellite, 5G)

1. Define network in [network_config.py](network_config.py)
2. Prepare training data with required features
3. Train with `train_network_model.py`
4. Automatically available in dashboard

**Documentation:** [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#adding-a-new-network-type)

## 🔄 Architecture Overview

```
User Interface (Web)
        ↓
    Flask App (app.py)
        ↓
Network Config (network_config.py)
    ├── SDN Features → Models
    ├── Traditional Features → Models
    ├── IoT Features → Models
    └── Hybrid Features → Models
        ↓
  Predictions with Confidence
```

## 📊 Project Structure

```
.
├── Documentation
│   ├── README.md (← Start here!)
│   ├── QUICK_REFERENCE.md
│   ├── MULTI_NETWORK_SETUP.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── CHANGES.md
│   └── INDEX.md (← You are here)
│
├── Application Code
│   ├── app.py (Flask application)
│   ├── network_config.py (Network definitions)
│   ├── requirements.txt (Dependencies)
│   └── users.db (User database, auto-created)
│
├── Utilities
│   ├── train_network_model.py (Model training)
│   └── generate_sample_datasets.py (Data generation)
│
├── Web Interface
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html (← Multi-network UI)
│   │   ├── login.html
│   │   └── signup.html
│   └── static/
│       ├── style.css
│       └── metrics.png
│
├── Trained Models
│   └── models/
│       ├── sdn_scaler.pkl
│       ├── sdn_lstm_svm_fed_model.pkl
│       ├── sdn_tcn_bigru_fed_model.pkl
│       └── [network_type]_*.pkl
│
└── Data
    └── data/ (Sample/training data)
```

## 🚦 Quick Navigation

### By Role

**🧑‍💼 Project Manager**
- Read: [README.md](README.md#features), [CHANGES.md](CHANGES.md)
- Focus: Features, timeline, status

**👨‍💻 Developer**
- Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md), [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md)
- Do: Run training, add new networks, modify models

**🔧 DevOps/SysAdmin**
- Read: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md), [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#troubleshooting)
- Do: Deploy, monitor, backup, troubleshoot

**👨‍🏫 Data Scientist**
- Read: All files, especially model architecture sections
- Focus: Training, validation, model improvements

**📊 End User**
- Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md#web-interface-quick-steps)
- Do: Login, select network, enter metrics, view predictions

### By Task

| Task | Documentation |
|------|----------------|
| Run the app | [README.md](README.md#running-the-application) |
| Use dashboard | [QUICK_REFERENCE.md](QUICK_REFERENCE.md#web-interface-quick-steps) |
| Train models | [QUICK_REFERENCE.md](QUICK_REFERENCE.md#train-a-model-for-traditional-network) |
| Generate data | [QUICK_REFERENCE.md](QUICK_REFERENCE.md#generate-sample-data-for-all-network-types) |
| Add new network | [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#adding-a-new-network-type) |
| Deploy to prod | [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) |
| Troubleshoot | [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#troubleshooting) |
| API usage | [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#using-the-system) |

## 🎓 Learning Path

### Beginner Path (30 minutes)
1. Read [README.md](README.md) sections: Overview, Features, Quick Start
2. Run: `python app.py`
3. Use web dashboard to make a prediction
4. Select different network types

### Intermediate Path (2-3 hours)
1. Complete Beginner Path
2. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Generate sample data: `python generate_sample_datasets.py`
4. Explore file structure: `ls -la models/ templates/`

### Advanced Path (1 day)
1. Complete Intermediate Path
2. Read [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md)
3. Train model: `python train_network_model.py --network-type traditional`
4. Add new network type to [network_config.py](network_config.py)
5. Train and test new network

### Expert Path (3+ days)
1. Complete Advanced Path
2. Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
3. Set up production deployment with Gunicorn + Nginx
4. Implement monitoring and health checks
5. Create automated model retraining pipeline

## ❓ FAQ Quick Answers

**Q: How do I add a new network type?**
A: See [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#adding-a-new-network-type) - Steps 1-5

**Q: How do I deploy to production?**
A: Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Pre-deployment through Maintenance

**Q: How do I train models?**
A: `python train_network_model.py --network-type <type> --data <csv>`
See [QUICK_REFERENCE.md](QUICK_REFERENCE.md#command-line-examples)

**Q: What if models don't exist for my network?**
A: System falls back to SDN models automatically (from [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#model-fallback-mechanism))

**Q: How are predictions made?**
A: Ensemble of LSTM+SVM and TCN+BiGRU outputs (from [README.md](README.md#model-architecture))

**Q: Can I use this for my custom network type?**
A: Yes! See "Adding a New Network Type" in [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md)

## 🆘 Support & Resources

### Getting Help

1. **Check Documentation:** Search relevant docs for keywords
2. **Check Troubleshooting:** [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#troubleshooting)
3. **Check Examples:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. **Review Code:** Comments in source files

### Common Questions Answered In

| Question | File | Section |
|----------|------|---------|
| "What network types are supported?" | [README.md](README.md#supported-network-types) | Supported Network Types |
| "How do I start the app?" | [README.md](README.md#running-the-application) | Running the Application |
| "What are features for SDN?" | [QUICK_REFERENCE.md](QUICK_REFERENCE.md#feature-lists-by-network-type) | Feature Lists |
| "How do I add a network?" | [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#adding-a-new-network-type) | Adding Network Type |
| "How do I deploy?" | [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Entire file |
| "How do I fix errors?" | [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md#troubleshooting) | Troubleshooting |

## ✨ Key Updates in This Version

✅ **Added:** Support for 3 new network types (Traditional, IoT, Hybrid)
✅ **Added:** Dynamic form UI based on selected network
✅ **Added:** Confidence scoring for predictions
✅ **Added:** Model training utility script
✅ **Added:** Synthetic data generation script
✅ **Added:** Comprehensive documentation (4 guides)
✅ **Improved:** Model loading with fallback mechanism
✅ **Improved:** Network-aware feature extraction
✅ **Improved:** Dashboard with network selector

See [CHANGES.md](CHANGES.md) for detailed information.

## 📈 What's Next?

1. **Train Models:** Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#2-train-models-for-each-network-type)
2. **Test System:** Use all 4 network types in dashboard
3. **Deploy:** Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#production-deployment)
4. **Monitor:** Set up health checks and logging
5. **Extend:** Add custom network types as needed

---

## 🎯 Quick Links

- **Start Here:** [README.md](README.md)
- **Quick Examples:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Setup Details:** [MULTI_NETWORK_SETUP.md](MULTI_NETWORK_SETUP.md)
- **Deploy to Prod:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **What Changed:** [CHANGES.md](CHANGES.md)

---

**Last Updated:** 2026-02-09
**Version:** 2.0 (Multi-Network Support)
**Status:** Ready for Production ✓
