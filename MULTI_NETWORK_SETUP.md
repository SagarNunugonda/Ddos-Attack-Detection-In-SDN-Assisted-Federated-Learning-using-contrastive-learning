# Multi-Network DDoS Detection System Setup Guide

## Overview

The DDoS Attack Detection system has been scaled from supporting only **SDN-Assisted Networks** to supporting **multiple network types**:

### Supported Network Types

1. **SDN-Assisted Network** (19 features)
   - Software-Defined Networks with centralized control
   - Includes OpenFlow switch metrics
   
2. **Traditional Network** (17 features)
   - Conventional routed networks
   - Uses standard TCP/IP protocol metrics
   
3. **IoT Network** (16 features)
   - Internet of Things networks with constrained devices
   - Includes device-level metrics (signal strength, battery level)
   
4. **Hybrid Network** (17 features)
   - Mixed SDN and Traditional network architecture
   - Combines features from both SDN and Traditional networks

## Architecture

### Key Components

#### 1. Network Configuration Module (`network_config.py`)
- Defines network types and their features
- Provides utility functions for feature extraction and validation
- Manages feature ordering for each network type

#### 2. Updated Flask Application (`app.py`)
- Multi-model loader with fallback mechanisms
- Network-aware model initialization
- Dynamic feature extraction based on network type
- Ensemble prediction with confidence scoring

#### 3. Enhanced Dashboard (`templates/dashboard.html`)
- Network type selector dropdown
- Dynamically generated form fields
- Real-time form updates
- Confidence score display

## Network Configuration Details

### SDN Network
```python
Features: [
    'dt', 'switch', 'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 
    'flows', 'packetins', 'pktperflow', 'byteperflow', 'pktrate', 
    'Pairflow', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
]
```

### Traditional Network
```python
Features: [
    'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows',
    'pktperflow', 'byteperflow', 'pktrate', 'src_port', 'dst_port',
    'protocol', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
]
# Train TCN+BiGRU model

### IoT Network
```python
joblib.dump(tcn.state_dict(), "models/my_network_tcn_bigru_fed_model.pkl")
    'pktcount', 'bytecount', 'dur', 'flows', 'pktperflow', 'byteperflow',
    'pktrate', 'device_id', 'signal_strength', 'battery_level', 'error_rate',
    'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
]
```

### Hybrid Network
```python
models/{network_type}_tcn_bigru_fed_model.pkl
    'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows',
    'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'port_no',
    'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps', 'routing_type'
]
```

## Adding a New Network Type

To add a new network type to the system:

│   ├── tcn_bigru_fed_model.pkl     # SDN TCN+BiGRU (fallback)

```python
NETWORK_TYPES = {
    # ... existing types ...
    "my_network": {
        "name": "My Custom Network",
        "description": "Description of your network type",
        "features": [
            'feature1', 'feature2', 'feature3',
            # ... list all features ...
        ],
        "model_prefix": "my_network",
        "feature_count": 20  # Update with actual count
    }
}
```

### Step 2: Prepare Training Data

Create datasets for your network type with the same feature order:
- Training data CSV file
- Test data CSV file

### Step 3: Train Models

Create scalers and models for your network:

```python
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch

# Create and fit scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "models/my_network_scaler.pkl")

# Train LSTM+SVM model
# Train TCN+BiGRU model

# Save models
joblib.dump(lstm.state_dict(), "models/my_network_lstm_svm_fed_model.pkl")
joblib.dump(tcn.state_dict(), "models/my_network_tcn_bigru_fed_model.pkl")
```

### Step 4: Model File Naming Convention

All model files should follow this naming pattern:
```
models/{network_type}_scaler.pkl
models/{network_type}_lstm_svm_fed_model.pkl
models/{network_type}_tcn_bigru_fed_model.pkl
```

**Note:** If network-specific models are not found, the system falls back to SDN models.

## Using the System

### Via Web UI

1. Go to `http://localhost:5000/login`
2. Login with your credentials
3. On the Dashboard:
   - Select your network type from the dropdown
   - Fill in the required features for that network
   - Click "Predict"
4. View the prediction result with confidence score

### Via API

#### Get Network Features
```bash
GET http://localhost:5000/api/network-features/<network_type>

# Example
GET http://localhost:5000/api/network-features/sdn

# Response
{
    "network_type": "sdn",
    "name": "SDN-Assisted Network",
    "description": "...",
    "features": ["dt", "switch", ...]
}
```

#### Make Prediction
```bash
POST http://localhost:5000/predict

# Form data includes:
# - network_type: sdn|traditional|iot|hybrid
# - All required features for selected network type
```

## Model Ensemble Strategy

The system uses an ensemble approach:

1. **LSTM+SVM Model**
   - LSTM extracts temporal patterns
   - SVM performs binary classification

2. **TCN+BiGRU Model**
    - TCN extracts temporal/spatial patterns via dilated convs
    - Bidirectional GRU captures sequence dependencies

3. **Ensemble Prediction**
   - Averages predictions from both models
   - Computes confidence score: `|prediction - 0.5| * 2 * 100`
   - Threshold: prediction > 0.5 → Attack, ≤ 0.5 → Normal

## Model Fallback Mechanism

The system implements intelligent fallback:

```
Try to load: models/{network_type}_MODELNAME.pkl
    ↓ (if found)
    Use network-specific model
    ↓ (if not found)
    Fallback to: models/MODELNAME.pkl (SDN model)
    ↓ (if not found)
    Model unavailable warning
```

This allows running the system even if network-specific models are not trained yet.

## Feature Scaling

Each network type can have its own scaler:
- **Primary:** `models/{network_type}_scaler.pkl`
- **Fallback:** `models/minmax_scaler.pkl` (SDN scaler)
- **No scaler:** Raw data used as-is

## Performance Metrics by Network Type

(Note: Update these after training models for each network type)

| Network Type | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| SDN | 0.8725 | 0.9312 | 0.8334 | 0.8796 |
| Traditional | (To be trained) | (To be trained) | (To be trained) | (To be trained) |
| IoT | (To be trained) | (To be trained) | (To be trained) | (To be trained) |
| Hybrid | (To be trained) | (To be trained) | (To be trained) | (To be trained) |

## Directory Structure

```
├── app.py                           # Flask application
├── network_config.py                # Network configuration
├── models/
│   ├── minmax_scaler.pkl           # SDN scaler (fallback)
│   ├── lstm_svm_fed_model.pkl      # SDN LSTM+SVM (fallback)
│   ├── tcn_bigru_fed_model.pkl     # SDN TCN+BiGRU (fallback)
│   ├── traditional_scaler.pkl
│   ├── traditional_lstm_svm_fed_model.pkl
│   ├── traditional_tcn_bigru_fed_model.pkl
│   ├── iot_scaler.pkl
│   ├── iot_lstm_svm_fed_model.pkl
│   ├── iot_tcn_bigru_fed_model.pkl
│   ├── hybrid_scaler.pkl
│   ├── hybrid_lstm_svm_fed_model.pkl
│   ├── hybrid_tcn_bigru_fed_model.pkl
│   └── ...
├── templates/
│   ├── base.html
│   ├── dashboard.html               # Updated with network selection
│   ├── login.html
│   └── signup.html
└── static/
    ├── style.css
    └── metrics.png
```

## Troubleshooting

### Issue: "Models not available for network type"
**Solution:** Check if model files exist in `models/` directory with correct naming convention.

### Issue: Prediction gives wrong values
**Solution:** Ensure scaler exists for your network type. If not, check that the feature order matches exactly.

### Issue: Form shows no fields
**Solution:** Clear browser cache and refresh. Ensure JavaScript is enabled.

### Issue: Feature validation fails
**Solution:** Verify that all required features are included in the form data and are numeric values.

## Future Enhancements

1. **Dynamic Model Registration:** Add UI to upload new models
2. **Transfer Learning:** Adapt SDN models to new network types with less training data
3. **Real-time Monitoring:** Dashboard showing live network traffic predictions
4. **Model Comparison:** Side-by-side comparison of models across network types
5. **Feature Importance:** Visualize which features are most important for each network
6. **A/B Testing:** Test different model architectures per network type
