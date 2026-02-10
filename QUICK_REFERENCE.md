# Quick Reference Guide: Multi-Network DDoS Detection

## 📱 Web Interface Quick Steps

### For SDN Networks
1. Login to dashboard
2. Select **"SDN-Assisted Network"** from dropdown
3. Fill form with OpenFlow metrics:
   - dt, switch, pktcount, bytecount, dur, etc.
4. Click "Predict"
5. View results with confidence score

### For Traditional Networks  
1. Login to dashboard
2. Select **"Traditional Network"** from dropdown
3. Fill form with TCP/IP metrics:
   - pktcount, bytecount, src_port, dst_port, protocol, etc.
4. Click "Predict"
5. View results

### For IoT Networks
1. Login to dashboard
2. Select **"IoT Network"** from dropdown
3. Fill form with device metrics:
   - device_id, signal_strength, battery_level, error_rate, etc.
4. Click "Predict"
5. View results

### For Hybrid Networks
1. Login to dashboard
2. Select **"Hybrid Network"** from dropdown
3. Fill form with mixed metrics:
   - Combination of SDN and Traditional features
   - routing_type indicator
4. Click "Predict"
5. View results

## 💻 Command Line Examples

### Generate Sample Data for All Network Types
```bash
python generate_sample_datasets.py --output data/ --samples 5000
```

### Train a Model for Traditional Network
```bash
python train_network_model.py \
  --network-type traditional \
  --data data/traditional_data.csv \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

### Train a Model for IoT Network
```bash
python train_network_model.py \
  --network-type iot \
  --data data/iot_data.csv \
  --epochs 100
```

### Train All Models at Once
```bash
for network in sdn traditional iot hybrid; do
  python train_network_model.py \
    --network-type $network \
    --data data/${network}_data.csv
done
```

## 🔄 Workflow: Adding Support for a New Network Type

### Step 1: Define Network Configuration
Edit `network_config.py`:
```python
NETWORK_TYPES = {
    # ... existing ...
    "wireless": {
        "name": "Wireless Network",
        "description": "WiFi/cellular networks",
        "features": ['rssi', 'packets', 'bytes', 'duration', ...],
        "model_prefix": "wireless",
        "feature_count": 18
    }
}
```

### Step 2: Prepare Your Data
Collect data with columns matching the features list + "Label":
```
rssi,packets,bytes,duration,...,Label
-45,150,8742,2.5,...,0
-55,45000,2500000,30.2,...,1
```

### Step 3: Generate Synthetic Data (Optional)
```bash
python generate_sample_datasets.py
```

### Step 4: Train Models
```bash
python train_network_model.py \
  --network-type wireless \
  --data data/wireless_data.csv
```

### Step 5: Deploy
- Models are auto-loaded on app startup
- Select from dashboard immediately

## 📊 Feature Lists by Network Type

### SDN Network (19 features)
```
dt, switch, pktcount, bytecount, dur, dur_nsec, tot_dur, flows, packetins,
pktperflow, byteperflow, pktrate, Pairflow, port_no, tx_bytes, rx_bytes,
tx_kbps, rx_kbps, tot_kbps
```

### Traditional Network (17 features)
```
pktcount, bytecount, dur, dur_nsec, tot_dur, flows, pktperflow, byteperflow,
pktrate, src_port, dst_port, protocol, tx_bytes, rx_bytes, tx_kbps, rx_kbps,
tot_kbps
```

### IoT Network (16 features)
```
pktcount, bytecount, dur, flows, pktperflow, byteperflow, pktrate, device_id,
signal_strength, battery_level, error_rate, tx_bytes, rx_bytes, tx_kbps,
rx_kbps, tot_kbps
```

### Hybrid Network (17 features)
```
pktcount, bytecount, dur, dur_nsec, tot_dur, flows, packetins, pktperflow,
byteperflow, pktrate, port_no, tx_bytes, rx_bytes, tx_kbps, rx_kbps, tot_kbps,
routing_type
```

## 🧪 Testing Different Networks

### Quick Test: Use Generated Sample Data
```bash
# Generate samples
python generate_sample_datasets.py --output test_data/ --samples 100

# Train on smallest dataset
python train_network_model.py \
  --network-type iot \
  --data test_data/iot_data.csv \
  --epochs 5
```

### Real Data Test
```bash
# Prepare your actual network data
# Ensure all required features are present
python train_network_model.py \
  --network-type traditional \
  --data your_network_data.csv
```

## 📈 Performance Comparison

After training models for each network type:

```python
# View baseline performance in app.py dashboard route
# Or check logs at end of training:
python train_network_model.py --network-type traditional --data data/traditional_data.csv 2>&1 | grep Accuracy
```

## 🔍 Debugging Tips

### Check Available Models
```python
from network_config import NETWORK_TYPES
from app import AVAILABLE_MODELS

for net_type in NETWORK_TYPES:
    models = AVAILABLE_MODELS.get(net_type, {})
    print(f"{net_type}: {list(models.keys())}")
```

### Validate Feature Order
```bash
# Generate reference data
python generate_sample_datasets.py --output ref_data/

# Compare your data columns
import pandas as pd
ref = pd.read_csv('ref_data/traditional_data.csv')
yours = pd.read_csv('your_data.csv')
print("Expected features:", ref.columns.tolist())
print("Your features:", yours.columns.tolist())
```

### Test Prediction Without Web UI
```bash
import numpy as np
from network_config import get_features_for_network, extract_features_in_order

network_type = 'traditional'
features = get_features_for_network(network_type)
print(f"Required features for {network_type}: {features}")

# Fill with test data
data = {f: np.random.rand() for f in features}
print(f"Data prepared with {len(data)} features")
```

## 🚀 Production Deployment

### 1. Train Models for All Networks
```bash
for net in sdn traditional iot hybrid; do
  python train_network_model.py \
    --network-type $net \
    --data production_data/${net}_traffic.csv \
    --epochs 200
done
```

### 2. Verify All Models Loaded
```bash
python -c "from app import AVAILABLE_MODELS; print({k: any(v.values()) for k,v in AVAILABLE_MODELS.items()})"
```

### 3. Run Flask Server
```bash
export FLASK_ENV=production
python app.py
```

### 4. Test Each Network Type
- Visit dashboard for each network
- Verify prediction latency < 100ms
- Check confidence scores

## 🎓 Example: Custom Network Type

### Scenario: Satellite Network
```python
# Step 1: Add to network_config.py
"satellite": {
    "name": "Satellite Network",
    "description": "LEO/GEO satellite networks",
    "features": [
        'packets', 'bytes', 'duration', 'latency', 'jitter',
        'signal_quality', 'handover_count', 'link_margin', 'rain_attenuation',
        'throughput', 'error_rate', 'retransmissions', 'packet_loss',
        'uplink_power', 'downlink_power', 'beam_id'
    ],
    "model_prefix": "satellite",
    "feature_count": 16
}
```

```bash
# Step 2: Generate sample data
python generate_sample_datasets.py

# Step 3: Prepare real satellite data in CSV format
# Step 4: Train
python train_network_model.py --network-type satellite --data data/satellite_data.csv

# Step 5: Use in dashboard
# Select "Satellite Network" from dropdown and predict
```

## 📞 Support

- Check feature counts match configuration
- Verify CSV has "Label" column (0=Normal, 1=Attack)
- Ensure all numeric values parse correctly
- See MULTI_NETWORK_SETUP.md for advanced topics
