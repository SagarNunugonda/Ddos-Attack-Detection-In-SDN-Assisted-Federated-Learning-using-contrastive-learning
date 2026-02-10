import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import sqlite3
from xgboost import XGBClassifier
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from network_config import (
    NETWORK_TYPES, get_network_config, get_features_for_network,
    validate_input_data, extract_features_in_order, get_feature_count
)

app = Flask(__name__)
app.secret_key = "super_secret_key"


DB_PATH = "users.db"
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.close()

# Track available models
AVAILABLE_MODELS = {}
AVAILABLE_SCALERS = {}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = F.relu(self.fc(out))
        return out

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

class TCNBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super().__init__()
        # Simple TCN-style temporal convolution (dilated conv)
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.bigru = nn.GRU(32, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_embed = nn.Linear(hidden_dim * 2, out_dim)
        self.fc_out = nn.Linear(out_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        tcn_out = self.tcn(x)
        rnn_out, _ = self.bigru(tcn_out.permute(0, 2, 1))
        emb = self.dropout(F.relu(self.fc_embed(rnn_out[:, -1, :])))
        out = self.fc_out(emb)
        return emb, out


def load_models_for_network(network_type):
    """Load models for a specific network type"""
    models = {}
    feature_count = get_feature_count(network_type)
    
    # Try to load pretrained models
    try:
        scaler = joblib.load(f"models/{network_type}_scaler.pkl")
        AVAILABLE_SCALERS[network_type] = scaler
    except:
        # Use existing scaler as fallback
        try:
            AVAILABLE_SCALERS[network_type] = joblib.load("models/minmax_scaler.pkl")
        except:
            AVAILABLE_SCALERS[network_type] = None
    
    try:
        lstm = LSTMModel(input_size=feature_count, hidden_size=128)
        lstm.load_state_dict(joblib.load(f"models/{network_type}_lstm_svm_fed_model.pkl"))
        lstm.eval()
        models['lstm'] = lstm
    except:
        # Use SDN model as fallback for any network type
        try:
            lstm = LSTMModel(input_size=feature_count, hidden_size=128)
            lstm.load_state_dict(joblib.load("models/lstm_svm_fed_model.pkl"))
            lstm.eval()
            models['lstm'] = lstm
        except:
            print(f"Warning: Could not load LSTM model for {network_type}")
            models['lstm'] = None
    
    try:
        svm = LinearSVM(128)
        models['svm'] = svm
    except:
        models['svm'] = None
    
    try:
        tcn = TCNBiGRU(input_dim=feature_count)
        tcn.load_state_dict(joblib.load(f"models/{network_type}_tcn_bigru_fed_model.pkl"))
        tcn.eval()
        models['tcn'] = tcn
    except:
        # Use SDN model as fallback
        try:
            tcn = TCNBiGRU(input_dim=feature_count)
            tcn.load_state_dict(joblib.load("models/tcn_bigru_fed_model.pkl"))
            tcn.eval()
            models['tcn'] = tcn
        except:
            print(f"Warning: Could not load TCN model for {network_type}")
            models['tcn'] = None

    try:
        xgb = joblib.load(f"models/{network_type}_xgboost_model.pkl")
        models['xgb'] = xgb
    except:
        # Use SDN model as fallback
        try:
            xgb = joblib.load("models/sdn_xgboost_model.pkl")
            models['xgb'] = xgb
        except:
            print(f"Warning: Could not load XGBoost model for {network_type}")
            models['xgb'] = None
    
    return models


# Initialize models for all available network types
for network_type in NETWORK_TYPES.keys():
    AVAILABLE_MODELS[network_type] = load_models_for_network(network_type)


@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except:
            return render_template('signup.html', msg="Username already exists.")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', msg="Invalid credentials.")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Plot metrics for default network (SDN)
    models = ["LSTM+SVM", "TCN+BiGRU"]
    acc = [0.872506, 0.826662]
    recall = [0.931181, 0.858266]
    precision = [0.833407, 0.807274]
    f1 = [0.879585, 0.831990]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(models))
    plt.bar(x - 0.2, acc, 0.2, label='Accuracy')
    plt.bar(x, recall, 0.2, label='Recall')
    plt.bar(x + 0.2, precision, 0.2, label='Precision')
    plt.bar(x + 0.4, f1, 0.2, label='F1')
    plt.xticks(x, models)
    plt.ylabel('Score')
    plt.legend()
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    graph_path = "static/metrics.png"
    plt.savefig(graph_path)
    plt.close()

    return render_template('dashboard.html', 
                          username=session['username'], 
                          graph=graph_path,
                          network_types=NETWORK_TYPES,
                          available_models=AVAILABLE_MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        network_type = request.form.get('network_type', 'sdn')
        if network_type not in NETWORK_TYPES:
            return jsonify({"error": f"Unknown network type: {network_type}"}), 400
        
        # Get features for this network type
        features = get_features_for_network(network_type)
        
        # Extract data from form
        data_dict = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            try:
                data_dict[feature] = float(value)
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}: {value}"}), 400
        
        # Extract features in correct order
        data = extract_features_in_order(network_type, data_dict)
        X = np.array(data).reshape(1, -1)
        
        # Get scaler for this network type
        scaler = AVAILABLE_SCALERS.get(network_type)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Get models for this network type
        models = AVAILABLE_MODELS.get(network_type, {})
        lstm = models.get('lstm')
        svm = models.get('svm')
        tcn = models.get('tcn')
        xgb = models.get('xgb')

        if lstm is None or tcn is None:
            return jsonify({"error": f"Models not available for network type: {network_type}"}), 500
        
        # Make predictions
        predictions = {}
        
        try:
            emb = lstm(X_tensor)
            out1 = svm(emb)
            pred1 = torch.sigmoid(out1).item()
            predictions['lstm_svm'] = pred1
        except Exception as e:
            predictions['lstm_svm'] = None
        
        try:
            emb2, out2 = tcn(X_tensor)
            pred2 = torch.sigmoid(out2).item()
            predictions['tcn_bigru'] = pred2
        except Exception as e:
            predictions['tcn_bigru'] = None

        try:
            if xgb is not None:
                pred3 = float(xgb.predict_proba(X_scaled)[:, 1][0])
                predictions['xgboost'] = pred3
            else:
                predictions['xgboost'] = None
        except Exception as e:
            predictions['xgboost'] = None
        
        # Calculate ensemble prediction
        valid_preds = [p for p in predictions.values() if p is not None]
        if valid_preds:
            final_pred = sum(valid_preds) / len(valid_preds)
        else:
            return jsonify({"error": "No valid predictions generated"}), 500
        
        label = "Botnet Attack" if final_pred > 0.5 else "Normal Traffic"
        confidence = abs(final_pred - 0.5) * 2 * 100  # Confidence score
        
        config = get_network_config(network_type)
        
        return render_template('dashboard.html', 
                              username=session['username'], 
                              prediction=label,
                              confidence=confidence,
                              network_name=config['name'],
                              network_type=network_type,
                              pred1=predictions.get('lstm_svm'),
                              pred2=predictions.get('tcn_bigru'),
                              pred3=predictions.get('xgboost'),
                              graph="static/metrics.png",
                              network_types=NETWORK_TYPES,
                              available_models=AVAILABLE_MODELS)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/network-features/<network_type>')
def get_network_features(network_type):
    """API endpoint to get features for a network type"""
    try:
        config = get_network_config(network_type)
        return jsonify({
            "network_type": network_type,
            "name": config["name"],
            "description": config["description"],
            "features": config["features"]
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
