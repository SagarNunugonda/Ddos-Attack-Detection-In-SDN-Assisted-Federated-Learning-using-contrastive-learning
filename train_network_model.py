"""
Utility script for training network-type-specific models.
This script helps prepare models for new network types.

Usage:
    python train_network_model.py --network-type traditional --data path/to/data.csv --output models/
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from xgboost import XGBClassifier
import sys
from network_config import get_network_config, get_feature_count


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


def prepare_data(csv_file, network_type, label_column='label'):
    """
    Load and prepare data for a specific network type.
    
    Args:
        csv_file: Path to CSV file
        network_type: Network type from network_config
        label_column: Name of label column (0=Normal, 1=Attack)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    # Drop unnamed index column if present
    if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])
    
    # Get required features
    config = get_network_config(network_type)
    features = config["features"]
    
    # Check if all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in CSV: {missing_features}")
    
    if label_column not in df.columns:
        if 'Label' in df.columns:
            label_column = 'Label'
        else:
            raise ValueError(f"Label column '{label_column}' not found in CSV.")

    X = df[features].values.astype(np.float32)
    y = df[label_column].values.astype(np.int32)
    
    # Scale features
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    print("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Class distribution - Normal: {(y==0).sum()}, Attack: {(y==1).sum()}")
    
    return X_train, X_test, y_train, y_test, scaler, features


def train_lstm_svm(X_train, y_train, X_test, y_test, input_size, 
                   epochs=50, batch_size=32, learning_rate=0.001):
    """Train LSTM+SVM model"""
    print("\n=== Training LSTM+SVM Model ===")
    
    lstm = LSTMModel(input_size=input_size, hidden_size=128)
    svm = LinearSVM(128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)
    svm.to(device)
    
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(svm.parameters()), 
                                 lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        lstm.train()
        svm.train()
        train_loss = 0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_x = X_train_t[i:i+batch_size].to(device)
            batch_y = y_train_t[i:i+batch_size].to(device)
            
            lstm_out = lstm(batch_x)
            svm_out = svm(lstm_out)
            loss = criterion(svm_out, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        lstm.eval()
        svm.eval()
        with torch.no_grad():
            lstm_out = lstm(X_test_t.to(device))
            svm_out = svm(lstm_out)
            val_loss = criterion(svm_out, y_test_t.to(device)).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return lstm, svm


def train_tcn_bigru(X_train, y_train, X_test, y_test, input_size,
                    epochs=50, batch_size=32, learning_rate=0.001):
    """Train TCN+BiGRU model"""
    print("\n=== Training TCN+BiGRU Model ===")
    
    model = TCNBiGRU(input_dim=input_size, hidden_dim=128, out_dim=128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_x = X_train_t[i:i+batch_size].to(device)
            batch_y = y_train_t[i:i+batch_size].to(device)
            
            _, out = model(batch_x)
            loss = criterion(out, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, out = model(X_test_t.to(device))
            val_loss = criterion(out, y_test_t.to(device)).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier for tabular features"""
    print("\n=== Training XGBoost Model ===")

    # Handle class imbalance if present
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    return model


def evaluate_model(model_lstm, model_svm, model_tcn, model_xgb, X_test, y_test, model_name=""):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = y_test
    
    # LSTM+SVM predictions
    if model_lstm is not None and model_svm is not None:
        model_lstm.eval()
        model_svm.eval()
        with torch.no_grad():
            lstm_out = model_lstm(X_test_t)
            svm_out = model_svm(lstm_out)
            lstm_preds = torch.sigmoid(svm_out).cpu().numpy().flatten()
        lstm_preds_binary = (lstm_preds > 0.5).astype(int)
        lstm_accuracy = (lstm_preds_binary == y_test_t).sum() / len(y_test_t)
        print(f"\n{model_name} LSTM+SVM Accuracy: {lstm_accuracy:.4f}")
    
    # TCN+BiGRU predictions
    if model_tcn is not None:
        model_tcn.eval()
        with torch.no_grad():
            _, tcn_out = model_tcn(X_test_t)
            tcn_preds = torch.sigmoid(tcn_out).cpu().numpy().flatten()
        tcn_preds_binary = (tcn_preds > 0.5).astype(int)
        tcn_accuracy = (tcn_preds_binary == y_test_t).sum() / len(y_test_t)
        print(f"{model_name} TCN+BiGRU Accuracy: {tcn_accuracy:.4f}")

    # XGBoost predictions
    if model_xgb is not None:
        xgb_probs = model_xgb.predict_proba(X_test)[:, 1]
        xgb_preds = (xgb_probs > 0.5).astype(int)
        xgb_accuracy = accuracy_score(y_test_t, xgb_preds)
        xgb_precision = precision_score(y_test_t, xgb_preds, zero_division=0)
        print(f"{model_name} XGBoost Accuracy: {xgb_accuracy:.4f}")
        print(f"{model_name} XGBoost Precision: {xgb_precision:.4f}")


def save_models(lstm, svm, tcn, xgb, scaler, network_type, output_dir="models/"):
    """Save trained models and scaler"""
    print(f"\nSaving models to {output_dir}...")
    
    # Save scaler
    scaler_path = f"{output_dir}{network_type}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Save LSTM+SVM
    if lstm is not None:
        lstm_path = f"{output_dir}{network_type}_lstm_svm_fed_model.pkl"
        joblib.dump(lstm.state_dict(), lstm_path)
        print(f"Saved LSTM+SVM model: {lstm_path}")
    
    # Save TCN+BiGRU
    if tcn is not None:
        tcn_path = f"{output_dir}{network_type}_tcn_bigru_fed_model.pkl"
        joblib.dump(tcn.state_dict(), tcn_path)
        print(f"Saved TCN+BiGRU model: {tcn_path}")

    # Save XGBoost
    if xgb is not None:
        xgb_path = f"{output_dir}{network_type}_xgboost_model.pkl"
        joblib.dump(xgb, xgb_path)
        print(f"Saved XGBoost model: {xgb_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train network-type-specific models for DDoS detection'
    )
    parser.add_argument('--network-type', required=True,
                       choices=['sdn', 'traditional', 'iot', 'hybrid'],
                       help='Network type to train model for')
    parser.add_argument('--data', required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--output', default='models/',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, features = prepare_data(
            args.data, args.network_type
        )
        
        input_size = len(features)
        
        # Train models
        lstm, svm = train_lstm_svm(
            X_train, y_train, X_test, y_test, input_size,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        tcn = train_tcn_bigru(
            X_train, y_train, X_test, y_test, input_size,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        xgb = train_xgboost(X_train, y_train, X_test, y_test)
        
        # Evaluate
        evaluate_model(lstm, svm, tcn, xgb, X_test, y_test, f"{args.network_type.upper()}")
        
        # Save models
        save_models(lstm, svm, tcn, xgb, scaler, args.network_type, args.output)
        
        print("\n✓ Model training complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
