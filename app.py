import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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

class CNNBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bigru = nn.GRU(32, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_embed = nn.Linear(hidden_dim * 2, out_dim)
        self.fc_out = nn.Linear(out_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = F.relu(self.cnn(x))
        rnn_out, _ = self.bigru(cnn_out.permute(0, 2, 1))
        emb = self.dropout(F.relu(self.fc_embed(rnn_out[:, -1, :])))
        out = self.fc_out(emb)
        return emb, out


input_size = 19
scaler = joblib.load("models/minmax_scaler.pkl")
lstm = LSTMModel(input_size=input_size, hidden_size=128)
svm = LinearSVM(128)
cnn = CNNBiGRU(input_dim=input_size)

lstm.load_state_dict(joblib.load("models/lstm_svm_fed_model.pkl"))
cnn.load_state_dict(joblib.load("models/cnn_bigru_fed_model.pkl"))
lstm.eval(), cnn.eval(), svm.eval()


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

    # Plot metrics
    models = ["LSTM+SVM", "CNN+BiGRU"]
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

    return render_template('dashboard.html', username=session['username'], graph=graph_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form.get(f)) for f in [
            'dt','switch','pktcount','bytecount','dur','dur_nsec','tot_dur','flows',
            'packetins','pktperflow','byteperflow','pktrate','Pairflow','port_no',
            'tx_bytes','rx_bytes','tx_kbps','rx_kbps','tot_kbps'
        ]]
        X = np.array(data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        emb = lstm(X_tensor)
        out1 = svm(emb)
        pred1 = torch.sigmoid(out1).item()

        emb2, out2 = cnn(X_tensor)
        pred2 = torch.sigmoid(out2).item()

        final_pred = (pred1 + pred2) / 2
        label = "Botnet Attack" if final_pred > 0.5 else "Normal Traffic"

        return render_template('dashboard.html', username=session['username'], 
                               prediction=label, pred1=pred1, pred2=pred2, graph="static/metrics.png")

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run()
