import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Dynamischen Pfad zur Datei definieren
script_dir = os.path.dirname(__file__)  # Verzeichnis des aktuellen Skripts
file_path = os.path.join(script_dir, "../sp500_data/SP500_Index_Historical_Data.csv")

# Daten laden
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Trainings- und Testdaten aufteilen
train_data = data[(data['Date'] >= '1994-01-01') & (data['Date'] <= '2015-12-31')]
test_data = data[(data['Date'] >= '2016-01-01') & (data['Date'] <= '2024-12-31')]

features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[features])
test_scaled = scaler.transform(test_data[features])


def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, 3])  # 'Close' ist an Index 3
    return np.array(sequences), np.array(labels)


seq_length = 20  # Optimierte Sequenzlänge
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# LSTM-Modell mit Optimierungen
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)  # Dropout zur Regularisierung
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 wegen Bidirektionalität

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Dropout auf die letzte LSTM-Ausgabe anwenden
        out = self.fc(out)
        return out


input_size = 5
hidden_size = 128  # Erhöhte Hidden-Size
output_size = 1
num_layers = 3  # Mehrere LSTM-Schichten

model = StockLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.SmoothL1Loss()  # Robuster gegenüber Ausreißern
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2-Regularisierung
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Lernraten-Scheduler

# Training
num_epochs = 100  # Erhöhte Anzahl der Epochen
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()  # Lernrate anpassen
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(X_train_tensor):.6f}")

# Testen
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()
    test_loss = criterion(torch.tensor(predictions), y_test_tensor.squeeze()).item()

print(f"Test Loss: {test_loss:.6f}")

# Denormalisieren der Daten für den Plot
y_test_denormalized = scaler.inverse_transform(np.hstack((
    np.zeros((len(y_test), 3)),  # Dummy-Spalten für 'Open', 'High', 'Low'
    y_test.reshape(-1, 1),
    np.zeros((len(y_test), 1))  # Dummy-Spalte für 'Volume'
)))[:, 3]

predictions_denormalized = scaler.inverse_transform(np.hstack((
    np.zeros((len(predictions), 3)),
    predictions.reshape(-1, 1),
    np.zeros((len(predictions), 1))
)))[:, 3]

# Plotten der Ergebnisse
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'][seq_length:], y_test_denormalized, label='Tatsächliche Werte', alpha=0.7)
plt.plot(test_data['Date'][seq_length:], predictions_denormalized, label='Vorhersagen', alpha=0.7)
plt.xlabel('Datum')
plt.ylabel('SP500 Schlusskurs')
plt.title('Tatsächliche vs Vorhergesagte Werte')
plt.legend()
plt.show()