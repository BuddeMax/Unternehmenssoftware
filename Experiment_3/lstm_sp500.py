import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Für die Datumsformatierung
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import time


# =============================================================================
# 1. Daten einlesen
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absoluter Pfad des Skripts
data_path = os.path.join(script_dir, "../shared_resources/sp500_data/SP500_Index_Historical_Data.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Datei nicht gefunden unter: {data_path}")
else:
    print(f"Verwende Daten von: {data_path}")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
close_prices = df[['Close']].values

# =============================================================================
# 2. Daten normalisieren basierend auf Trainingsdaten
# =============================================================================
look_back = 60  # Erhöhtes Zeitfenster

train_size = int(len(close_prices) * 0.8)
train_data = close_prices[:train_size]
test_data = close_prices[train_size - look_back:]  # Überlappung für look_back

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)


# =============================================================================
# 3. Datensätze erstellen
# =============================================================================
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


X_train, y_train = create_dataset(scaled_train, look_back)
X_test, y_test = create_dataset(scaled_test, look_back)

# Konvertierung zu PyTorch-Tensoren
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =============================================================================
# 4. Trainings- und Testdaten aufteilen
# =============================================================================
dates_train = df['Date'].iloc[look_back:train_size].reset_index(drop=True)
dates_test = df['Date'].iloc[train_size:].reset_index(drop=True)  # Korrigierte Zeile

# Überprüfen der Längen
assert len(dates_test) == len(y_test), "Längen von dates_test und y_test stimmen nicht überein."

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# =============================================================================
# 5. DataLoader erstellen
# =============================================================================
batch_size = 64  # Erhöhte Batch-Größe

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# 6. LSTM-Modell definieren
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# =============================================================================
# 7. Gerät (CPU/GPU) konfigurieren
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Gerät: {device}")

# =============================================================================
# 8. Modell, Verlustfunktion und Optimierer initialisieren
# =============================================================================
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Entfernt verbose=True

# =============================================================================
# 9. Modell trainieren
# =============================================================================
num_epochs = 100
clip_value = 1.0  # Wert für Gradient Clipping

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_loader.dataset)

    # Validierungsphase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(test_loader.dataset)
    scheduler.step(val_loss)

    # Optional: Protokolliere die aktuelle Lernrate
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}, Lernrate: {current_lr:.6f}")

# =============================================================================
# 10. Vorhersagen treffen
# =============================================================================
model.eval()
with torch.no_grad():
    train_preds = model(X_train.to(device)).cpu().numpy()
    test_preds = model(X_test.to(device)).cpu().numpy()

# =============================================================================
# 11. Rückskalieren der Vorhersagen
# =============================================================================
train_preds = scaler.inverse_transform(train_preds)
y_train_actual = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
test_preds = scaler.inverse_transform(test_preds)
y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# =============================================================================
# 12. Prozentuale Abweichung berechnen
# =============================================================================
percentage_deviation_train = ((train_preds.flatten() - y_train_actual.flatten()) / y_train_actual.flatten()) * 100
percentage_deviation_test = ((test_preds.flatten() - y_test_actual.flatten()) / y_test_actual.flatten()) * 100

# =============================================================================
# 13. Mean Absolute Percentage Error (MAPE) berechnen
# =============================================================================
mape_train = np.mean(np.abs(percentage_deviation_train))
mape_test = np.mean(np.abs(percentage_deviation_test))
print(f"Mean Absolute Percentage Error (MAPE) der Trainingsdaten: {mape_train:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE) der Testdaten: {mape_test:.2f}%")

# =============================================================================
# 14. Ergebnisse visualisieren (nur Test-Plot anzeigen)
# =============================================================================
plt.figure(figsize=(14, 7))

# Plot der tatsächlichen Testdaten
plt.plot(dates_test, y_test_actual.flatten(), label='Tatsächliche Testdaten', color='blue')

# Plot der vorhergesagten Testdaten
plt.plot(dates_test, test_preds.flatten(), label='Vorhersage Testdaten', color='orange')

# Diagrammbeschriftungen und Legende
plt.xlabel('Datum')
plt.ylabel('Close Preis')
plt.title('LSTM Vorhersage des S&P 500 Close Preises (Testdaten)')
plt.legend()

# Plot anzeigen
plt.show()


# =============================================================================
# 15. CSV mit Vorhersagen, tatsächlichen Werten und prozentualer Abweichung erstellen
# =============================================================================

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
output_dir = os.path.join(script_dir, 'lstm_sp500_data')
os.makedirs(output_dir, exist_ok=True)  # Erstellt das Verzeichnis, falls es nicht existiert

output_path = os.path.join(output_dir, 'lstm_sp500_results_5.csv')

# Erstellung eines DataFrames für die Trainingsvorhersagen
train_df = pd.DataFrame({
    'Date': dates_train,
    'Actual_Close': y_train_actual.flatten(),
    'Predicted_Close': train_preds.flatten(),
    'Percentage_Deviation': percentage_deviation_train
})

# Erstellung eines DataFrames für die Testvorhersagen inklusive prozentualer Abweichung
test_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test_actual.flatten(),
    'Predicted_Close': test_preds.flatten(),
    'Percentage_Deviation': percentage_deviation_test
})

# Kombination der Trainings- und Testdaten in einem vollständigen DataFrame
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values('Date').reset_index(drop=True)

# Speicherung des vollständigen DataFrames als CSV-Datei ohne den Index
full_df.to_csv(output_path, index=False)

print(
    f"CSV-Datei mit den Vorhersagen, tatsächlichen Werten und prozentualen Abweichungen wurde unter '{output_path}' gespeichert.")
print(f"Mean Absolute Percentage Error (MAPE) der Trainingsdaten: {mape_train:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE) der Testdaten: {mape_test:.2f}%")
