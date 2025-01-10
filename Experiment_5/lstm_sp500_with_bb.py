import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.dates as mdates


# =============================================================================
# 1. Daten einlesen
# =============================================================================
# Ermittelt das Verzeichnis des aktuellen Skripts
script_dir = os.path.dirname(__file__)

# Erstellt den relativen Pfad zur CSV-Datei mit Bollinger Bands
data_path = os.path.join(script_dir, "../shared_resources/sp500_data/BollingerBands/SP500_Index_Historical_Data_with_BB.csv")

# Überprüfen, ob die Datei existiert
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Datei nicht gefunden unter: {data_path}")
else:
    print(f"Verwende Daten von: {data_path}")

# Laden der Daten unter Berücksichtigung des Datumsformats
df = pd.read_csv(data_path, parse_dates=['Date'])

# Sortieren der Daten nach Datum aufsteigend und Zurücksetzen des Index
df = df.sort_values('Date').reset_index(drop=True)

# Auswahl der 'Close', 'BB_Middle', 'BB_Upper' und 'BB_Lower' Preise für die Analyse
features = df[['Close', 'BB_Middle', 'BB_Upper', 'BB_Lower']].values  # Inklusive Bollinger Bands als zusätzliche Features

# =============================================================================
# 2. Daten normalisieren
# =============================================================================

# Initialisierung des MinMaxScaler zur Skalierung der Daten auf den Bereich [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)  # Skalieren von 'Close' und Bollinger Bands

# =============================================================================
# 3. Datensätze erstellen
# =============================================================================

def create_dataset(dataset, look_back=1):
    """
    Erstellt Eingabe- und Ausgabesätze für das LSTM-Modell.

    Args:
        dataset (numpy.ndarray): Skalierte Close-Preise und Bollinger Bands.
        look_back (int): Anzahl der vorhergehenden Zeitpunkte als Features.

    Returns:
        tuple: Arrays von Eingaben (X) und Ausgaben (y).
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])  # Alle Features (Close, BB_Middle, BB_Upper, BB_Lower)
        y.append(dataset[i + look_back, 0])      # 'Close' als Zielvariable
    return np.array(X), np.array(y)

# Anzahl der vorhergehenden Zeitpunkte, die als Features verwendet werden
look_back = 1

# Erstellung der Eingabe- und Ausgabesätze
X, y = create_dataset(scaled_features, look_back)

# Konvertierung der Daten zu PyTorch-Tensoren für die Modellierung
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# =============================================================================
# 4. Trainings- und Testdaten aufteilen
# =============================================================================

# Bestimmung der Größe des Trainingsdatensatzes (80% der Daten)
train_size = int(len(X) * 0.8)

# Aufteilung der Daten in Trainings- und Testsets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Extraktion der zugehörigen Datumsangaben für Trainings- und Testdaten
dates_train = df['Date'].iloc[look_back:train_size + look_back].reset_index(drop=True)
dates_test = df['Date'].iloc[train_size + look_back:].reset_index(drop=True)

# Erstellung von TensorDatasets für die Nutzung mit DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# =============================================================================
# 5. DataLoader erstellen
# =============================================================================

# Festlegung der Batch-Größe
batch_size = 128

# Erstellung der DataLoader für Trainings- und Testdaten
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 6. LSTM-Modell definieren
# =============================================================================

class LSTMModel(nn.Module):
    """
    Definition eines einfachen LSTM-Modells zur Vorhersage von S&P 500 Close-Preisen.
    """
    def __init__(self, input_size=4, hidden_size=128, num_layers=1, output_size=1):
        """
        Initialisierung des LSTM-Modells.

        Args:
            input_size (int): Anzahl der Eingabe-Features (hier 4: Close, BB_Middle, BB_Upper, BB_Lower).
            hidden_size (int): Anzahl der LSTM-Einheiten im versteckten Zustand.
            num_layers (int): Anzahl der LSTM-Schichten.
            output_size (int): Anzahl der Ausgabewerte.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM-Schicht
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Voll verbundene Schicht zur Ausgabe
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Vorwärtsdurchlauf des Modells.

        Args:
            x (torch.Tensor): Eingabedaten mit Form [Batch, Time Steps, Features].

        Returns:
            torch.Tensor: Modellvorhersagen.
        """
        # Initialisierung der versteckten und Zellzustände mit Nullen
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Durchlauf durch die LSTM-Schicht
        out, _ = self.lstm(x, (h0, c0))  # out: [Batch, Time Steps, Hidden Size]

        # Auswahl der letzten Zeitschritt-Ausgabe
        out = out[:, -1, :]  # [Batch, Hidden Size]

        # Durchlauf durch die voll verbundene Schicht
        out = self.fc(out)  # [Batch, Output Size]
        return out

# =============================================================================
# 7. Gerät (CPU/GPU) konfigurieren
# =============================================================================

# Auswahl des Geräts: GPU, falls verfügbar, ansonsten CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Gerät: {device}")

# =============================================================================
# 8. Modell, Verlustfunktion und Optimierer initialisieren
# =============================================================================

# Erstellung einer Instanz des LSTM-Modells und Übertragung auf das gewählte Gerät
model = LSTMModel(input_size=4).to(device)  # input_size=4 für die vier Features

# Definition der Verlustfunktion (Mean Squared Error) und des Optimierers (Adam)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 9. Modell trainieren
# =============================================================================

# Anzahl der Trainings-Epochen
num_epochs = 150

for epoch in range(num_epochs):
    model.train()  # Setzt das Modell in den Trainingsmodus
    epoch_loss = 0  # Initialisierung der kumulierten Verlustfunktion

    # Durchlauf durch alle Batches im Trainings-DataLoader
    for X_batch, y_batch in train_loader:
        # Übertragung der Daten auf das gewählte Gerät
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Vorwärtsdurchlauf: Berechnung der Vorhersagen
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)  # Berechnung des Verlusts

        # Rückwärtsdurchlauf und Optimierung
        optimizer.zero_grad()  # Rücksetzen der Gradienten
        loss.backward()        # Berechnung der Gradienten
        optimizer.step()       # Aktualisierung der Modellparameter

        # Akkumulierung des Verlusts
        epoch_loss += loss.item() * X_batch.size(0)

    # Durchschnittlicher Verlust über den gesamten Trainingsdatensatz
    epoch_loss /= len(train_loader.dataset)

    # Validierungsphase (Evaluation des Modells auf Testdaten)
    model.eval()  # Setzt das Modell in den Evaluationsmodus
    with torch.no_grad():  # Deaktiviert die Berechnung der Gradienten
        val_loss = 0
        for X_batch, y_batch in test_loader:
            # Übertragung der Daten auf das gewählte Gerät
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Vorwärtsdurchlauf: Berechnung der Vorhersagen
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)  # Berechnung des Verlusts

            # Akkumulierung des Validierungsverlusts
            val_loss += loss.item() * X_batch.size(0)

        # Durchschnittlicher Verlust über den gesamten Testdatensatz
        val_loss /= len(test_loader.dataset)

    # Ausgabe des Trainings- und Validierungsverlusts alle 10 Epochen und in der ersten Epoche
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")

# =============================================================================
# 10. Vorhersagen treffen
# =============================================================================

model.eval()  # Setzt das Modell in den Evaluationsmodus
with torch.no_grad():  # Deaktiviert die Berechnung der Gradienten
    # Vorhersage auf den Trainingsdaten
    train_preds = model(X_train.to(device)).cpu().numpy()
    # Vorhersage auf den Testdaten
    test_preds = model(X_test.to(device)).cpu().numpy()

# =============================================================================
# 11. Rückskalieren der Vorhersagen
# =============================================================================

# Da wir alle vier Features skaliert haben, müssen wir nur den 'Close' Preis zurückskalieren
# Der 'Close' Preis ist die erste Spalte im scaler

# Extrahieren der skalierten 'Close' Vorhersagen und tatsächlichen Werte
train_preds_scaled = train_preds
test_preds_scaled = test_preds
y_train_scaled = y_train.numpy().reshape(-1, 1)
y_test_scaled = y_test.numpy().reshape(-1, 1)

# Erstellen von Arrays mit vier Spalten für die Inverse Transformation
train_preds_full = np.hstack((train_preds_scaled, np.zeros((train_preds_scaled.shape[0], 3))))
test_preds_full = np.hstack((test_preds_scaled, np.zeros((test_preds_scaled.shape[0], 3))))
y_train_full = np.hstack((y_train_scaled, np.zeros((y_train_scaled.shape[0], 3))))
y_test_full = np.hstack((y_test_scaled, np.zeros((y_test_scaled.shape[0], 3))))

# Rücktransformation der normalisierten Vorhersagen auf die ursprüngliche Skala
train_preds = scaler.inverse_transform(train_preds_full)[:, 0]
y_train_actual = scaler.inverse_transform(y_train_full)[:, 0]
test_preds = scaler.inverse_transform(test_preds_full)[:, 0]
y_test_actual = scaler.inverse_transform(y_test_full)[:, 0]

# =============================================================================
# 12. Prozentuale Abweichung berechnen
# =============================================================================

# Berechnung der prozentualen Abweichung für jeden Testdatenpunkt
percentage_deviation = ((test_preds.flatten() - y_test_actual.flatten()) / y_test_actual.flatten()) * 100

# =============================================================================
# 13. Mean Absolute Percentage Error (MAPE) berechnen
# =============================================================================

# Berechnung des durchschnittlichen absoluten prozentualen Fehlers
mape = np.mean(np.abs(percentage_deviation))
print(f"Mean Absolute Percentage Error (MAPE) der Testdaten: {mape:.2f}%")

# =============================================================================
# 14. Ergebnisse visualisieren
# =============================================================================

plt.figure(figsize=(14, 7))

# Plot der tatsächlichen Testdaten
plt.plot(dates_test, y_test_actual.flatten(), label='Tatsächliche Testdaten', color='blue')

# Plot der vorhergesagten Testdaten
plt.plot(dates_test, test_preds.flatten(), label='Vorhersage Testdaten', color='orange')

# Hinzufügen von Beschriftungen und Titel
plt.xlabel('Datum')
plt.ylabel('Close Preis')
plt.title('LSTM Vorhersage des S&P 500 Close Preises (nur Testdaten)')

# Hinzufügen einer Legende
plt.legend()

# Verbesserung der Datumsanzeige
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Anzeige des Plots
plt.tight_layout()
plt.show()

# =============================================================================
# 15. CSV mit Vorhersagen, tatsächlichen Werten und prozentualer Abweichung erstellen
# =============================================================================

# Erstellung eines DataFrames für die Trainingsvorhersagen
train_df = pd.DataFrame({
    'Date': dates_train,
    'Actual_Close': y_train_actual.flatten(),
    'Predicted_Close': train_preds.flatten()
})

# Erstellung eines DataFrames für die Testvorhersagen inklusive prozentualer Abweichung
test_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test_actual.flatten(),
    'Predicted_Close': test_preds.flatten(),
    'Percentage_Deviation': percentage_deviation  # Hinzufügen der prozentualen Abweichung
})

# Kombination der Trainings- und Testdaten in einem vollständigen DataFrame
full_df = pd.concat([train_df, test_df], ignore_index=True)

# Optional: Sortieren des vollständigen DataFrames nach Datum
full_df = full_df.sort_values('Date').reset_index(drop=True)

# Pfad zur Ausgabe-CSV-Datei
output_path = os.path.join(script_dir, 'lstm_sp500_data/bb/lstm_sp500_results_with_BB_1.csv')

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Speicherung des vollständigen DataFrames als CSV-Datei ohne den Index
full_df.to_csv(output_path, index=False)

# Ausgabe einer Bestätigungsmeldung
print(f"CSV-Datei mit den Vorhersagen, tatsächlichen Werten und prozentualen Abweichungen wurde unter '{output_path}' gespeichert.")
print(f"Mean Absolute Percentage Error (MAPE) der Testdaten: {mape:.2f}%")
