import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ----------------------------
# Geräteeinrichtung
# ----------------------------
# Bestimmt, ob eine GPU verfügbar ist und verwendet diese, andernfalls CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

# ----------------------------
# Dynamischer Pfad zur Datei
# ----------------------------
# Bestimmt das Verzeichnis des aktuellen Skripts und erstellt den Pfad zur Datendatei
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../shared_resources/sp500_data/SP500_Index_Historical_Data.csv")


# ----------------------------
# Datenvorbereitung
# ----------------------------
def load_and_prepare_data(file_path):
    """
    Lädt die CSV-Datei und bereitet die Daten vor.

    Args:
        file_path (str): Pfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Vorbereiteter und sortierter DataFrame.

    Raises:
        FileNotFoundError: Wenn die Datei nicht gefunden wird.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datei nicht gefunden unter: {file_path}")

    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date').reset_index(drop=True)
    return data


def split_train_test(data, train_start, train_end, test_start, test_end):
    """
    Teilt die Daten in Trainings- und Testdatensätze basierend auf den angegebenen Datumsbereichen.

    Args:
        data (pd.DataFrame): Gesamtdatensatz.
        train_start (str): Startdatum für das Training im Format 'YYYY-MM-DD'.
        train_end (str): Enddatum für das Training im Format 'YYYY-MM-DD'.
        test_start (str): Startdatum für den Test im Format 'YYYY-MM-DD'.
        test_end (str): Enddatum für den Test im Format 'YYYY-MM-DD'.

    Returns:
        tuple: Trainings- und Testdaten als DataFrames.

    Raises:
        ValueError: Wenn der Trainings- oder Testdatensatz leer ist.
    """
    train = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].reset_index(drop=True)
    test = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)].reset_index(drop=True)

    if train.empty:
        raise ValueError("Trainingsdaten sind leer. Überprüfen Sie die Datumsbereiche.")
    if test.empty:
        raise ValueError("Testdaten sind leer. Überprüfen Sie die Datumsbereiche.")

    return train, test


def scale_features(train, test, feature_columns):
    """
    Skaliert die angegebenen Merkmale mit Min-Max-Skalierung.

    Args:
        train (pd.DataFrame): Trainingsdaten.
        test (pd.DataFrame): Testdaten.
        feature_columns (list): Liste der zu skalierenden Merkmale.

    Returns:
        tuple: Skalierte Trainings- und Testdaten sowie der verwendete Scaler.
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[feature_columns])
    test_scaled = scaler.transform(test[feature_columns])
    return train_scaled, test_scaled, scaler


def create_single_day_sequences(data):
    """
    Erstellt Eingabe- und Zielsequenzen für das Modell basierend auf Einzelertagen.

    Args:
        data (np.ndarray): Skalierte Daten.

    Returns:
        tuple: Eingaben und Ziele als NumPy-Arrays.
    """
    inputs, labels = [], []
    for i in range(1, len(data)):
        inputs.append(data[i - 1])  # Wert des Vortages als Eingabe
        labels.append(data[i, 3])  # 'Close'-Wert als Zielwert
    return np.array(inputs), np.array(labels)


# ----------------------------
# Datenvorbereitung ausführen
# ----------------------------
# Laden und Vorbereiten der Daten
data = load_and_prepare_data(file_path)

# Aufteilen der Daten in Trainings- und Testdatensätze
# Änderung des Trainingszeitraums auf 1980-01-01 bis 2015-12-31
train_data, test_data = split_train_test(data, '1980-01-01', '2015-12-31', '2016-01-01', '2024-12-31')

# Definieren der zu verwendenden Merkmale
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Skalieren der Merkmale
train_scaled, test_scaled, scaler = scale_features(train_data, test_data, features)

# Erstellen der Eingabe- und Zielsequenzen
X_train, y_train = create_single_day_sequences(train_scaled)
X_test, y_test = create_single_day_sequences(test_scaled)

# Konvertieren der Daten in PyTorch-Tensoren
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# ----------------------------
# Modell definieren
# ----------------------------
class StockPredictor(nn.Module):
    """
    Einfaches neuronales Netzwerk zur Vorhersage des Schlusskurses.
    """

    def __init__(self, input_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


# Festlegen der Eingabe- und Ausgabegrößen
input_size = 5  # Anzahl der Merkmale (Open, High, Low, Close, Volume)
output_size = 1  # Vorhersage des 'Close'-Werts

# Initialisieren des Modells, Verlustfunktion und Optimierer
model = StockPredictor(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training
# ----------------------------
num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # Iterieren über die Trainingsdaten in Batches
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size].to(device)
        y_batch = y_train_tensor[i:i + batch_size].to(device)

        # Vorwärtsdurchlauf
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Rückwärtsdurchlauf und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    # Durchschnittlichen Verlust pro Epoche berechnen
    avg_loss = epoch_loss / len(X_train_tensor)
    print(f"Epoch {epoch + 1}/{num_epochs}, Verlust: {avg_loss:.6f}")

# ----------------------------
# Testen
# ----------------------------
model.eval()
with torch.no_grad():
    # Vorhersagen auf den Testdaten
    predictions = model(X_test_tensor.to(device)).squeeze().cpu().numpy()
    # Berechnung des Testverlusts
    test_loss = criterion(torch.tensor(predictions), y_test_tensor.squeeze()).item()

print(f"Testverlust: {test_loss:.6f}")


# ----------------------------
# Ergebnisse visualisieren
# ----------------------------
def denormalize_data(scaler, data, feature_index=3):
    """
    Denormalisiert die Daten für ein bestimmtes Merkmal.

    Args:
        scaler (MinMaxScaler): Verwendeter Skalierer.
        data (np.ndarray): Normalisierte Daten.
        feature_index (int): Index des zu denormalisierenden Merkmals.

    Returns:
        np.ndarray: Denormalisierte Daten.
    """
    dummy = np.zeros((len(data), scaler.scale_.shape[0]))
    dummy[:, feature_index] = data
    denormalized = scaler.inverse_transform(dummy)[:, feature_index]
    return denormalized


# Denormalisieren der tatsächlichen und vorhergesagten Werte
y_test_denormalized = denormalize_data(scaler, y_test)
predictions_denormalized = denormalize_data(scaler, predictions)

# Überprüfen der Datenlängen für die Visualisierung
if not (len(test_data['Date'][1:]) == len(y_test_denormalized) == len(predictions_denormalized)):
    raise ValueError("Längen von Datum, tatsächlichen Werten und Vorhersagen stimmen nicht überein.")

# Plotten der tatsächlichen vs. vorhergesagten Werte
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'][1:], y_test_denormalized, label='Tatsächliche Werte', alpha=0.7)
plt.plot(test_data['Date'][1:], predictions_denormalized, label='Vorhersagen', alpha=0.7)
plt.xlabel('Datum')
plt.ylabel('SP500 Schlusskurs')
plt.title('Tatsächliche vs. Vorhergesagte Werte (Vortageswert)')
plt.legend()
plt.grid(False)
plt.show()

# ----------------------------
# Daten für CSV erstellen
# ----------------------------
# Vorhersagen auf den Trainingsdaten
train_predictions = model(X_train_tensor.to(device)).squeeze().detach().cpu().numpy()
y_train_denormalized = denormalize_data(scaler, y_train)
train_predictions_denormalized = denormalize_data(scaler, train_predictions)

# Erstellen eines DataFrames mit den Ergebnissen
results_df = pd.DataFrame({
    'Date': train_data['Date'][1:],  # Start bei 1, da vorheriger Tag für Input verwendet wird
    'Actual': y_train_denormalized,
    'Predicted': train_predictions_denormalized,
})

# Prozentuale Abweichung berechnen
results_df['Percentage_Error'] = abs(results_df['Actual'] - results_df['Predicted']) / results_df['Actual'] * 100

# Durchschnittliche prozentuale Abweichung berechnen
mean_percentage_error = results_df['Percentage_Error'].mean()
print(f"Durchschnittliche prozentuale Abweichung (Training): {mean_percentage_error:.2f}%")

# ----------------------------
# CSV-Datei speichern
# ----------------------------
# Pfad zur Speicherung der Trainingsresultate
csv_file_path_train = os.path.join(script_dir, '../Experiment_1/rnn_sp500_train_results_1.csv')

# Speichern des DataFrames als CSV
results_df.to_csv(csv_file_path_train, index=False)
print(f"Trainingsergebnisse wurden gespeichert unter: {csv_file_path_train}")
