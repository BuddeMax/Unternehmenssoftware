import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 1. Daten einlesen
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absoluter Pfad des Skripts

# -- Datei 1 (CombinedIndicators1.csv) -------------------------------
data_path_combined = os.path.join(
    script_dir,
    "../shared_resources/sp500_data/CombinedIndicators/CombinedIndicators1.csv"
)

if not os.path.exists(data_path_combined):
    raise FileNotFoundError(f"Datei nicht gefunden unter: {data_path_combined}")
else:
    print(f"Verwende Daten von: {data_path_combined}")

df_combined = pd.read_csv(data_path_combined, parse_dates=['Date'])
df_combined = df_combined.sort_values('Date').reset_index(drop=True)

# -- Datei 2 (SP500_Index_Historical_Data.csv) -----------------------
data_path_sp500 = os.path.join(
    script_dir,
    "../shared_resources/sp500_data/SP500_Index_Historical_Data.csv"
)

if not os.path.exists(data_path_sp500):
    raise FileNotFoundError(f"Datei nicht gefunden unter: {data_path_sp500}")
else:
    print(f"Zusätzliche Daten von: {data_path_sp500}")

df_sp500 = pd.read_csv(data_path_sp500, parse_dates=['Date'])
df_sp500 = df_sp500.sort_values('Date').reset_index(drop=True)

# =============================================================================
# 2. Überprüfen und Anpassen der Spalten + Merge
# =============================================================================
print("Spalten in CombinedIndicators1.csv:", df_combined.columns.tolist())
print("Spalten in SP500_Index_Historical_Data.csv:", df_sp500.columns.tolist())

# Falls 'Close' nicht vorhanden ist, aus 'Close+1' rückverschieben
if 'Close' not in df_combined.columns and 'Close+1' in df_combined.columns:
    df_combined['Close'] = df_combined['Close+1'].shift(1)
    print("Spalte 'Close' wurde erstellt durch Zurückverschieben von 'Close+1'.")

# Jetzt Zusammenführen (Merge) der beiden DataFrames über 'Date'.
# Wir holen uns zusätzlich aus df_sp500 die Spalten 'Open', 'High', 'Low'.
df_sp500_reduced = df_sp500[['Date', 'Open', 'High', 'Low']]  # Nur Spalten, die wir noch nicht haben
df_merged = pd.merge(df_combined, df_sp500_reduced, on='Date', how='inner')

# Fehlende Zeilen entfernen, wenn beim Verschieben Lücken entstanden
df_merged = df_merged.dropna().reset_index(drop=True)

# =============================================================================
# 3. Feature-Auswahl und Zielvariable festlegen
# =============================================================================
# Du möchtest 'Close+1' NICHT nutzen, daher entfernen wir es komplett (falls vorhanden)
if 'Close+1' in df_merged.columns:
    df_merged.drop(columns=['Close+1'], inplace=True)
    print("Spalte 'Close+1' wurde aus dem DataFrame entfernt.")

# Beispiel-Feature-Liste (erweitert um 'Open', 'High', 'Low'):
features = [
    'Open',
    'High',
    'Low',
    'Close',      # Ziel wird später extrahiert, wird aber zur Normalisierung benötigt
    'Volume',
    'RSI',
    'MACD',
    'Signal_Line',
    'BB_UpperBreak',
    'BB_LowerBreak',
    'BB_Squeeze'
]
# Du kannst hier natürlich noch mehr Indikatoren einbinden (z.B. RSI_Overbought, RSI_Slope etc.),
# falls du sie als Feature haben möchtest.

# Check, ob alle diese Features in df_merged sind
missing_features = [feat for feat in features if feat not in df_merged.columns]
if missing_features:
    raise KeyError(f"Die folgenden Features fehlen in den Daten: {missing_features}")
else:
    print("Alle ausgewählten Features sind in den Daten vorhanden.")

# =============================================================================
# 4. Daten normalisieren basierend auf Trainingsdaten
# =============================================================================
df = df_merged.copy()  # Kürzere Variable zum Weiterverarbeiten

look_back = 90  # Zeitfenster
# Du könntest das später auf 60, 90 oder mehr erhöhen, wenn du magst.

train_size = int(len(df) * 0.8)
train_data = df[features][:train_size]
test_data = df[features][train_size - look_back:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# =============================================================================
# 5. Datensätze erstellen
# =============================================================================
# Zielvariable wird hier der 'Close'-Wert sein (späteres Stock-Price-Prediction).
# Wir nehmen also die letzte Spalte (index 0) für X und index 0 für y in create_dataset.
# !!! Achtung: Wir müssen den Index anpassen, da 'Close' nicht unbedingt an erster Stelle steht.

# Erst herausfinden, an welcher Stelle in 'features' sich 'Close' befindet:
close_index = features.index('Close')  # z.B. 3

def create_dataset(dataset, look_back=1, target_index=0):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        # Alle Spalten als Input
        data_slice = dataset[i:(i + look_back), :]
        # Ziel ist hier die Spalte 'Close' zum Zeitpunkt i+look_back
        X.append(data_slice[:, :])  # Alle Features
        y.append(dataset[i + look_back, target_index])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(scaled_train, look_back=look_back, target_index=close_index)
X_test, y_test = create_dataset(scaled_test, look_back=look_back, target_index=close_index)

# Konvertierung zu PyTorch-Tensoren
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =============================================================================
# 6. Trainings- und Testdaten aufteilen (Dates)
# =============================================================================
dates_train = df['Date'].iloc[look_back:train_size].reset_index(drop=True)
dates_test = df['Date'].iloc[train_size:].reset_index(drop=True)

# Überprüfen der Längen
assert len(dates_test) == len(y_test), "Längen von dates_test und y_test stimmen nicht überein."

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# =============================================================================
# 7. DataLoader erstellen
# =============================================================================
batch_size = 128  # Kannst du anpassen
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 8. LSTM-Modell definieren (ohne Dropout)
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Mehrschichtiges LSTM ohne Dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialisieren der versteckten Zustände
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Durchlaufen der LSTM-Schichten
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Letzter Zeitschritt
        out = self.fc(out)
        return out

# =============================================================================
# 9. Gerät (OpenCL/CPU) konfigurieren
# =============================================================================
# ---------------------------------------------------------------------
# WICHTIGER HINWEIS:
# Standard-PyTorch unterstützt nur CPU, CUDA (Nvidia) oder MPS (Apple).
# Offiziell gibt es KEIN 'opencl'-Device. Wenn du AMD GPUs mit ROCm benutzt,
# müsstest du eine spezielle PyTorch-Version installieren.
# Hier zeigen wir dir nur, wie du eine Ausgabe simulieren kannst,
# um zu sehen, ob du "OpenCL" oder CPU verwendest.
# ---------------------------------------------------------------------

use_opencl = True  # Falls du OpenCL verwenden möchtest (simuliert)
if use_opencl:
    # In einer echten Umgebung mit ROCm/OpenCL wäre hier die Abfrage anders.
    device = torch.device("cpu")  # Hier nur Dummy, weil torch.device("opencl") so nicht funktioniert.
    print("Hinweis: OpenCL (AMD) wird simuliert. Tatsächlich läuft es ggf. noch über CPU.")
    print("Um echte GPU-Beschleunigung zu nutzen, ist eine spezielle PyTorch-Version (z.B. ROCm) nötig.")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Aktuell verwendetes Gerät (laut Einstellung): {device}")

# =============================================================================
# 10. Modell, Verlustfunktion und Optimierer initialisieren
# =============================================================================
input_size = X_train.shape[2]  # Anzahl der Features
hidden_size = 128              # Erhöhte Anzahl der versteckten Einheiten
num_layers = 3                 # Mehr LSTM-Schichten
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Angepasste Lernrate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       patience=10, factor=0.5)

# =============================================================================
# 11. Modell trainieren (ohne Early Stopping)
# =============================================================================
# Epochen erst einmal niedriger setzen (z.B. 20), kannst du selbst erhöhen
num_epochs = 200

clip_value = 5.0  # Gradient Clipping

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

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
    train_losses.append(epoch_loss)

    # Validierung
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Protokollierung
    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Training Loss: {epoch_loss:.6f} | "
              f"Validation Loss: {val_loss:.6f} | "
              f"Lernrate: {current_lr:.6f}")

# =============================================================================
# 12. Vorhersagen treffen
# =============================================================================
model.eval()
with torch.no_grad():
    train_preds = model(X_train.to(device)).cpu().numpy()
    test_preds = model(X_test.to(device)).cpu().numpy()

# =============================================================================
# 13. Rückskalieren der Vorhersagen
# =============================================================================
# Wir müssen wissen, an welcher Stelle 'Close' steht (close_index).
# Inverse transform benötigt ein Array gleicher Dimension wie beim Scaler.

def inverse_transform_close(preds, scaler, feature_index):
    # Leeres Array in derselben Breite wie die Features
    temp = np.zeros((preds.shape[0], scaler.scale_.shape[0]))
    # Unsere Vorhersagen nur in der 'Close'-Spalte (feature_index)
    temp[:, feature_index] = preds[:, 0]
    inversed = scaler.inverse_transform(temp)
    return inversed[:, feature_index]

train_preds_unscaled = inverse_transform_close(train_preds, scaler, close_index)
y_train_unscaled = inverse_transform_close(
    y_train.cpu().numpy().reshape(-1, 1),
    scaler,
    close_index
)
test_preds_unscaled = inverse_transform_close(test_preds, scaler, close_index)
y_test_unscaled = inverse_transform_close(
    y_test.cpu().numpy().reshape(-1, 1),
    scaler,
    close_index
)

# =============================================================================
# 14. Prozentuale Abweichung berechnen
# =============================================================================
percentage_deviation_train = (
    (train_preds_unscaled - y_train_unscaled) / y_train_unscaled
) * 100
percentage_deviation_test = (
    (test_preds_unscaled - y_test_unscaled) / y_test_unscaled
) * 100

# =============================================================================
# 15. MAPE berechnen
# =============================================================================
mape_train = np.mean(np.abs(percentage_deviation_train))
mape_test = np.mean(np.abs(percentage_deviation_test))
print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test) : {mape_test:.2f}%")

# =============================================================================
# 16. Ergebnisse visualisieren (Testdaten)
# =============================================================================
plt.figure(figsize=(14, 7))
plt.plot(dates_test, y_test_unscaled, label='Tatsächliche Testdaten', color='blue')
plt.plot(dates_test, test_preds_unscaled, label='Vorhersage Testdaten', color='orange')
plt.xlabel('Datum')
plt.ylabel('Close Preis')
plt.title('LSTM Vorhersage des S&P 500 Close Preises (Testdaten)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 17. Trainings- und Validierungsverlust visualisieren
# =============================================================================
plt.figure(figsize=(14, 7))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training und Validierungsverlust über die Epochen')
plt.legend()
plt.show()

# =============================================================================
# 18. CSV mit Vorhersagen, tatsächlichen Werten und prozentualer Abweichung
# =============================================================================
output_dir = os.path.join(script_dir, 'lstm_sp500_data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'lstm_sp500_results_no_dropout_no_early_stopping.csv')

# Erstellung DataFrame für Trainingsdaten
train_df = pd.DataFrame({
    'Date': dates_train,
    'Actual_Close': y_train_unscaled,
    'Predicted_Close': train_preds_unscaled,
    'Percentage_Deviation': percentage_deviation_train
})

# Erstellung DataFrame für Testdaten
test_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test_unscaled,
    'Predicted_Close': test_preds_unscaled,
    'Percentage_Deviation': percentage_deviation_test
})

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values('Date').reset_index(drop=True)
full_df.to_csv(output_path, index=False)

print(f"CSV-Datei erstellt unter: {output_path}")
print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test) : {mape_test:.2f}%")
