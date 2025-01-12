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
script_dir = os.path.dirname(os.path.abspath(__file__))

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
# 2. Vorgesehene End-Spalten definieren und Zusammenführen
# =============================================================================
final_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close_Direction",
    "RSI",
    "MACD",
    "Signal_Line",
    "RSI_Overbought",
    "RSI_Oversold",
    "RSI_Overbought_Streak",
    "RSI_Oversold_Streak",
    "RSI_CrossOver_70",
    "RSI_CrossUnder_30",
    "RSI_Slope",
    "MACD_BullishCrossover",
    "MACD_BearishCrossover",
    "MACD_Bullish_Streak",
    "MACD_Bearish_Streak",
    "MACD_ZeroLine_Crossover",
    "MACD_Slope",
    "BB_UpperBreak",
    "BB_LowerBreak",
    "BB_UpperBreak_Streak",
    "BB_LowerBreak_Streak",
    "BB_Squeeze"
]

print("Spalten in CombinedIndicators1.csv:", df_combined.columns.tolist())
print("Spalten in SP500_Index_Historical_Data.csv:", df_sp500.columns.tolist())

# Falls "Close+1" vorhanden ist, direkt entfernen (du willst es NICHT nutzen)
if "Close+1" in df_combined.columns:
    df_combined.drop(columns=["Close+1"], inplace=True)
    print("'Close+1' wurde entfernt und wird nicht genutzt.")

# Merge (inner) beider DataFrames über "Date".
# Dabei enthalten:
#  - Aus df_sp500: Date, Open, High, Low, Close, Volume
#  - Aus df_combined: sämtliche Indikatoren (Close_Direction, RSI, MACD, ...)
# Nach dem Merge behalten wir **nur** die final_columns (z.B. um Duplikate oder überflüssige Spalten loszuwerden).
df_merged_all = pd.merge(df_combined, df_sp500, on="Date", how="inner", suffixes=("", "_SP500"))

# Manche Spalten können doppelt vorhanden sein (z.B. 'Close', 'Volume'),
# in df_combined könnte es bereits 'Open' etc. geben. Wir erzwingen jetzt,
# dass die Spalten so heißen wie in final_columns.
# Falls dein df_combined bereits Spalten 'Open', 'Volume' usw. enthält und die
# Daten aus df_sp500 bevorzugt werden sollen, behalte die aus df_sp500 (hier als
# ..._SP500). Danach passen wir die Namen an.
# Beispiel: Wenn df_combined 'Volume' enthält, im Merge kann es 'Volume_SP500' geben.

# Um es möglichst eindeutig zu machen, so wie du es gefordert hast:
# Wir wollen am Ende GENAU die 28 final_columns.
# Darum mappen wir die Spalten so:
column_mapping = {
    "Open_SP500": "Open",
    "High_SP500": "High",
    "Low_SP500": "Low",
    "Close_SP500": "Close",
    "Volume_SP500": "Volume"
}
# Falls also _SP500-Versionen existieren, benennen wir sie entsprechend um:
for old_col, new_col in column_mapping.items():
    if old_col in df_merged_all.columns:
        df_merged_all.drop(columns=[new_col], errors='ignore', inplace=True)  # Alte evtl. vorhandene "Open"/"Volume" entfernen
        df_merged_all.rename(columns={old_col: new_col}, inplace=True)

# Nun wählen wir exakt die final_columns aus (falls dort Lücken sind, raise KeyError).
# Achtung: Hier stellen wir sicher, dass wir nur diese Spalten behalten.
try:
    df_merged = df_merged_all[final_columns]
except KeyError as e:
    raise KeyError(f"Mindestens eine der geforderten Spalten ist nicht vorhanden: {e}")

# Zeilen mit NaN entfernen
df_merged = df_merged.dropna().reset_index(drop=True)

# =============================================================================
# 3. Feature-Auswahl und Zielvariable
# =============================================================================
# Die final_columns enthalten bereits alles, was wir nutzen wollen.
# Wir legen hier fest, welche Features wir ins Modell stecken (ohne 'Date').
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close_Direction",
    "RSI",
    "MACD",
    "Signal_Line",
    "RSI_Overbought",
    "RSI_Oversold",
    "RSI_Overbought_Streak",
    "RSI_Oversold_Streak",
    "RSI_CrossOver_70",
    "RSI_CrossUnder_30",
    "RSI_Slope",
    "MACD_BullishCrossover",
    "MACD_BearishCrossover",
    "MACD_Bullish_Streak",
    "MACD_Bearish_Streak",
    "MACD_ZeroLine_Crossover",
    "MACD_Slope",
    "BB_UpperBreak",
    "BB_LowerBreak",
    "BB_UpperBreak_Streak",
    "BB_LowerBreak_Streak",
    "BB_Squeeze"
]
# Hinweis: "Date" ist keine numerische Spalte und wird nicht skaliert.

missing_features = [feat for feat in features if feat not in df_merged.columns]
if missing_features:
    raise KeyError(f"Die folgenden Features fehlen in df_merged: {missing_features}")

df = df_merged.copy()

# =============================================================================
# 4. Daten normalisieren (Train/Test Split)
# =============================================================================
look_back = 90
train_size = int(len(df) * 0.8)

train_data = df[features].iloc[:train_size]
test_data = df[features].iloc[train_size - look_back:]  # Zurück, um Lookback sicherzustellen

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# =============================================================================
# 5. Datensätze erstellen (Time-Series)
# =============================================================================
close_index = features.index("Close")  # Wo 'Close' steht

def create_dataset(dataset, look_back=1, target_index=0):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        data_slice = dataset[i : (i + look_back), :]
        X.append(data_slice)  # Alle Spalten
        y.append(dataset[i + look_back, target_index])  # Ziel = 'Close' beim Zeitschritt i + look_back
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(scaled_train, look_back=look_back, target_index=close_index)
X_test, y_test = create_dataset(scaled_test, look_back=look_back, target_index=close_index)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# =============================================================================
# 6. Datums-Arrays für spätere Darstellung
# =============================================================================
dates_train = df["Date"].iloc[look_back : train_size].reset_index(drop=True)
dates_test = df["Date"].iloc[train_size :].reset_index(drop=True)

assert len(dates_test) == len(y_test), "Längen von dates_test und y_test stimmen nicht überein."

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# =============================================================================
# 7. DataLoader
# =============================================================================
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 8. LSTM-Modell (ohne Dropout)
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Letzter Zeitschritt
        out = self.fc(out)
        return out

# =============================================================================
# 9. Device-Auswahl (CPU/OpenCL-Simulation/CUDA)
# =============================================================================
use_opencl = True  # Nur Demo. Für echtes AMD/OpenCL: ROCm-Version von PyTorch nötig.
if use_opencl:
    device = torch.device("cpu")  # Simuliert "OpenCL"
    print("Hinweis: OpenCL wird simuliert und läuft in Wirklichkeit über CPU.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Aktuell verwendetes Gerät: {device}")

# =============================================================================
# 10. Modell, Loss, Optimizer, Scheduler
# =============================================================================
input_size  = X_train.shape[2]
hidden_size = 128
num_layers  = 3
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=10, factor=0.5
)

# =============================================================================
# 11. Training (ohne Early Stopping)
# =============================================================================
num_epochs = 200
clip_value = 5.0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()

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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {epoch_loss:.6f} "
              f"| Val Loss: {val_loss:.6f} "
              f"| LR: {lr_current:.6f}")

# =============================================================================
# 12. Vorhersagen
# =============================================================================
model.eval()
with torch.no_grad():
    train_preds = model(X_train.to(device)).cpu().numpy()
    test_preds  = model(X_test.to(device)).cpu().numpy()

# =============================================================================
# 13. Rückskalieren der Vorhersage (Close)
# =============================================================================
def inverse_transform_close(preds, scaler, feature_index):
    temp = np.zeros((preds.shape[0], scaler.scale_.shape[0]))
    temp[:, feature_index] = preds[:, 0]
    inversed = scaler.inverse_transform(temp)
    return inversed[:, feature_index]

train_preds_unscaled = inverse_transform_close(train_preds, scaler, close_index)
y_train_unscaled = inverse_transform_close(
    y_train.cpu().numpy().reshape(-1, 1), scaler, close_index
)
test_preds_unscaled = inverse_transform_close(test_preds, scaler, close_index)
y_test_unscaled = inverse_transform_close(
    y_test.cpu().numpy().reshape(-1, 1), scaler, close_index
)

# =============================================================================
# 14. Prozentuale Abweichung + MAPE
# =============================================================================
dev_train = (train_preds_unscaled - y_train_unscaled) / y_train_unscaled * 100
dev_test  = (test_preds_unscaled  - y_test_unscaled)  / y_test_unscaled  * 100

mape_train = np.mean(np.abs(dev_train))
mape_test  = np.mean(np.abs(dev_test))

print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test) : {mape_test:.2f}%")

# =============================================================================
# 15. Plot: Testdaten
# =============================================================================
plt.figure(figsize=(14, 7))
plt.plot(dates_test, y_test_unscaled, label='Tatsächliche Testdaten (Close)', color='blue')
plt.plot(dates_test, test_preds_unscaled, label='Vorhersage (Close)', color='orange')
plt.xlabel('Datum')
plt.ylabel('Preis')
plt.title('LSTM Vorhersage S&P 500 - Testdaten')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 16. Plot: Trainings- und Validierungsverlust
# =============================================================================
plt.figure(figsize=(14, 7))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training und Validierungsverlust')
plt.legend()
plt.show()

# =============================================================================
# 17. CSV-Ausgabe (Train + Test)
# =============================================================================
output_dir = os.path.join(script_dir, 'lstm_sp500_data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'lstm_sp500_results_no_dropout_no_early_stopping.csv')

train_df = pd.DataFrame({
    'Date': dates_train,
    'Actual_Close': y_train_unscaled,
    'Predicted_Close': train_preds_unscaled,
    'Percentage_Deviation': dev_train
})
test_df = pd.DataFrame({
    'Date': dates_test,
    'Actual_Close': y_test_unscaled,
    'Predicted_Close': test_preds_unscaled,
    'Percentage_Deviation': dev_test
})

full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('Date')
full_df.to_csv(output_path, index=False)

print(f"CSV-Datei erstellt unter: {output_path}")
print(f"MAPE (Train): {mape_train:.2f}%")
print(f"MAPE (Test) : {mape_test:.2f}%")
