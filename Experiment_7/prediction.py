import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib
import datetime

# =============================================================================
# LSTM-Modell definieren (muss identisch mit dem Trainingsmodell sein)
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Letzter Zeitschritt
        out = self.fc(out)
        return out


# =============================================================================
# 1. Modell und Scaler laden
# =============================================================================
def load_model_and_scaler(model_path, scaler_path, device):
    # Modell initialisieren
    model = LSTMModel().to(device)

    # Laden des state_dict mit weights_only=True zur Behebung der FutureWarning
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Falls PyTorch-Version das Argument weights_only noch nicht unterstützt
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modell geladen von: {model_path}")

    # Scaler laden
    scaler = joblib.load(scaler_path)
    print(f"Scaler geladen von: {scaler_path}")

    return model, scaler


# =============================================================================
# 2. Neue Daten laden
# =============================================================================
def load_latest_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Datei nicht gefunden unter: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


# =============================================================================
# 3. Vorhersage durchführen
# =============================================================================
def predict_next_day(model, scaler, df, look_back=1, device='cpu'):
    # Letzte 'look_back' Datenpunkte auswählen
    last_data = df[['Close', 'RSI']].values[-look_back:]

    # Skalieren
    scaled_last_data = scaler.transform(last_data)

    # Reshape für LSTM [Batch, Time Steps, Features]
    input_data = scaled_last_data.reshape(1, look_back, 2)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # Vorhersage
    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy().flatten()[0]

    # Rückskalieren
    # Da der Scaler auf beiden Features angewendet wurde, müssen wir eine Dummy-Spalte hinzufügen
    predicted_close_scaled = prediction_scaled
    predicted_close_full = np.array([[predicted_close_scaled, 0]])  # Dummy RSI
    predicted_close_unscaled = scaler.inverse_transform(predicted_close_full)[:, 0]
    predicted_close = predicted_close_unscaled[0]

    # Letzter tatsächlicher Close
    last_close = df['Close'].iloc[-1]

    # Berechnung der Änderung
    change = predicted_close - last_close
    change_percent = (change / last_close) * 100
    direction = "hoch" if change > 0 else "runter"

    # Datum des nächsten Tages (angenommen, die Daten sind täglich und Montag-Freitag)
    last_date = df['Date'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:  # Samstag=5, Sonntag=6
        next_date += pd.Timedelta(days=1)

    # Ergebnisse
    result = {
        'Letzter_Close': last_close,
        'Vorhergesagter_Close': predicted_close,
        'Änderung': change,
        'Änderung_%': change_percent,
        'Richtung': direction,
        'Vorhersage_Datum': next_date.date()
    }

    return result


# =============================================================================
# 4. Hauptfunktion
# =============================================================================
def main():
    # Verzeichnis des Skripts ermitteln
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Gehe ein Verzeichnis nach oben, um shared_resources zu finden
    shared_resources_dir = os.path.join(script_dir, "..", "shared_resources")
    shared_resources_dir = os.path.abspath(shared_resources_dir)

    # Dynamische Pfade basierend auf dem shared_resources-Verzeichnis
    model_path = os.path.join(script_dir, "lstm_sp500_model.pth")
    scaler_path = os.path.join(script_dir, "scaler.pkl")
    data_path = os.path.join(shared_resources_dir, "sp500_data", "RSI", "SP500_Index_Historical_Data_with_RSI.csv")

    # Überprüfen, ob alle Dateien existieren
    for path in [model_path, scaler_path, data_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Erforderliche Datei nicht gefunden: {path}")

    # Gerät konfigurieren
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {device}")

    # Modell und Scaler laden
    model, scaler = load_model_and_scaler(model_path, scaler_path, device)

    # Daten laden
    df = load_latest_data(data_path)

    # Vorhersage durchführen
    result = predict_next_day(model, scaler, df, look_back=1, device=device)

    # Ergebnisse anzeigen
    print("\n=== Vorhersage für den nächsten Tag ===")
    print(f"Datum des letzten Datensatzes: {df['Date'].iloc[-1].date()}")
    print(f"Vorhergesagter Schlusskurs für {result['Vorhersage_Datum']}: {result['Vorhergesagter_Close']:.2f}")
    print(f"Letzter tatsächlicher Schlusskurs: {result['Letzter_Close']:.2f}")
    print(f"Änderung: {result['Änderung']:.2f} ({result['Änderung_%']:.2f}%)")
    print(f"Richtung: {result['Richtung']}")


if __name__ == "__main__":
    main()
