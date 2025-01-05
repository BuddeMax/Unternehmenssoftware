import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Daten laden und zusammenführen
def lade_und_merge_daten(pfad_bb, pfad_macd, pfad_rsi):
    # Lade Hauptdaten
    daten_main = pd.read_csv(pfad_bb)  # Angenommen, BB-Daten enthalten die Hauptdaten
    daten_macd = pd.read_csv(pfad_macd)
    daten_rsi = pd.read_csv(pfad_rsi)

    # Konvertiere das Datum in ein einheitliches Format, falls nötig
    daten_main['Date'] = pd.to_datetime(daten_main['Date'])
    daten_macd['Date'] = pd.to_datetime(daten_macd['Date'])
    daten_rsi['Date'] = pd.to_datetime(daten_rsi['Date'])

    # Merge die Daten basierend auf dem Datum
    daten = daten_main.merge(daten_macd, on='Date', how='left')
    daten = daten.merge(daten_rsi, on='Date', how='left')

    # Optional: Sortiere die Daten nach Datum
    daten = daten.sort_values('Date').reset_index(drop=True)

    # Optional: Überprüfe auf fehlende Werte und handle sie
    daten.fillna(method='ffill', inplace=True)  # Vorwärtsextrapolation
    daten.dropna(inplace=True)  # Entferne verbleibende fehlende Werte

    return daten


# 2. Daten vorbereiten
def bereite_daten_vor(daten, spalten, sequenz_länge=60):
    # Daten skalieren zwischen 0 und 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    skaliert = scaler.fit_transform(daten[spalten].values)

    X, y = [], []
    for i in range(sequenz_länge, len(skaliert)):
        X.append(skaliert[i - sequenz_länge:i])  # Eingabesequenz
        y.append(skaliert[i, 0])  # 0=Open Zielwert (Open-Preis)

    X, y = np.array(X), np.array(y)

    # Daten in Training und Test aufteilen (80% Training, 20% Test)
    training = int(0.8 * len(X))
    X_train, X_test = X[:training], X[training:]
    y_train, y_test = y[:training], y[training:]

    # In Torch-Tensoren umwandeln
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler


# 3. LSTM-Modell definieren
class EinfachesLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=50, output_dim=1, num_layers=2):
        super(EinfachesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM-Schicht durchlaufen
        out = out[:, -1, :]  # Letzten Zeitschritt nehmen
        out = self.fc(out)  # Ausgabe durch die vollverbundene Schicht
        return out


# 4. Modell trainieren
def trainiere_modell(modell, daten_loader, verlustfunktion, optimierer, gerät, epochen=100):
    verlust_werte = []
    for epoch in range(epochen):
        modell.train()
        gesamt_verlust = 0
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)

            optimierer.zero_grad()  # Gradienten zurücksetzen
            vorhersage = modell(inputs)  # Vorhersage berechnen
            verlust = verlustfunktion(vorhersage, labels)  # Verlust berechnen
            verlust.backward()  # Rückwärtsdurchlauf (Gradienten berechnen)
            optimierer.step()  # Gewichte aktualisieren

            gesamt_verlust += verlust.item() * inputs.size(0)

        durchschnitt_verlust = gesamt_verlust / len(daten_loader.dataset)
        verlust_werte.append(durchschnitt_verlust)

        # Alle 10 Epochen den Verlust anzeigen
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoche {epoch + 1}/{epochen}, Verlust: {durchschnitt_verlust:.6f}')

    return verlust_werte


# 5. Modell testen
def teste_modell(modell, daten_loader, scaler, gerät):
    modell.eval()  # Modell in den Evaluierungsmodus setzen
    vorhersagen, tatsächliche = [], []

    with torch.no_grad():  # Kein Gradientenberechnung
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)
            outputs = modell(inputs)  # Vorhersage berechnen
            vorhersagen.append(outputs.cpu().numpy())
            tatsächliche.append(labels.cpu().numpy())

    vorhersagen = np.concatenate(vorhersagen).flatten()
    tatsächliche = np.concatenate(tatsächliche).flatten()

    # Daten denormalisieren (zurückskalieren)
    vorhersagen_denorm = scaler.inverse_transform(
        np.hstack((
            vorhersagen.reshape(-1, 1),  # Vorhersagen in Spalte 0 (Open)
            np.zeros((vorhersagen.shape[0], scaler.n_features_in_ - 1))  # Nullen für die anderen Spalten
        ))
    )[:, 0]  # Erste Spalte auswählen, da dort Open steht

    tatsächliche_denorm = scaler.inverse_transform(
        np.hstack((
            tatsächliche.reshape(-1, 1),
            np.zeros((tatsächliche.shape[0], scaler.n_features_in_ - 1))
        ))
    )[:, 0]

    # Ergebnisse plotten
    plt.figure(figsize=(12, 6))
    plt.plot(tatsächliche_denorm, label='Tatsächlicher Preis')
    plt.plot(vorhersagen_denorm, label='Vorhergesagter Preis')
    plt.xlabel('Zeit')
    plt.ylabel('S&P 500 Open Preis')
    plt.title('Tatsächlicher vs. Vorhergesagter S&P 500 Open Preis')
    plt.legend()
    plt.show()

    # RMSE berechnen (Root Mean Square Error)
    rmse = np.sqrt(np.mean((vorhersagen_denorm - tatsächliche_denorm) ** 2))
    print(f'RMSE auf Testdaten: {rmse:.2f}')

    return vorhersagen_denorm, tatsächliche_denorm


# 6. Hauptfunktion
def haupt():
    # Pfade zu den Daten
    pfad_bb = '/Users/maxbudde/Unternehmenssoftware/sp500_data/BollingerBands/SP500_Index_Historical_Data_with_BB.csv'
    pfad_macd = '/Users/maxbudde/Unternehmenssoftware/sp500_data/MACD/SP500_Index_Historical_Data_with_MACD.csv'
    pfad_rsi = '/Users/maxbudde/Unternehmenssoftware/sp500_data/RSI/SP500_Index_Historical_Data_with_RSI.csv'

    # Schritt 1: Daten laden und zusammenführen
    print("Daten werden geladen und zusammengeführt...")
    daten = lade_und_merge_daten(pfad_bb, pfad_macd, pfad_rsi)
    print(f"Daten erfolgreich geladen. Gesamtanzahl der Datensätze: {len(daten)}")

    # Schritt 2: Daten vorbereiten
    print("Daten werden vorbereitet...")
    # Definiere die zu verwendenden Spalten: Open, High, Low, Close, Volume, BB_Middle, BB_Upper, BB_Lower, MACD, Signal_Line, RSI
    spalten = ['Open', 'High', 'Low', 'Close', 'Volume',
               'BB_Middle', 'BB_Upper', 'BB_Lower',
               'MACD', 'Signal_Line', 'RSI']

    sequenz_länge = 60
    X_train, y_train, X_test, y_test, scaler = bereite_daten_vor(daten, spalten=spalten, sequenz_länge=sequenz_länge)
    print(f"Daten erfolgreich vorbereitet. Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

    # Schritt 3: Modell erstellen
    print("Erstelle das LSTM-Modell...")
    input_dim = X_train.shape[2]  # Anzahl der Merkmale
    modell = EinfachesLSTM(input_dim=input_dim)
    print(f"Modell erfolgreich erstellt mit input_dim={input_dim}.")

    # Schritt 4: Hyperparameter festlegen
    epochen = 100
    batch_größe = 64
    lernrate = 0.001
    print(f"Hyperparameter festgelegt: Epochen={epochen}, Batch-Größe={batch_größe}, Lernrate={lernrate}")

    # Schritt 5: DataLoader erstellen
    print("Erstelle DataLoader für Trainings- und Testdaten...")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_größe, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_größe, shuffle=False)
    print("DataLoader erfolgreich erstellt.")

    # Schritt 6: Gerät auswählen
    gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {gerät}")
    modell.to(gerät)

    # Schritt 7: Verlustfunktion und Optimierer definieren
    verlustfunktion = nn.MSELoss()
    optimierer = torch.optim.Adam(modell.parameters(), lr=lernrate)
    print("Verlustfunktion und Optimierer definiert.")

    # Schritt 8: Modell trainieren
    print("Starte das Training des Modells...")
    verlust_werte = trainiere_modell(modell, train_loader, verlustfunktion, optimierer, gerät, epochen=epochen)
    print("Training abgeschlossen.")

    # Schritt 9: Verlustkurve plotten
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochen + 1), verlust_werte, label='Training Verlust')
    plt.xlabel('Epoche')
    plt.ylabel('MSE Verlust')
    plt.title('Training Verlust über Epochen')
    plt.legend()
    plt.show()

    # Schritt 10: Modell testen
    print("Teste das Modell mit den Testdaten...")
    teste_modell(modell, test_loader, scaler, gerät)
    print("Modell erfolgreich getestet.")


# 7. Programm starten
if __name__ == "__main__":
    haupt()
