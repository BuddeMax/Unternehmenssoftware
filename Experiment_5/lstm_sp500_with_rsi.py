import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Daten laden
def lade_daten(pfad=None):
    """
    Lädt die historischen S&P 500 Daten mit RSI aus einer CSV-Datei.

    Args:
        pfad (str, optional): Pfad zur CSV-Datei. Wenn keiner angegeben wird,
                               wird der Standardpfad im Heimverzeichnis verwendet.

    Returns:
        pd.DataFrame: DataFrame mit den geladenen Daten.

    Raises:
        FileNotFoundError: Wenn die angegebene Datei nicht gefunden wird.
    """
    if pfad is None:
        # Heimverzeichnis des aktuellen Benutzers ermitteln
        home_dir = os.path.expanduser('~')
        # Relativen Pfad zum Datenverzeichnis hinzufügen
        pfad = os.path.join(home_dir, 'Unternehmenssoftware/shared_resources/sp500_data/RSI/SP500_Index_Historical_Data_with_RSI.csv')

    # Überprüfen, ob die Datei existiert
    if not os.path.exists(pfad):
        raise FileNotFoundError(f"Die Datei wurde nicht gefunden: {pfad}")

    # CSV-Datei laden und das Datum als datetime-Objekt parsen
    daten = pd.read_csv(pfad, parse_dates=['Date'])
    return daten


# 2. Daten vorbereiten
def bereite_daten_vor(daten, spalten=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI'], sequenz_länge=60):
    """
    Bereitet die Daten für das LSTM-Modell vor, indem sie skaliert und in Sequenzen unterteilt werden.

    Args:
        daten (pd.DataFrame): DataFrame mit den Rohdaten.
        spalten (list, optional): Liste der zu verwendenden Spalten.
        sequenz_länge (int, optional): Länge der Eingabesequenz für das LSTM.

    Returns:
        Tuple: Tensoren für Training und Test, der Scaler und die Test-Daten mit Datum.
    """
    # Fehlende RSI-Werte überprüfen und ggf. entfernen
    if 'RSI' in spalten:
        daten = daten.dropna(subset=['RSI'])

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

    # Korrespondierende Datumsangaben für den Testbereich
    dates = daten['Date'].values[sequenz_länge:]
    dates_train, dates_test = dates[:training], dates[training:]

    # In Torch-Tensoren umwandeln
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler, dates_test


# 3. Einfaches LSTM-Modell definieren
class EinfachesLSTM(nn.Module):
    """
    Ein einfaches LSTM-Modell für die Vorhersage von Aktienpreisen.
    """

    def __init__(self, input_dim=6, hidden_dim=50, output_dim=1, num_layers=2):
        """
        Initialisiert das LSTM-Modell.

        Args:
            input_dim (int, optional): Anzahl der Eingabefunktionen.
            hidden_dim (int, optional): Anzahl der Neuronen in der versteckten Schicht.
            output_dim (int, optional): Anzahl der Ausgabefunktionen.
            num_layers (int, optional): Anzahl der LSTM-Schichten.
        """
        super(EinfachesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchlauf des Modells.

        Args:
            x (torch.Tensor): Eingabedaten.

        Returns:
            torch.Tensor: Modellvorhersage.
        """
        out, _ = self.lstm(x)  # LSTM-Schicht durchlaufen
        out = out[:, -1, :]  # Letzten Zeitschritt nehmen
        out = self.fc(out)  # Ausgabe durch die vollverbundene Schicht
        return out


# 4. Modell trainieren
def trainiere_modell(modell, daten_loader, verlustfunktion, optimierer, gerät, epochen=100):
    """
    Trainiert das LSTM-Modell.

    Args:
        modell (nn.Module): Das zu trainierende Modell.
        daten_loader (DataLoader): DataLoader für die Trainingsdaten.
        verlustfunktion (nn.Module): Verlustfunktion.
        optimierer (torch.optim.Optimizer): Optimierer.
        gerät (torch.device): Gerät (CPU oder GPU) für das Training.
        epochen (int, optional): Anzahl der Trainingsepochen.

    Returns:
        list: Liste der Verlustwerte pro Epoche.
    """
    verlust_werte = []
    for epoch in range(epochen):
        modell.train()  # Modell in den Trainingsmodus setzen
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

        # Alle 10 Epochen und die erste Epoche den Verlust anzeigen
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoche {epoch + 1}/{epochen}, Verlust: {durchschnitt_verlust:.6f}')

    return verlust_werte


# 5. Modell testen
def teste_modell(modell, daten_loader, scaler, gerät, test_dates, output_csv='lstm_sp500_with_rsi.md.csv'):
    """
    Testet das trainierte Modell und speichert die Ergebnisse in einer CSV-Datei.

    Args:
        modell (nn.Module): Das trainierte Modell.
        daten_loader (DataLoader): DataLoader für die Testdaten.
        scaler (MinMaxScaler): Scaler zur Rücktransformation der Daten.
        gerät (torch.device): Gerät (CPU oder GPU) für die Berechnungen.
        test_dates (np.array): Array mit den Datumsangaben der Testdaten.
        output_csv (str, optional): Name der Ausgabedatei für die Vorhersagen.

    Returns:
        Tuple: Vorhergesagte und tatsächliche Preise (denormalisiert).
    """
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

    # Ergebnisse plotten mit Jahren auf der x-Achse
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, tatsächliche_denorm, label='Tatsächlicher Preis')
    plt.plot(test_dates, vorhersagen_denorm, label='Vorhergesagter Preis')
    plt.xlabel('Datum')
    plt.ylabel('S&P 500 Open Preis')
    plt.title('Tatsächlicher vs. Vorhergesagter S&P 500 Open Preis mit RSI')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # RMSE berechnen (Root Mean Square Error)
    rmse = np.sqrt(np.mean((vorhersagen_denorm - tatsächliche_denorm) ** 2))
    print(f'RMSE auf Testdaten: {rmse:.2f}')

    # Ergebnisse in einer CSV-Datei speichern
    ergebnisse = pd.DataFrame({
        'Datum': test_dates,
        'Tatsächlicher_Open_Preis': tatsächliche_denorm,
        'Vorhergesagter_Open_Preis': vorhersagen_denorm
    })
    ergebnisse.to_csv(output_csv, index=False)
    print(f'Ergebnisse wurden in {output_csv} gespeichert.')

    return vorhersagen_denorm, tatsächliche_denorm


# 6. Hauptfunktion
def haupt():
    """
    Hauptfunktion, die den gesamten Ablauf vom Laden der Daten bis zur Modellbewertung steuert.
    """
    # Schritt 1: Daten laden
    print("Daten werden geladen...")
    daten = lade_daten()
    print("Daten erfolgreich geladen.")

    # Schritt 2: Daten vorbereiten
    print("Daten werden vorbereitet...")
    sequenz_länge = 60  # Anzahl der vorherigen Tage, die zur Vorhersage verwendet werden
    X_train, y_train, X_test, y_test, scaler, dates_test = bereite_daten_vor(daten, sequenz_länge=sequenz_länge)
    print("Daten erfolgreich vorbereitet.")

    # Schritt 3: Modell erstellen
    print("Erstelle das LSTM-Modell...")
    modell = EinfachesLSTM(input_dim=X_train.shape[2])  # input_dim basierend auf den Daten
    print("Modell erfolgreich erstellt.")

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
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Schritt 10: Modell testen
    print("Teste das Modell mit den Testdaten...")
    teste_modell(modell, test_loader, scaler, gerät, dates_test)
    print("Modell erfolgreich getestet.")


# 7. Programm starten
if __name__ == "__main__":
    haupt()
