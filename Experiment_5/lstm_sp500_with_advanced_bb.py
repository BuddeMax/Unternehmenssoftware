import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Für die Datumsformatierung
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Daten laden
def lade_daten(bollinger_pfad=None, normal_pfad=None):
    """
    Lädt Bollinger Bands Daten und normale S&P 500 Daten von den angegebenen Pfaden.
    Wenn keine Pfade angegeben sind, werden Standardpfade im Benutzer-Home-Verzeichnis verwendet.

    Args:
        bollinger_pfad (str, optional): Pfad zur Bollinger Bands CSV-Datei.
        normal_pfad (str, optional): Pfad zur normalen Daten CSV-Datei.

    Returns:
        pd.DataFrame: Zusammengeführter DataFrame mit den geladenen Daten.

    Raises:
        FileNotFoundError: Wenn eine der angegebenen Dateien nicht gefunden wird.
    """
    # Standardpfade festlegen, falls keine Pfade angegeben sind
    if bollinger_pfad is None:
        bollinger_pfad = os.path.join(
            os.path.expanduser('~'),
            'Unternehmenssoftware/shared_resources/sp500_data/BollingerBands/SP500_Index_Historical_Data_with_BB.csv'
        )
    if normal_pfad is None:
        normal_pfad = os.path.join(
            os.path.expanduser('~'),
            'Unternehmenssoftware/shared_resources/sp500_data/BollingerBands/Final_Bollinger_Stats.csv'
        )

    # Überprüfen, ob die Dateien existieren
    if not os.path.exists(bollinger_pfad):
        raise FileNotFoundError(f"Die Bollinger Bands Datei wurde nicht gefunden: {bollinger_pfad}")
    if not os.path.exists(normal_pfad):
        raise FileNotFoundError(f"Die normale Daten Datei wurde nicht gefunden: {normal_pfad}")

    # Daten aus den CSV-Dateien laden und das 'Date'-Feld als Datum interpretieren
    bollinger_daten = pd.read_csv(bollinger_pfad, parse_dates=['Date'])
    normal_daten = pd.read_csv(normal_pfad, parse_dates=['Date'])

    # Zusammenführen der beiden Datensätze anhand des 'Date'-Feldes
    # 'left' sorgt dafür, dass alle Daten aus 'normal_daten' behalten werden
    daten = pd.merge(normal_daten, bollinger_daten, on='Date', how='left', suffixes=('_normal', '_bb'))

    # Sortieren der Daten nach Datum in aufsteigender Reihenfolge
    daten.sort_values('Date', inplace=True)
    daten.reset_index(drop=True, inplace=True)

    # Entfernen von Zeilen mit fehlenden Werten, um die Datenqualität zu gewährleisten
    daten.dropna(inplace=True)

    # Optional: Anzeige der Spalten nach dem Zusammenführen, um die Datenstruktur zu überprüfen
    print("Spalten nach dem Merge:", daten.columns.tolist())

    # Entscheiden, welche 'Volume'-Spalte verwendet werden soll
    # Hier behalten wir 'Volume_normal' und entfernen 'Volume_bb', falls vorhanden
    if 'Volume_bb' in daten.columns:
        daten.drop('Volume_bb', axis=1, inplace=True)

    return daten

# 2. Daten vorbereiten
def bereite_daten_vor(daten, spalten=None, sequenz_länge=60):
    """
    Bereitet die Daten für das LSTM-Modell vor, einschließlich Auswahl der relevanten Spalten,
    Skalierung der Daten und Erstellung von Eingabesequenzen.

    Args:
        daten (pd.DataFrame): Der zusammengeführte DataFrame mit den Daten.
        spalten (list, optional): Liste der Spalten, die verwendet werden sollen.
                                  Standard ist ['Open', 'High', 'Low', 'Close', 'Volume_normal',
                                             'BB_Middle', 'BB_Upper', 'BB_Lower'].
        sequenz_länge (int, optional): Die Länge der Eingabesequenzen. Standard ist 60.

    Returns:
        tuple: Tensoren für X_train, y_train, X_test, y_test, der Scaler und die Test-Daten-Dates.

    Raises:
        KeyError: Wenn eine der gewünschten Spalten im DataFrame fehlt.
    """
    if spalten is None:
        # Standardspalten auswählen, die für das Modell verwendet werden
        spalten = ['Open', 'High', 'Low', 'Close', 'Volume_normal',
                   'BB_Middle', 'BB_Upper', 'BB_Lower']

    # Überprüfen, ob alle gewünschten Spalten im DataFrame vorhanden sind
    fehlende_spalten = [spalte for spalte in spalten if spalte not in daten.columns]
    if fehlende_spalten:
        raise KeyError(f"Die folgenden Spalten fehlen im Datensatz: {fehlende_spalten}")

    # Skalierung der Daten zwischen 0 und 1 für bessere Modellleistung
    scaler = MinMaxScaler(feature_range=(0, 1))
    skaliert = scaler.fit_transform(daten[spalten].values)

    X, y = [], []
    # Erstellen von Sequenzen der festgelegten Länge für die Eingabe
    for i in range(sequenz_länge, len(skaliert)):
        X.append(skaliert[i - sequenz_länge:i])  # Eingabesequenz
        y.append(skaliert[i, 3])  # 3 entspricht dem 'Close'-Preis, der als Zielwert dient

    X, y = np.array(X), np.array(y)

    # Aufteilen der Daten in Trainings- und Testdatensätze (80% Training, 20% Test)
    training = int(0.8 * len(X))
    X_train, X_test = X[:training], X[training:]
    y_train, y_test = y[:training], y[training:]

    # Extrahieren der Datumsangaben für die Testdaten zur späteren Analyse
    test_dates = daten['Date'].iloc[sequenz_länge + training:].reset_index(drop=True)

    # Umwandeln der numpy-Arrays in Torch-Tensoren für die Verwendung mit PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Hinzufügen einer Dimension
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Hinzufügen einer Dimension

    return X_train, y_train, X_test, y_test, scaler, test_dates

# 3. Einfaches LSTM-Modell definieren
class EinfachesLSTM(nn.Module):
    """
    Ein einfaches LSTM-Modell zur Vorhersage des S&P 500 Close-Preises.

    Architektur:
        - LSTM-Schicht(en)
        - Vollverbundene (fully connected) Schicht zur Ausgabe

    Args:
        input_dim (int): Anzahl der Eingabefeatures.
        hidden_dim (int, optional): Anzahl der Neuronen in der LSTM-Schicht. Standard ist 50.
        output_dim (int, optional): Anzahl der Ausgabewerte. Standard ist 1 (der Close-Preis).
        num_layers (int, optional): Anzahl der LSTM-Schichten. Standard ist 2.
    """
    def __init__(self, input_dim=8, hidden_dim=50, output_dim=1, num_layers=2):
        super(EinfachesLSTM, self).__init__()
        # Definieren der LSTM-Schicht
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Definieren der vollverbundenen Schicht zur Ausgabe
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchlauf des Modells.

        Args:
            x (torch.Tensor): Eingabedaten mit Form (Batch, Sequenzlänge, Features).

        Returns:
            torch.Tensor: Vorhergesagte Close-Preise.
        """
        out, _ = self.lstm(x)  # Durchlaufen der LSTM-Schicht
        out = out[:, -1, :]     # Nehmen des letzten Zeitschritts
        out = self.fc(out)      # Durchlaufen der vollverbundenen Schicht
        return out

# 4. Modell trainieren
def trainiere_modell(modell, daten_loader, verlustfunktion, optimierer, gerät, epochen=100):
    """
    Trainiert das LSTM-Modell über eine festgelegte Anzahl von Epochen.

    Args:
        modell (nn.Module): Das zu trainierende Modell.
        daten_loader (DataLoader): DataLoader für die Trainingsdaten.
        verlustfunktion (nn.Module): Die Verlustfunktion (z.B. MSELoss).
        optimierer (torch.optim.Optimizer): Der Optimierer (z.B. Adam).
        gerät (torch.device): Das Gerät, auf dem trainiert wird (CPU oder GPU).
        epochen (int, optional): Anzahl der Trainingsdurchläufe. Standard ist 100.

    Returns:
        list: Liste der durchschnittlichen Verlustwerte pro Epoche.
    """
    verlust_werte = []  # Liste zur Speicherung der Verlustwerte pro Epoche

    for epoch in range(epochen):
        modell.train()  # Setzen des Modells in den Trainingsmodus
        gesamt_verlust = 0  # Variable zur Akkumulation des Verlusts

        # Durchlaufen der Trainingsdaten in Batches
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)  # Verschieben der Daten auf das Gerät

            optimierer.zero_grad()       # Zurücksetzen der Gradienten
            vorhersage = modell(inputs)  # Berechnung der Vorhersagen
            verlust = verlustfunktion(vorhersage, labels)  # Berechnung des Verlusts
            verlust.backward()           # Rückpropagation zur Berechnung der Gradienten
            optimierer.step()            # Aktualisierung der Modellgewichte

            gesamt_verlust += verlust.item() * inputs.size(0)  # Akkumulieren des Verlusts

        # Durchschnittlichen Verlust für die Epoche berechnen
        durchschnitt_verlust = gesamt_verlust / len(daten_loader.dataset)
        verlust_werte.append(durchschnitt_verlust)

        # Alle 10 Epochen (und die erste Epoche) den Verlust anzeigen
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoche {epoch + 1}/{epochen}, Verlust: {durchschnitt_verlust:.6f}')

    return verlust_werte

# 5. Modell testen
def teste_modell(modell, daten_loader, scaler, gerät, test_dates, speicher_pfad='lstm_sp500_with_advanced_bb.csv'):
    """
    Testet das trainierte Modell mit den Testdaten, berechnet Metriken und speichert die Ergebnisse.

    Args:
        modell (nn.Module): Das trainierte Modell.
        daten_loader (DataLoader): DataLoader für die Testdaten.
        scaler (MinMaxScaler): Der Scaler, der zum Skalieren der Daten verwendet wurde.
        gerät (torch.device): Das Gerät, auf dem getestet wird (CPU oder GPU).
        test_dates (pd.Series): Die Datumsangaben der Testdaten.
        speicher_pfad (str, optional): Pfad zur Speicherung der Ergebnisse-CSV. Standard ist 'lstm_sp500_with_advanced_bb.csv'.

    Returns:
        tuple: Denormalisierte Vorhersagen, tatsächliche Werte und deren Differenz.
    """
    modell.eval()  # Setzen des Modells in den Evaluierungsmodus
    vorhersagen, tatsächliche = [], []  # Listen zur Speicherung der Vorhersagen und tatsächlichen Werte

    with torch.no_grad():  # Deaktivieren der Gradientenberechnung
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)  # Verschieben der Daten auf das Gerät
            outputs = modell(inputs)  # Berechnung der Vorhersagen
            vorhersagen.append(outputs.cpu().numpy())  # Sammeln der Vorhersagen
            tatsächliche.append(labels.cpu().numpy())    # Sammeln der tatsächlichen Werte

    # Konvertieren der Listen zu numpy-Arrays und Abflachen
    vorhersagen = np.concatenate(vorhersagen).flatten()
    tatsächliche = np.concatenate(tatsächliche).flatten()

    # Denormalisieren der Daten (Rückskalierung) nur für den 'Close'-Preis
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]  # 'Close' ist die 4. Spalte (Index 3)

    vorhersagen_denorm = vorhersagen / close_scaler.scale_ + close_scaler.min_
    tatsächliche_denorm = tatsächliche / close_scaler.scale_ + close_scaler.min_

    # Berechnung der Differenz zwischen vorhergesagten und tatsächlichen Werten
    differenz = vorhersagen_denorm - tatsächliche_denorm

    # Plotten der Ergebnisse zur visuellen Analyse mit Jahresangaben
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, tatsächliche_denorm, label='Tatsächlicher Close Preis', color='blue')
    plt.plot(test_dates, vorhersagen_denorm, label='Vorhergesagter Close Preis', color='red')
    plt.xlabel('Datum')
    plt.ylabel('S&P 500 Close Preis')
    plt.title('Tatsächlicher vs. Vorhergesagter S&P 500 Close Preis')
    plt.legend()

    # Formatieren der x-Achse, um die Jahre anzuzeigen
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Setzt Locator für jedes Jahr
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formatiert das Datum als Jahr

    plt.gcf().autofmt_xdate()  # Automatisches Formatieren der Datumsanzeige für bessere Lesbarkeit
    plt.grid(True)  # Hinzufügen eines Gitternetzes für bessere Lesbarkeit
    plt.show()

    # Berechnung des Root Mean Square Error (RMSE) als Leistungsmetrik
    rmse = np.sqrt(np.mean((vorhersagen_denorm - tatsächliche_denorm) ** 2))
    print(f'RMSE auf Testdaten: {rmse:.2f}')

    # Zusammenfassen der Ergebnisse in einem DataFrame für die CSV
    ergebnisse = pd.DataFrame({
        'Date': test_dates,
        'Vorhergesagt': vorhersagen_denorm,
        'Tatsächlich': tatsächliche_denorm,
        'Differenz': differenz
    })

    # Speichern der Ergebnisse in einer CSV-Datei
    ergebnisse.to_csv(speicher_pfad, index=False)
    print(f'Ergebnisse wurden in {speicher_pfad} gespeichert.')

    return vorhersagen_denorm, tatsächliche_denorm, differenz

# 6. Hauptfunktion
def haupt(bollinger_pfad=None, normal_pfad=None, speicher_pfad='lstm_sp500_with_advanced_bb.csv'):
    """
    Hauptfunktion, die den gesamten Workflow ausführt:
    Laden der Daten, Vorbereiten der Daten, Erstellen und Trainieren des Modells,
    und Testen des Modells.

    Args:
        bollinger_pfad (str, optional): Pfad zur Bollinger Bands CSV-Datei.
        normal_pfad (str, optional): Pfad zur normalen Daten CSV-Datei.
        speicher_pfad (str, optional): Pfad zur Speicherung der Ergebnisse-CSV. Standard ist 'lstm_sp500_with_advanced_bb.csv'.
    """
    # Schritt 1: Daten laden
    print("Daten werden geladen...")
    daten = lade_daten(bollinger_pfad, normal_pfad)
    print(f"Daten erfolgreich geladen. Gesamtanzahl der Datenpunkte: {len(daten)}")

    # Schritt 2: Daten vorbereiten
    print("Daten werden vorbereitet...")
    sequenz_länge = 60  # Länge der Eingabesequenzen
    spalten = ['Open', 'High', 'Low', 'Close', 'Volume_normal',
               'BB_Middle', 'BB_Upper', 'BB_Lower']  # Relevante Spalten für das Modell
    X_train, y_train, X_test, y_test, scaler, test_dates = bereite_daten_vor(
        daten, spalten=spalten, sequenz_länge=sequenz_länge
    )
    print("Daten erfolgreich vorbereitet.")

    # Schritt 3: Modell erstellen
    print("Erstelle das LSTM-Modell...")
    input_dim = X_train.shape[2]  # Anzahl der Features pro Zeitschritt
    modell = EinfachesLSTM(input_dim=input_dim)
    print(f"Modell erfolgreich erstellt mit input_dim={input_dim}.")

    # Schritt 4: Hyperparameter festlegen
    epochen = 100       # Anzahl der Trainingsdurchläufe
    batch_größe = 64    # Größe der Daten-Batches
    lernrate = 0.001    # Lernrate für den Optimierer
    print(f"Hyperparameter festgelegt: Epochen={epochen}, Batch-Größe={batch_größe}, Lernrate={lernrate}")

    # Schritt 5: DataLoader erstellen
    print("Erstelle DataLoader für Trainings- und Testdaten...")
    train_dataset = TensorDataset(X_train, y_train)  # Trainingsdaten als Dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_größe, shuffle=True)  # DataLoader für Training

    test_dataset = TensorDataset(X_test, y_test)      # Testdaten als Dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_größe, shuffle=False)  # DataLoader für Test
    print("DataLoader erfolgreich erstellt.")

    # Schritt 6: Gerät auswählen (CPU oder GPU)
    gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {gerät}")
    modell.to(gerät)  # Verschieben des Modells auf das gewählte Gerät

    # Schritt 7: Verlustfunktion und Optimierer definieren
    verlustfunktion = nn.MSELoss()  # Mean Squared Error als Verlustfunktion
    optimierer = torch.optim.Adam(modell.parameters(), lr=lernrate)  # Adam-Optimierer
    print("Verlustfunktion und Optimierer definiert.")

    # Schritt 8: Modell trainieren
    print("Starte das Training des Modells...")
    verlust_werte = trainiere_modell(
        modell, train_loader, verlustfunktion, optimierer, gerät, epochen=epochen
    )
    print("Training abgeschlossen.")

    # Schritt 9: Verlustkurve plotten
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochen + 1), verlust_werte, label='Training Verlust', color='green')
    plt.xlabel('Epoche')
    plt.ylabel('MSE Verlust')
    plt.title('Training Verlust über Epochen')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Schritt 10: Modell testen
    print("Teste das Modell mit den Testdaten...")
    teste_modell(modell, test_loader, scaler, gerät, test_dates, speicher_pfad=speicher_pfad)
    print("Modell erfolgreich getestet.")

# 7. Programm starten
if __name__ == "__main__":
    """
    Der Einstiegspunkt des Programms. Hier können optionale benutzerdefinierte Pfade angegeben werden.
    Andernfalls werden die Standardpfade verwendet.
    """
    # Beispiel für die Verwendung benutzerdefinierter Pfade:
    # bollinger_pfad = os.path.join(os.path.expanduser('~'), 'anderer/Pfad/BollingerDatei.csv')
    # normal_pfad = os.path.join(os.path.expanduser('~'), 'anderer/Pfad/NormalDatei.csv')
    # speicher_pfad = os.path.join(os.path.expanduser('~'), 'anderer/Pfad/lstm_sp500_with_advanced_bb.csv')
    # haupt(bollinger_pfad, normal_pfad, speicher_pfad)

    # Ohne Argumente, verwendet die Standardpfade
    haupt()
