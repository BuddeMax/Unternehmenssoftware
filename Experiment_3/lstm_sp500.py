import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# 🔧 Geräteeinrichtung
# ----------------------------
# Bestimmt, ob eine GPU verfügbar ist und setzt das entsprechende Gerät.
# Dies ist wichtig für die Beschleunigung von Trainingsprozessen bei großen Modellen.
gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Verwendetes Gerät: {gerät}")


# ----------------------------
# 📂 Datenladen
# ----------------------------
def lade_daten(pfad=None):
    """
    Lädt die historischen SP500-Daten aus einer CSV-Datei.

    Diese Funktion liest die SP500-Historien aus einer CSV-Datei, konvertiert die 'Date'-Spalte
    in Datetime-Objekte, sortiert die Daten chronologisch und setzt den Index zurück.

    Args:
        pfad (str, optional): Der Pfad zur CSV-Datei mit den SP500-Daten.
                               Wenn kein Pfad angegeben wird, wird ein Standardpfad verwendet.

    Returns:
        pd.DataFrame: Ein DataFrame mit den geladenen und sortierten SP500-Daten.

    Raises:
        FileNotFoundError: Wenn die angegebene Datei nicht gefunden wird.
    """
    if pfad is None:
        # Heimverzeichnis des aktuellen Benutzers ermitteln
        home_dir = os.path.expanduser('~')
        # Relativen Pfad zum Datenverzeichnis hinzufügen
        pfad = os.path.join(home_dir, 'Unternehmenssoftware', 'shared_resources', 'sp500_data',
                            'SP500_Index_Historical_Data.csv')

    # Überprüfen, ob die Datei existiert
    if not os.path.exists(pfad):
        raise FileNotFoundError(f"❌ Die Datei wurde nicht gefunden: {pfad}")

    # Laden der Daten aus der CSV-Datei
    daten = pd.read_csv(pfad, parse_dates=['Date'])

    # Sicherstellen, dass die Daten nach Datum sortiert sind
    daten.sort_values('Date', inplace=True)
    daten.reset_index(drop=True, inplace=True)

    return daten


# ----------------------------
# 📊 Datenvorbereitung
# ----------------------------
def bereite_daten_vor(daten, spalten=['Open', 'High', 'Low', 'Close', 'Volume'], sequenz_länge=60):
    """
    Bereitet die Daten für das Training und Testen des Modells vor.

    Diese Funktion skaliert die angegebenen Merkmale, erstellt Sequenzen und teilt die Daten
    in Trainings- und Testsets auf.

    Args:
        daten (pd.DataFrame): Das vollständige DataFrame mit den SP500-Daten.
        spalten (list, optional): Liste der zu skalierenden Merkmale. Standard ist ['Open', 'High', 'Low', 'Close', 'Volume'].
        sequenz_länge (int, optional): Die Länge jeder Sequenz (Anzahl der Zeitschritte). Standard ist 60.

    Returns:
        tuple: Enthält die folgenden Elemente:
            - X_train (torch.Tensor): Trainingssequenzen.
            - y_train (torch.Tensor): Trainingslabels.
            - X_test (torch.Tensor): Testsequenzen.
            - y_test (torch.Tensor): Testlabels.
            - scaler (MinMaxScaler): Der verwendete Scaler zur Denormalisierung.
            - dates_train (np.ndarray): Datumsangaben für das Trainingsset.
            - dates_test (np.ndarray): Datumsangaben für das Testset.
    """
    # Initialisieren des MinMaxScaler zur Skalierung der Daten zwischen 0 und 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    skaliert = scaler.fit_transform(daten[spalten].values)

    X, y = [], []
    dates = []
    for i in range(sequenz_länge, len(skaliert)):
        # Eingabesequenz: die vorherigen 'sequenz_länge' Zeitpunkte
        X.append(skaliert[i - sequenz_länge:i])
        # Label: der 'Open'-Preis zum aktuellen Zeitpunkt
        y.append(skaliert[i, 0])  # 0=Open Zielwert (Open-Preis)
        # Datum des Zielwerts
        dates.append(daten['Date'].iloc[i])

    X, y = np.array(X), np.array(y)
    dates = np.array(dates)

    # Daten in Training und Test aufteilen (80% Training, 20% Test)
    training = int(0.8 * len(X))
    X_train, X_test = X[:training], X[training:]
    y_train, y_test = y[:training], y[training:]
    dates_train, dates_test = dates[:training], dates[training:]

    # In Torch-Tensoren umwandeln
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler, dates_train, dates_test


# ----------------------------
# 🤖 Einfaches LSTM-Modell definieren
# ----------------------------
class EinfachesLSTM(nn.Module):
    """
    Einfaches LSTM-Modell für die Vorhersage des 'Open'-Preises des SP500.

    Dieses Modell besteht aus einer LSTM-Schicht gefolgt von einer vollverbundenen Schicht,
    die den finalen Vorhersagewert berechnet.

    Attributes:
        lstm (nn.LSTM): Die LSTM-Schicht zur Verarbeitung der Sequenzdaten.
        fc (nn.Linear): Die vollverbundene Schicht zur Ausgabe der Vorhersage.
    """

    def __init__(self, input_dim=5, hidden_dim=50, output_dim=1, num_layers=2):
        """
        Initialisiert das EinfachesLSTM-Modell.

        Args:
            input_dim (int, optional): Die Anzahl der Eingangsmerkmale (Features) pro Zeitschritt. Standard ist 5.
            hidden_dim (int, optional): Die Anzahl der Neuronen in der LSTM-Schicht. Standard ist 50.
            output_dim (int, optional): Die Anzahl der Ausgangsmerkmale (in diesem Fall 1 für den 'Open'-Preis). Standard ist 1.
            num_layers (int, optional): Die Anzahl der LSTM-Schichten. Mehr Schichten können komplexere Muster lernen. Standard ist 2.
        """
        super(EinfachesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Führt eine Vorwärtsdurchführung durch.

        Diese Methode definiert, wie die Eingabedaten durch das Netzwerk fließen.

        Args:
            x (torch.Tensor): Die Eingabesequenz mit der Form (Batch, Sequenz, Feature).

        Returns:
            torch.Tensor: Die vorhergesagten 'Open'-Preise mit der Form (Batch, Output_Size).
        """
        out, _ = self.lstm(x)  # Durchläuft die LSTM-Schicht
        out = out[:, -1, :]  # Nimmt den letzten Zeitschritt der LSTM-Ausgabe
        out = self.fc(out)  # Führt die Ausgabe durch die vollverbundene Schicht
        return out


# ----------------------------
# 🏋️‍♂️ Modell trainieren
# ----------------------------
def trainiere_modell(modell, daten_loader, verlustfunktion, optimierer, gerät, epochen=100):
    """
    Trainiert das gegebene Modell mit den bereitgestellten Daten.

    Diese Funktion führt das Training des Modells über eine festgelegte Anzahl von Epochen durch.
    Sie berechnet den Verlust, führt die Rückwärtsdurchführung durch und aktualisiert die Modellgewichte.

    Args:
        modell (nn.Module): Das zu trainierende Modell.
        daten_loader (DataLoader): Der DataLoader für das Trainingsset.
        verlustfunktion (nn.Module): Die Verlustfunktion zur Berechnung des Fehlers.
        optimierer (torch.optim.Optimizer): Der Optimierer zur Aktualisierung der Modellgewichte.
        gerät (torch.device): Das Gerät (CPU/GPU), auf dem das Training durchgeführt wird.
        epochen (int, optional): Die Anzahl der Trainingsdurchläufe. Standard ist 100.

    Returns:
        list: Eine Liste der durchschnittlichen Verlustwerte pro Epoche.
    """
    verlust_werte = []
    for epoch in range(epochen):
        modell.train()  # Setzt das Modell in den Trainingsmodus (aktiviert Dropout, BatchNorm etc.)
        gesamt_verlust = 0  # Initialisiert den Verlust für die aktuelle Epoche

        # Iteriert über alle Batches im Trainingsloader
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)

            optimierer.zero_grad()  # Gradienten zurücksetzen
            vorhersage = modell(inputs)  # Vorhersage berechnen
            verlust = verlustfunktion(vorhersage, labels)  # Verlust berechnen
            verlust.backward()  # Rückwärtsdurchlauf (Gradienten berechnen)
            optimierer.step()  # Gewichte aktualisieren

            gesamt_verlust += verlust.item() * inputs.size(0)  # Summiert den Verlust über alle Samples

        durchschnitt_verlust = gesamt_verlust / len(daten_loader.dataset)  # Durchschnittlicher Verlust pro Sample
        verlust_werte.append(durchschnitt_verlust)

        # Gibt den Verlust alle 10 Epochen und in der ersten Epoche aus.
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoche {epoch + 1}/{epochen}, Verlust: {durchschnitt_verlust:.6f}')

    return verlust_werte


# ----------------------------
# 🧪 Modell testen
# ----------------------------
def teste_modell(modell, daten_loader, scaler, gerät, dates_test, speichere_csv=False, csv_pfad='vorhersagen.csv'):
    """
    Testet das trainierte Modell mit den Testdaten und visualisiert die Ergebnisse.

    Diese Funktion führt Vorhersagen auf den Testdaten durch, denormalisiert die Ergebnisse,
    berechnet Fehlermaße und erstellt einen Plot der tatsächlichen vs. vorhergesagten Werte.

    Args:
        modell (nn.Module): Das trainierte Modell.
        daten_loader (DataLoader): Der DataLoader für das Testset.
        scaler (MinMaxScaler): Der verwendete Scaler zur Denormalisierung.
        gerät (torch.device): Das Gerät (CPU/GPU), auf dem das Modell ausgeführt wird.
        dates_test (np.ndarray): Die Datumsangaben für das Testset.
        speichere_csv (bool, optional): Gibt an, ob die Vorhersagen in einer CSV-Datei gespeichert werden sollen. Standard ist False.
        csv_pfad (str, optional): Der Pfad zur CSV-Datei, in der die Vorhersagen gespeichert werden sollen. Standard ist 'vorhersagen.csv'.

    Returns:
        tuple: Enthält die folgenden Elemente:
            - vorhersagen_denorm (np.ndarray): Denormalisierte vorhergesagte 'Open'-Preise.
            - tatsächliche_denorm (np.ndarray): Denormalisierte tatsächliche 'Open'-Preise.
            - alle_dates (np.ndarray): Die entsprechenden Datumsangaben für die Vorhersagen.
    """
    modell.eval()  # Setzt das Modell in den Evaluierungsmodus (deaktiviert Dropout etc.)
    vorhersagen, tatsächliche = [], []
    alle_dates = []

    with torch.no_grad():  # Deaktiviert die Gradientenberechnung für effizienteres Testen
        for batch_idx, (inputs, labels) in enumerate(daten_loader):
            inputs, labels = inputs.to(gerät), labels.to(gerät)
            outputs = modell(inputs)  # Vorhersage berechnen
            vorhersagen.append(outputs.cpu().numpy())
            tatsächliche.append(labels.cpu().numpy())

            # Berechne die Start- und Endindizes für die aktuellen Batch-Daten
            start = batch_idx * daten_loader.batch_size
            end = start + inputs.size(0)

            # Füge die entsprechenden Datumswerte hinzu
            alle_dates.extend(dates_test[start:end])

    # Konvertiere die Listen in numpy-Arrays
    vorhersagen = np.concatenate(vorhersagen).flatten()
    tatsächliche = np.concatenate(tatsächliche).flatten()
    alle_dates = np.array(alle_dates)

    # Denormalisieren der 'Open'-Preise (zurückskalieren)
    vorhersagen_denorm = scaler.inverse_transform(
        np.hstack((
            vorhersagen.reshape(-1, 1),  # Vorhersagen in Spalte 0 (Open)
            np.zeros((vorhersagen.shape[0], 4))  # Nullen für die anderen Spalten
        ))
    )[:, 0]  # Erste Spalte auswählen, da dort Open steht

    tatsächliche_denorm = scaler.inverse_transform(
        np.hstack((
            tatsächliche.reshape(-1, 1),
            np.zeros((tatsächliche.shape[0], 4))
        ))
    )[:, 0]

    # Überprüfen, ob die Längen der Daten übereinstimmen
    if not (len(alle_dates) == len(tatsächliche_denorm) == len(vorhersagen_denorm)):
        raise ValueError("Die Längen von Datum, tatsächlichen Werten und Vorhersagen müssen übereinstimmen.")

    # Ergebnisse plotten mit Datum auf der X-Achse
    plt.figure(figsize=(12, 6))
    plt.plot(alle_dates, tatsächliche_denorm, label='Tatsächlicher Preis', alpha=0.7)
    plt.plot(alle_dates, vorhersagen_denorm, label='Vorhergesagter Preis', alpha=0.7)
    plt.xlabel('Datum')
    plt.ylabel('S&P 500 Open Preis')
    plt.title('Tatsächlicher vs. Vorhergesagter S&P 500 Open Preis')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Fehlermaße berechnen
    rmse = np.sqrt(np.mean((vorhersagen_denorm - tatsächliche_denorm) ** 2))
    mae = np.mean(np.abs(vorhersagen_denorm - tatsächliche_denorm))
    mape = np.mean(np.abs((tatsächliche_denorm - vorhersagen_denorm) / tatsächliche_denorm)) * 100

    print(f'RMSE auf Testdaten: {rmse:.2f}')
    print(f'MAE auf Testdaten: {mae:.2f}')
    print(f'MAPE auf Testdaten: {mape:.2f}%')

    # Optional: Speichern der Vorhersagen und tatsächlichen Werte als CSV
    if speichere_csv:
        ergebnisse_df = pd.DataFrame({
            'Datum': alle_dates,
            'Tatsächlich': tatsächliche_denorm,
            'Vorhergesagt': vorhersagen_denorm
        })
        # Sicherstellen, dass das Verzeichnis existiert
        if os.path.dirname(csv_pfad):
            os.makedirs(os.path.dirname(csv_pfad), exist_ok=True)
        ergebnisse_df.to_csv(csv_pfad, index=False)
        print(f'Vorhersagen und tatsächliche Werte wurden in {csv_pfad} gespeichert.')

    return vorhersagen_denorm, tatsächliche_denorm, alle_dates


# ----------------------------
# 🏁 Hauptfunktion
# ----------------------------
def haupt():
    """
    Hauptfunktion zur Ausführung des gesamten Prozesses.

    Dieser Prozess umfasst:
        1. Laden der Daten.
        2. Vorbereiten der Daten (Skalierung, Sequenzbildung, Aufteilung).
        3. Erstellen des LSTM-Modells.
        4. Festlegen der Hyperparameter.
        5. Erstellen der DataLoader für Training und Test.
        6. Definieren der Verlustfunktion und des Optimierers.
        7. Trainieren des Modells.
        8. Plotten der Trainingsverlustkurve.
        9. Testen des Modells und Visualisieren der Ergebnisse.
    """
    # Schritt 1: Daten laden
    print("📥 Daten werden geladen...")
    daten = lade_daten()
    print("✅ Daten erfolgreich geladen.")

    # Schritt 2: Daten vorbereiten
    print("🔧 Daten werden vorbereitet...")
    sequenz_länge = 60  # Anzahl der Zeitschritte pro Sequenz
    X_train, y_train, X_test, y_test, scaler, dates_train, dates_test = bereite_daten_vor(
        daten, sequenz_länge=sequenz_länge
    )
    print("✅ Daten erfolgreich vorbereitet.")

    # Schritt 3: Modell erstellen
    print("🤖 Erstelle das LSTM-Modell...")
    modell = EinfachesLSTM()
    modell.to(gerät)  # Modell auf das ausgewählte Gerät verschieben
    print("✅ Modell erfolgreich erstellt und auf das Gerät verschoben.")

    # Schritt 4: Hyperparameter festlegen
    epochen = 100
    batch_größe = 64
    lernrate = 0.001
    print(f"⚙️ Hyperparameter festgelegt: Epochen={epochen}, Batch-Größe={batch_größe}, Lernrate={lernrate}")

    # Schritt 5: DataLoader erstellen
    print("📦 Erstelle DataLoader für Trainings- und Testdaten...")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_größe, shuffle=True)

    # Für Test-DataLoader müssen wir auch die Datumsinformationen berücksichtigen
    # Da DataLoader standardmäßig nur die Tensoren liefert, verarbeiten wir die Test-Daten separat
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_größe, shuffle=False)
    print("✅ DataLoader erfolgreich erstellt.")

    # Schritt 6: Verlustfunktion und Optimierer definieren
    verlustfunktion = nn.MSELoss()
    optimierer = torch.optim.Adam(modell.parameters(), lr=lernrate)
    print("🔍 Verlustfunktion und Optimierer definiert.")

    # Schritt 7: Modell trainieren
    print("🏋️‍♂️ Starte das Training des Modells...")
    verlust_werte = trainiere_modell(modell, train_loader, verlustfunktion, optimierer, gerät, epochen=epochen)
    print("🏁 Training abgeschlossen.")

    # Schritt 8: Verlustkurve plotten
    print("📈 Plotten der Trainingsverlustkurve...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochen + 1), verlust_werte, label='Training Verlust')
    plt.xlabel('Epoche')
    plt.ylabel('MSE Verlust')
    plt.title('Training Verlust über Epochen')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("✅ Trainingsverlustkurve erfolgreich geplottet.")

    # Schritt 9: Modell testen
    print("🔬 Testen des Modells mit den Testdaten...")
    # Setzen Sie speichere_csv auf True, um die Vorhersagen als CSV zu speichern
    vorhersagen_denorm, tatsächliche_denorm, alle_dates = teste_modell(
        modell, test_loader, scaler, gerät, dates_test, speichere_csv=True, csv_pfad='vorhersagen.csv'
    )
    print("✅ Modell erfolgreich getestet und Ergebnisse visualisiert.")


# ----------------------------
# 🏁 Programm starten
# ----------------------------
if __name__ == "__main__":
    """
    Der Einstiegspunkt des Skripts.

    Wenn dieses Skript direkt ausgeführt wird, startet es die Hauptfunktion.
    Dies ermöglicht es, den Code sowohl als Modul zu importieren als auch direkt auszuführen.
    """
    haupt()
