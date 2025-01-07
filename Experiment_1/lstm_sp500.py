import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ----------------------------
# 🔧 Geräteeinrichtung
# ----------------------------
# Bestimmt, ob eine GPU verfügbar ist und setzt das entsprechende Gerät.
# Dies ist wichtig für die Beschleunigung von Trainingsprozessen bei großen Modellen.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Verwendetes Gerät: {device}")


# ----------------------------
# 📂 Dynamischer Pfad zur Datei
# ----------------------------
def get_file_path():
    """
    Ermittle den Pfad zur Datendatei dynamisch.

    Diese Funktion berechnet den absoluten Pfad zur CSV-Datei mit den historischen SP500-Daten,
    indem sie das Verzeichnis der aktuellen Datei nutzt und relativ zum Projektstammverzeichnis
    navigiert. Dies ermöglicht eine flexible Platzierung der Daten innerhalb der Projektstruktur.

    Returns:
        str: Der absolute Pfad zur Datendatei.

    Raises:
        FileNotFoundError: Wenn die Datendatei nicht am ermittelten Pfad gefunden wird.
    """
    # Bestimmt das Stammverzeichnis des Projekts, indem es ein Verzeichnis nach oben navigiert.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Baut den vollständigen Pfad zur CSV-Datei zusammen.
    file_path = os.path.join(project_root, "shared_resources", "sp500_data", "SP500_Index_Historical_Data.csv")

    # Debug-Ausgabe: Zeigt den ermittelten Pfad an.
    print(f"🔍 Datenpfad: {file_path}")

    # Überprüft, ob die Datei existiert. Wenn nicht, wird eine Ausnahme ausgelöst.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden unter: {file_path}")
    return file_path


# ----------------------------
# 📊 Datenvorbereitung
# ----------------------------
def load_data(file_path):
    """
    Daten laden und sortieren.

    Diese Funktion liest die historischen SP500-Daten aus einer CSV-Datei, parst die Datumsangaben
    und sortiert die Daten chronologisch. Dies ist essentiell, um sicherzustellen, dass die
    Zeitreihenanalyse korrekt durchgeführt wird.

    Args:
        file_path (str): Der Pfad zur CSV-Datei mit den SP500-Daten.

    Returns:
        pd.DataFrame: Ein DataFrame mit den geladenen und sortierten Daten.

    Raises:
        pd.errors.EmptyDataError: Wenn die CSV-Datei leer ist.
        pd.errors.ParserError: Wenn die CSV-Datei nicht korrekt formatiert ist.
    """
    # Liest die CSV-Datei ein und parst die 'Date'-Spalte als Datumsobjekte.
    data = pd.read_csv(file_path, parse_dates=['Date'])
    # Sortiert die Daten nach dem Datum aufsteigend und setzt den Index zurück.
    data = data.sort_values(by='Date').reset_index(drop=True)
    return data


def split_data(data, train_start, train_end, test_start, test_end):
    """
    Daten in Trainings- und Testsets aufteilen.

    Diese Funktion teilt die gesamten Daten in zwei separate Datensätze:
    - Ein Trainingsset zur Modellbildung und -anpassung.
    - Ein Testset zur Evaluierung der Modellleistung.

    Die Aufteilung erfolgt basierend auf den angegebenen Datumsbereichen, um zeitliche
    Korrelationen und Überlappungen zu vermeiden.

    Args:
        data (pd.DataFrame): Das vollständige DataFrame mit den SP500-Daten.
        train_start (str): Startdatum für das Trainingsset im Format 'YYYY-MM-DD'.
        train_end (str): Enddatum für das Trainingsset im Format 'YYYY-MM-DD'.
        test_start (str): Startdatum für das Testset im Format 'YYYY-MM-DD'.
        test_end (str): Enddatum für das Testset im Format 'YYYY-MM-DD'.

    Returns:
        tuple: Ein Tupel bestehend aus dem Trainings- und Test-DataFrame.

    Raises:
        ValueError: Wenn die angegebenen Datumsbereiche nicht innerhalb der Daten liegen.
    """
    # Filtert die Daten für das Trainingsset basierend auf den Start- und Enddaten.
    train = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)]
    # Filtert die Daten für das Testset basierend auf den Start- und Enddaten.
    test = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)]

    # Überprüft, ob die Trainings- und Testsets nicht leer sind.
    if train.empty:
        raise ValueError(
            f"Trainingsdaten sind leer. Überprüfen Sie die Trainingsdatumsbereiche: {train_start} bis {train_end}.")
    if test.empty:
        raise ValueError(f"Testdaten sind leer. Überprüfen Sie die Testdatumsbereiche: {test_start} bis {test_end}.")

    return train, test


def scale_data(train, test, features):
    """
    Daten skalieren.

    Diese Funktion skaliert die angegebenen Merkmale der Trainings- und Testdaten auf einen
    Bereich zwischen 0 und 1 unter Verwendung des MinMaxScaler. Skalierung ist ein wichtiger
    Schritt in der Datenvorverarbeitung, um sicherzustellen, dass alle Features den gleichen
    Einfluss auf das Modell haben.

    Args:
        train (pd.DataFrame): Das Trainings-DataFrame.
        test (pd.DataFrame): Das Test-DataFrame.
        features (list): Liste der zu skalierenden Merkmale.

    Returns:
        tuple: Skalierte Trainingsdaten, skalierte Testdaten und der verwendete Scaler.
    """
    # Initialisiert den MinMaxScaler, der die Daten in den Bereich [0, 1] skaliert.
    scaler = MinMaxScaler()
    # Passt den Scaler an die Trainingsdaten an und transformiert diese.
    train_scaled = scaler.fit_transform(train[features])
    # Transformiert die Testdaten basierend auf dem im Trainingsset angepassten Scaler.
    test_scaled = scaler.transform(test[features])
    return train_scaled, test_scaled, scaler


def create_sequences(data, seq_length, feature_index=3):
    """
    Sequenzen und Labels erstellen.

    Diese Funktion erstellt aus den skalisierten Daten Sequenzen fester Länge, die als
    Eingaben für das LSTM-Modell dienen, sowie die zugehörigen Labels, die die zukünftigen
    Werte repräsentieren, die vorhergesagt werden sollen.

    Args:
        data (np.ndarray): Das skalierte Datenarray.
        seq_length (int): Die Länge jeder Sequenz (Anzahl der Zeitschritte).
        feature_index (int, optional): Der Index des Features, das als Label verwendet wird.
            Standard ist 3, was dem 'Close'-Preis entspricht.

    Returns:
        tuple: Ein Tupel bestehend aus den Sequenzen und den Labels.

    Raises:
        IndexError: Wenn der angegebene feature_index außerhalb der Datenmatrix liegt.
    """
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        # Extrahiert eine Sequenz von seq_length Zeitpunkten.
        sequences.append(data[i:i + seq_length])
        try:
            # Extrahiert das Label, das den 'Close'-Preis nach der Sequenz darstellt.
            labels.append(data[i + seq_length, feature_index])  # 'Close' ist an Index 3
        except IndexError:
            raise IndexError(
                f"feature_index {feature_index} ist außerhalb der Datenmatrix mit {data.shape[1]} Features.")
    return np.array(sequences), np.array(labels)


# ----------------------------
# 🤖 LSTM-Modell
# ----------------------------
class StockLSTM(nn.Module):
    """
    Optimiertes LSTM-Modell für Aktienvorhersagen.

    Dieses Modell verwendet ein bidirektionales LSTM mit mehreren Schichten und Dropout zur
    Vorhersage der zukünftigen Schlusskurse des SP500-Index. Durch die Bidirektionalität kann
    das Modell sowohl vergangene als auch zukünftige Kontextinformationen nutzen, was die
    Vorhersagegenauigkeit verbessern kann.

    Attributes:
        lstm (nn.LSTM): Das LSTM-Modul, das die Sequenzdaten verarbeitet.
        fc (nn.Linear): Die vollverbundene Schicht zur Ausgabe der finalen Vorhersage.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        """
        Initialisiert das StockLSTM-Modell.

        Args:
            input_size (int): Die Anzahl der Eingangsmerkmale (Features) pro Zeitschritt.
            hidden_size (int): Die Anzahl der Neuronen in den LSTM-Schichten.
            output_size (int): Die Anzahl der Ausgangsmerkmale (in diesem Fall 1 für den 'Close'-Preis).
            num_layers (int, optional): Die Anzahl der LSTM-Schichten. Mehr Schichten können komplexere Muster lernen.
                Standard ist 3.
            dropout (float, optional): Die Dropout-Rate zur Vermeidung von Überanpassung. Dropout deaktiviert zufällig
                einen Teil der Neuronen während des Trainings. Standard ist 0.3.
        """
        super(StockLSTM, self).__init__()
        # Initialisiert das LSTM-Modul mit den angegebenen Parametern.
        # batch_first=True bedeutet, dass die Eingaben die Form (Batch, Sequenz, Feature) haben.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Die vollverbundene Schicht transformiert die LSTM-Ausgabe in die gewünschte Ausgabegröße.
        # Da das LSTM bidirektional ist, wird die versteckte Größe mit 2 multipliziert.
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 wegen Bidirektionalität

    def forward(self, x):
        """
        Führt eine Vorwärtsdurchführung durch.

        Diese Methode definiert, wie die Eingabedaten durch das Netzwerk fließen.

        Args:
            x (torch.Tensor): Die Eingabesequenz mit der Form (Batch, Sequenz, Feature).

        Returns:
            torch.Tensor: Die vorhergesagten Werte mit der Form (Batch, Output_Size).
        """
        # Durchläuft die Eingabe durch das LSTM-Modul.
        out, _ = self.lstm(x)
        # Extrahiert die Ausgabe des letzten Zeitschritts jeder Sequenz.
        # Dies repräsentiert die aggregierte Information der gesamten Sequenz.
        out = self.fc(out[:, -1, :])  # Nur die letzte Zeitschritt-Ausgabe verwenden
        return out


# ----------------------------
# 📈 Visualisierung der Ergebnisse
# ----------------------------
def plot_results(dates, actual, predicted):
    """
    Ergebnisse plotten.

    Diese Funktion visualisiert die tatsächlichen und vorhergesagten Schlusskurse des SP500-Index
    über die Testperiode. Dies ermöglicht eine visuelle Bewertung der Modellleistung.

    Args:
        dates (pd.Series): Die Datumsangaben für die Testdaten.
        actual (list): Die tatsächlichen Schlusskurse (normalisiert).
        predicted (list): Die vorhergesagten Schlusskurse (normalisiert).

    Raises:
        ValueError: Wenn die Längen der Listen dates, actual und predicted nicht übereinstimmen.
    """
    # Überprüft, ob alle Eingabelisten die gleiche Länge haben.
    if not (len(dates) == len(actual) == len(predicted)):
        raise ValueError("Die Längen von dates, actual und predicted müssen übereinstimmen.")

    plt.figure(figsize=(14, 7))
    # Plot der tatsächlichen Werte.
    plt.plot(dates, actual, label='Tatsächliche Werte', alpha=0.7)
    # Plot der vorhergesagten Werte.
    plt.plot(dates, predicted, label='Vorhersagen', alpha=0.7)
    plt.xlabel('Datum')
    plt.ylabel('SP500 Schlusskurs (normalisiert)')
    plt.title('Tatsächliche vs. Vorhergesagte Werte (normalisiert)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ----------------------------
# 🚀 Hauptfunktion
# ----------------------------
def main():
    """
    Hauptfunktion zur Ausführung des gesamten Prozesses.

    Dieser Prozess umfasst:
        1. Ermitteln des Dateipfads zur Datenquelle.
        2. Laden und Sortieren der Daten.
        3. Aufteilen der Daten in Trainings- und Testsets.
        4. Skalieren der Daten, um die Modellkonvergenz zu verbessern.
        5. Erstellen von Sequenzen und zugehörigen Labels für das LSTM-Modell.
        6. Umwandeln der Daten in Tensoren und Erstellen von DataLoadern für effizientes Batch-Training.
        7. Definieren des LSTM-Modells, der Verlustfunktion und des Optimierers.
        8. Training des Modells über eine festgelegte Anzahl von Epochen.
        9. Evaluierung des Modells auf den Testdaten und Visualisierung der Ergebnisse.
    """
    # Schritt 1: Dynamischen Pfad zur Datei definieren
    file_path = get_file_path()

    # Schritt 2: Daten laden und sortieren
    data = load_data(file_path)

    # Schritt 3: Daten in Trainings- und Testsets aufteilen
    train_data, test_data = split_data(data, '1994-01-01', '2015-12-31', '2016-01-01', '2024-12-31')

    # Schritt 4: Features definieren und skalieren
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    train_scaled, test_scaled, scaler = scale_data(train_data, test_data, features)

    # Schritt 5: Sequenzen erstellen
    seq_length = 20  # Anzahl der Zeitschritte pro Sequenz
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Schritt 6: Tensor-Datensätze erstellen und DataLoader verwenden
    batch_size = 64  # Anzahl der Samples pro Batch

    # Wandelt die numpy-Arrays in PyTorch-Tensoren um und erstellt TensorDatasets.
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    # Erstellt DataLoader für effizientes Batch-Training und -Testen.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Schritt 7: Modell, Verlustfunktion und Optimierer definieren
    input_size = len(features)  # Anzahl der Eingangsmerkmale
    hidden_size = 128  # Anzahl der Neuronen in den LSTM-Schichten
    output_size = 1  # Anzahl der Ausgangsmerkmale (1 für 'Close')
    num_layers = 4  # Anzahl der LSTM-Schichten
    dropout = 0.3  # Dropout-Rate

    # Initialisiert das LSTM-Modell und verschiebt es auf das festgelegte Gerät (CPU/GPU).
    model = StockLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)

    # Definiert die Verlustfunktion. SmoothL1Loss ist weniger empfindlich gegenüber Ausreißern als MSELoss.
    criterion = nn.SmoothL1Loss()

    # Definiert den Optimierer. Adam ist ein weit verbreiteter Optimierer, der adaptiv die Lernrate anpasst.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Definiert einen Lernraten-Scheduler, der die Lernrate alle 10 Epochen um den Faktor 0.5 reduziert.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Schritt 8: Training des Modells
    num_epochs = 100  # Gesamtanzahl der Trainingsdurchläufe

    for epoch in range(1, num_epochs + 1):
        model.train()  # Setzt das Modell in den Trainingsmodus (aktiviert Dropout, BatchNorm etc.)
        epoch_loss = 0.0  # Initialisiert den Verlust für die aktuelle Epoche

        # Iteriert über alle Batches im Trainingsloader
        for X_batch, y_batch in train_loader:
            # Verschiebt die Daten auf das festgelegte Gerät (CPU/GPU)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()  # Setzt die Gradienten des Optimierers zurück
            outputs = model(X_batch)  # Führt eine Vorwärtsdurchführung durch
            loss = criterion(outputs, y_batch)  # Berechnet den Verlust
            loss.backward()  # Führt die Rückwärtsdurchführung durch (Backpropagation)
            optimizer.step()  # Aktualisiert die Modellgewichte

            epoch_loss += loss.item() * X_batch.size(0)  # Summiert den Verlust über alle Samples

        scheduler.step()  # Aktualisiert die Lernrate gemäß dem Scheduler
        avg_loss = epoch_loss / len(train_loader.dataset)  # Durchschnittlicher Verlust pro Sample

        # Gibt den Verlust alle 10 Epochen und in der ersten Epoche aus.
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Schritt 9: Testen des Modells und Visualisierung der Ergebnisse
    model.eval()  # Setzt das Modell in den Evaluationsmodus (deaktiviert Dropout etc.)
    predictions = []  # Liste zur Speicherung der Vorhersagen
    actuals = []  # Liste zur Speicherung der tatsächlichen Werte

    with torch.no_grad():  # Deaktiviert die Gradientenberechnung für effizienteres Testen
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)  # Verschiebt die Eingabedaten auf das Gerät
            preds = model(
                X_batch).cpu().numpy()  # Führt eine Vorhersage durch und verschiebt die Ergebnisse zurück auf die CPU
            predictions.extend(preds.flatten())  # Fügt die Vorhersagen zur Liste hinzu
            actuals.extend(y_batch.numpy().flatten())  # Fügt die tatsächlichen Werte zur Liste hinzu

    # Extrahiert die entsprechenden Datumsangaben für die Testperiode, abzüglich der Sequenzlänge.
    test_dates = test_data['Date'].iloc[seq_length:].reset_index(drop=True)

    # Visualisiert die tatsächlichen vs. vorhergesagten Werte.
    plot_results(test_dates, actuals, predictions)


# ----------------------------
# 🔑 Einstiegspunkt
# ----------------------------
if __name__ == "__main__":
    """
    Der Einstiegspunkt des Skripts.

    Wenn dieses Skript direkt ausgeführt wird, startet es die Hauptfunktion.
    Dies ermöglicht es, den Code sowohl als Modul zu importieren als auch direkt auszuführen.
    """
    main()
