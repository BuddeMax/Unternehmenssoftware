import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # Importieren von Matplotlib für die Visualisierung

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
# Bestimmt das Verzeichnis des aktuellen Skripts und baut den Pfad zur CSV-Datei.
script_dir = os.path.dirname(__file__)  # Verzeichnis des aktuellen Skripts
file_path = os.path.join(script_dir, "../shared_resources/sp500_data/SP500_Index_Historical_Data.csv")


# ----------------------------
# 📊 Datenvorbereitung
# ----------------------------
def load_and_prepare_data(file_path):
    """
    Daten laden, Datumsangaben parsen und sortieren.

    Diese Funktion liest die historischen SP500-Daten aus einer CSV-Datei, parst die Datumsangaben
    und sortiert die Daten chronologisch. Dies ist essentiell, um sicherzustellen, dass die
    Zeitreihenanalyse korrekt durchgeführt wird.

    Args:
        file_path (str): Der Pfad zur CSV-Datei mit den SP500-Daten.

    Returns:
        pd.DataFrame: Ein DataFrame mit den geladenen und sortierten Daten.

    Raises:
        FileNotFoundError: Wenn die CSV-Datei nicht gefunden wird.
        pd.errors.EmptyDataError: Wenn die CSV-Datei leer ist.
        pd.errors.ParserError: Wenn die CSV-Datei nicht korrekt formatiert ist.
    """
    # Überprüft, ob die Datei existiert.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden unter: {file_path}")

    # Liest die CSV-Datei ein.
    data = pd.read_csv(file_path)

    # Konvertiert die 'Date'-Spalte in Datetime-Objekte.
    data['Date'] = pd.to_datetime(data['Date'])

    # Sortiert die Daten nach Datum aufsteigend.
    data = data.sort_values(by='Date').reset_index(drop=True)

    return data


def split_train_test(data, train_start, train_end, test_start, test_end):
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
    train = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].reset_index(drop=True)

    # Filtert die Daten für das Testset basierend auf den Start- und Enddaten.
    test = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)].reset_index(drop=True)

    # Überprüft, ob die Trainings- und Testsets nicht leer sind.
    if train.empty:
        raise ValueError(
            f"Trainingsdaten sind leer. Überprüfen Sie die Trainingsdatumsbereiche: {train_start} bis {train_end}.")
    if test.empty:
        raise ValueError(f"Testdaten sind leer. Überprüfen Sie die Testdatumsbereiche: {test_start} bis {test_end}.")

    return train, test


def scale_features(train, test, feature_columns):
    """
    Daten skalieren.

    Diese Funktion skaliert die angegebenen Merkmale der Trainings- und Testdaten auf einen
    Bereich zwischen 0 und 1 unter Verwendung des MinMaxScaler. Skalierung ist ein wichtiger
    Schritt in der Datenvorverarbeitung, um sicherzustellen, dass alle Features den gleichen
    Einfluss auf das Modell haben.

    Args:
        train (pd.DataFrame): Das Trainings-DataFrame.
        test (pd.DataFrame): Das Test-DataFrame.
        feature_columns (list): Liste der zu skalierenden Merkmale.

    Returns:
        tuple: Skalierte Trainingsdaten, skalierte Testdaten und der verwendete Scaler.
    """
    scaler = MinMaxScaler()

    # Passt den Scaler an die Trainingsdaten an und transformiert diese.
    train_scaled = scaler.fit_transform(train[feature_columns])

    # Transformiert die Testdaten basierend auf dem im Trainingsset angepassten Scaler.
    test_scaled = scaler.transform(test[feature_columns])

    return train_scaled, test_scaled, scaler


def create_sequences(data, seq_length):
    """
    Sequenzen und Labels erstellen.

    Diese Funktion erstellt aus den skalisierten Daten Sequenzen fester Länge, die als
    Eingaben für das RNN-Modell dienen, sowie die zugehörigen Labels, die die zukünftigen
    Werte repräsentieren, die vorhergesagt werden sollen.

    Args:
        data (np.ndarray): Das skalierte Datenarray.
        seq_length (int): Die Länge jeder Sequenz (Anzahl der Zeitschritte).

    Returns:
        tuple: Ein Tupel bestehend aus den Sequenzen und den Labels.

    Raises:
        IndexError: Wenn der angegebene Feature-Index außerhalb der Datenmatrix liegt.
    """
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        # Extrahiert eine Sequenz von seq_length Zeitpunkten.
        sequences.append(data[i:i + seq_length])
        try:
            # Extrahiert das Label, das den 'Close'-Preis nach der Sequenz darstellt.
            labels.append(data[i + seq_length, 3])  # 'Close' ist an Index 3
        except IndexError:
            raise IndexError(f"Feature-Index 3 ist außerhalb der Datenmatrix mit {data.shape[1]} Features.")
    return np.array(sequences), np.array(labels)


# ----------------------------
# 🧠 Datenvorbereitung ausführen
# ----------------------------
# Laden und vorbereiten der Daten.
data = load_and_prepare_data(file_path)

# Aufteilen der Daten in Trainings- und Testsets.
train_data, test_data = split_train_test(data, '1994-01-01', '2015-12-31', '2016-01-01', '2024-12-31')

# Definieren der zu skalierenden Features.
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Skalieren der Features.
train_scaled, test_scaled, scaler = scale_features(train_data, test_data, features)

# Erstellen von Sequenzen für das RNN-Modell.
seq_length = 10  # Anzahl der Zeitschritte pro Sequenz
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Umwandeln der numpy-Arrays in PyTorch-Tensoren.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# ----------------------------
# 🤖 RNN-Modell definieren
# ----------------------------
class StockRNN(nn.Module):
    """
    Einfaches RNN-Modell für Aktienvorhersagen.

    Dieses Modell verwendet ein einfaches RNN mit mehreren Schichten zur Vorhersage der zukünftigen
    Schlusskurse des SP500-Index.

    Attributes:
        rnn (nn.RNN): Das RNN-Modul, das die Sequenzdaten verarbeitet.
        fc (nn.Linear): Die vollverbundene Schicht zur Ausgabe der finalen Vorhersage.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        Initialisiert das StockRNN-Modell.

        Args:
            input_size (int): Die Anzahl der Eingangsmerkmale (Features) pro Zeitschritt.
            hidden_size (int): Die Anzahl der Neuronen in den RNN-Schichten.
            output_size (int): Die Anzahl der Ausgangsmerkmale (in diesem Fall 1 für den 'Close'-Preis).
            num_layers (int, optional): Die Anzahl der RNN-Schichten. Mehr Schichten können komplexere Muster lernen.
                Standard ist 2.
        """
        super(StockRNN, self).__init__()
        # Initialisiert das RNN-Modul mit den angegebenen Parametern.
        # batch_first=True bedeutet, dass die Eingaben die Form (Batch, Sequenz, Feature) haben.
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Die vollverbundene Schicht transformiert die RNN-Ausgabe in die gewünschte Ausgabegröße.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Führt eine Vorwärtsdurchführung durch.

        Diese Methode definiert, wie die Eingabedaten durch das Netzwerk fließen.

        Args:
            x (torch.Tensor): Die Eingabesequenz mit der Form (Batch, Sequenz, Feature).

        Returns:
            torch.Tensor: Die vorhergesagten Werte mit der Form (Batch, Output_Size).
        """
        # Durchläuft die Eingabe durch das RNN-Modul.
        out, _ = self.rnn(x)

        # Extrahiert die Ausgabe des letzten Zeitschritts jeder Sequenz.
        # Dies repräsentiert die aggregierte Information der gesamten Sequenz.
        out = self.fc(out[:, -1, :])
        return out


# ----------------------------
# 🏋️‍♂️ Modell- und Trainingsparameter definieren
# ----------------------------
input_size = 5  # Anzahl der Eingangsmerkmale
hidden_size = 64  # Anzahl der Neuronen in den RNN-Schichten
output_size = 1  # Anzahl der Ausgangsmerkmale (1 für 'Close')
num_layers = 2  # Anzahl der RNN-Schichten

# Initialisiert das RNN-Modell und verschiebt es auf das festgelegte Gerät (CPU/GPU).
model = StockRNN(input_size, hidden_size, output_size, num_layers).to(device)

# Definiert die Verlustfunktion. MSELoss ist geeignet für Regressionsaufgaben.
criterion = nn.MSELoss()

# Definiert den Optimierer. Adam ist ein weit verbreiteter Optimierer, der adaptiv die Lernrate anpasst.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 🏋️‍♂️ Training des Modells
# ----------------------------
num_epochs = 50  # Gesamtanzahl der Trainingsdurchläufe
batch_size = 64  # Anzahl der Samples pro Batch

for epoch in range(num_epochs):
    model.train()  # Setzt das Modell in den Trainingsmodus (aktiviert Dropout, BatchNorm etc.)
    epoch_loss = 0  # Initialisiert den Verlust für die aktuelle Epoche

    # Iteriert über alle Batches im Trainingsset.
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size].to(device)  # Extrahiert und verschiebt den Batch auf das Gerät
        y_batch = y_train_tensor[i:i + batch_size].to(device)

        # Führt eine Vorwärtsdurchführung durch.
        outputs = model(X_batch)

        # Berechnet den Verlust zwischen den Vorhersagen und den tatsächlichen Werten.
        loss = criterion(outputs, y_batch)

        # Setzt die Gradienten des Optimierers zurück.
        optimizer.zero_grad()

        # Führt die Rückwärtsdurchführung durch (Backpropagation).
        loss.backward()

        # Aktualisiert die Modellgewichte.
        optimizer.step()

        # Summiert den Verlust über alle Batches.
        epoch_loss += loss.item() * X_batch.size(0)

    # Berechnet den durchschnittlichen Verlust pro Sample.
    avg_loss = epoch_loss / len(X_train_tensor)

    # Gibt den Verlust nach jeder Epoche aus.
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

# ----------------------------
# 🧪 Testen des Modells
# ----------------------------
model.eval()  # Setzt das Modell in den Evaluationsmodus (deaktiviert Dropout etc.)
with torch.no_grad():  # Deaktiviert die Gradientenberechnung für effizienteres Testen
    # Führt eine Vorhersage auf den Testdaten durch.
    predictions = model(X_test_tensor.to(device)).squeeze().cpu().numpy()

    # Berechnet den Testverlust.
    test_loss = criterion(torch.tensor(predictions), y_test_tensor.squeeze()).item()

print(f"Test Loss: {test_loss:.6f}")


# ----------------------------
# 📈 Ergebnisse visualisieren
# ----------------------------
def denormalize_data(scaler, data, feature_index=3):
    """
    Denormalisiert die 'Close'-Preise.

    Diese Funktion wandelt die normalisierten 'Close'-Preise zurück in ihre ursprünglichen Werte.

    Args:
        scaler (MinMaxScaler): Der verwendete Scaler zur Denormalisierung.
        data (np.ndarray): Die normalisierten Daten (Vorhersagen oder tatsächliche Werte).
        feature_index (int, optional): Der Index des Features, das den 'Close'-Preis repräsentiert.
            Standard ist 3.

    Returns:
        np.ndarray: Die denormalisierten 'Close'-Preise.
    """
    # Erstellt ein Array mit Nullen für die nicht interessierenden Features.
    dummy = np.zeros((len(data), scaler.scale_.shape[0]))

    # Setzt die 'Close'-Preise an der entsprechenden Position.
    dummy[:, feature_index] = data

    # Denormalisiert die Daten.
    denormalized = scaler.inverse_transform(dummy)[:, feature_index]

    return denormalized


# Denormalisiert die tatsächlichen und vorhergesagten 'Close'-Preise.
y_test_denormalized = denormalize_data(scaler, y_test)
predictions_denormalized = denormalize_data(scaler, predictions)

# Überprüft, ob die Längen der Daten übereinstimmen.
if not (len(test_data['Date'][seq_length:]) == len(y_test_denormalized) == len(predictions_denormalized)):
    raise ValueError("Die Längen von Datum, tatsächlichen Werten und Vorhersagen müssen übereinstimmen.")

# Plotten der Ergebnisse.
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'][seq_length:], y_test_denormalized, label='Tatsächliche Werte', alpha=0.7)
plt.plot(test_data['Date'][seq_length:], predictions_denormalized, label='Vorhersagen', alpha=0.7)
plt.xlabel('Datum')
plt.ylabel('SP500 Schlusskurs')
plt.title('Tatsächliche vs Vorhergesagte Werte')
plt.legend()
plt.grid(True)
plt.show()
