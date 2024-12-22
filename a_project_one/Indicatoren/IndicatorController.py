import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class IndicatorController:
    """
    Eine Controller-Klasse zur Verwaltung und Verarbeitung von Indikatoren aus CSV-Dateien
    für die Verwendung in LSTM-Modellen.
    """

    def __init__(self, data_dir="BollingerBands", file_name="bollinger_bands.csv"):
        """
        Initialisiert den Controller mit dem Pfad zu den Indikatordaten.

        Parameters:
        - data_dir (str): Verzeichnis, in dem sich die CSV-Datei befindet.
        - file_name (str): Name der CSV-Datei mit den Indikatordaten.
        """
        self.data_dir = data_dir
        self.file_name = file_name
        self.file_path = os.path.join(data_dir, file_name)
        self.indicator_data = None
        self.scaler = MinMaxScaler()

    def load_indicator_data(self):
        """
        Lädt die Indikatordaten aus der CSV-Datei.

        Raises:
        - FileNotFoundError: Wenn die Datei nicht existiert.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Die Datei {self.file_path} wurde nicht gefunden.")

        self.indicator_data = pd.read_csv(self.file_path, parse_dates=['Date'])
        print(f"Daten erfolgreich aus {self.file_path} geladen.")

    def preprocess_data(self, features):
        """
        Skaliert die angegebenen Features und fügt sie als Array zurück.

        Parameters:
        - features (list of str): Liste der Spaltennamen, die skaliert werden sollen.

        Returns:
        - np.ndarray: Das skalierte Feature-Array.
        """
        if self.indicator_data is None:
            raise ValueError("Indikatordaten sind nicht geladen. Bitte führen Sie 'load_indicator_data()' aus.")

        scaled_data = self.scaler.fit_transform(self.indicator_data[features])
        return scaled_data

    def get_sequences(self, data, seq_length):
        """
        Erstellt Sequenzen für das LSTM-Modell.

        Parameters:
        - data (np.ndarray): Das Array mit den Daten.
        - seq_length (int): Länge der Sequenzen.

        Returns:
        - tuple: Sequenzen (X) und Zielwerte (y).
        """
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length, 0])  # Verwende das erste Feature als Zielwert
        return np.array(sequences), np.array(labels)

    def prepare_for_lstm(self, seq_length=20):
        """
        Bereitet die Indikatordaten für das LSTM-Modell vor.

        Parameters:
        - seq_length (int): Länge der Sequenzen.

        Returns:
        - tuple: Trainingssequenzen (X), Zielwerte (y) und Skalierer (MinMaxScaler).
        """
        if self.indicator_data is None:
            raise ValueError("Indikatordaten sind nicht geladen. Bitte führen Sie 'load_indicator_data()' aus.")

        features = ['Close', 'Moving_Avg', 'Upper_Band', 'Lower_Band']
        scaled_data = self.preprocess_data(features)
        X, y = self.get_sequences(scaled_data, seq_length)
        return X, y, self.scaler

    def add_new_indicator(self, new_data, feature_name):
        """
        Fügt neue Indikatorwerte zur bestehenden Datenstruktur hinzu.

        Parameters:
        - new_data (pd.Series or np.ndarray): Neue Indikatordaten.
        - feature_name (str): Name der neuen Feature-Spalte.
        """
        if self.indicator_data is None:
            raise ValueError("Indikatordaten sind nicht geladen. Bitte führen Sie 'load_indicator_data()' aus.")

        self.indicator_data[feature_name] = new_data
        print(f"Neuer Indikator '{feature_name}' hinzugefügt.")

# Beispielnutzung
if __name__ == "__main__":
    controller = IndicatorController()

    # Lade die Indikatordaten
    controller.load_indicator_data()

    # Daten für das LSTM vorbereiten
    X, y, scaler = controller.prepare_for_lstm(seq_length=20)

    print(f"Trainingsdaten: {X.shape}, Zielwerte: {y.shape}")
