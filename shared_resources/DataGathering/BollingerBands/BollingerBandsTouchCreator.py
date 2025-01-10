import os
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------------
# Hilfsfunktionen zum Auffinden und Laden der CSV-Dateien
# -------------------------------------------------------------------------
def find_csv_file(file_name):
    """
    Durchsucht das Projektverzeichnis (drei Ebenen über diesem Skript) und alle Unterverzeichnisse
    nach einer Datei mit dem Namen file_name. Gibt den Pfad zurück, wenn gefunden.
    """
    project_dir = Path(__file__).resolve().parents[3]  # Geht drei Ebenen hoch – ggf. anpassen
    for file_path in project_dir.rglob(file_name):
        if file_path.is_file():
            return file_path
    print(f"Datei '{file_name}' wurde im Projektverzeichnis nicht gefunden.")
    return None


def load_data(file_name):
    """
    Lädt die CSV-Datei mit parse_dates=['Date'] und gibt ein pandas DataFrame zurück.
    """
    csv_path = find_csv_file(file_name)
    if csv_path:
        return pd.read_csv(csv_path, parse_dates=['Date'])
    else:
        raise FileNotFoundError(f"Datei '{file_name}' nicht gefunden.")


# -------------------------------------------------------------------------
# Klasse zur Erstellung der Bollinger_bands_touches_2.csv
# -------------------------------------------------------------------------
class BollingerBandsTouchCreator:
    """
    Diese Klasse erstellt die 'Bollinger_bands_touches_2.csv' Datei, die für jedes Datum
    die Anzahl der aufeinanderfolgenden Berührungen der oberen, mittleren und
    unteren Bollinger-Bänder enthält.
    """

    def __init__(self,
                 bollinger_df: pd.DataFrame,
                 historical_df: pd.DataFrame,
                 output_directory: str = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\BollingerBands"):
        """
        :param bollinger_df: DataFrame mit Bollinger-Band-Daten (Moving_Avg, Upper_Band, Lower_Band)
        :param historical_df: DataFrame mit den historischen S&P 500 Daten (Date, High, Low)
        :param output_directory: Zielordner, in dem die CSV-Datei abgelegt werden soll
        """
        self.bollinger_df = bollinger_df.copy()
        self.historical_df = historical_df.copy()
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Merge der DataFrames auf 'Date'
        self.merged_df = pd.merge(
            self.historical_df[['Date', 'High', 'Low']],
            self.bollinger_df[['Date', 'Moving_Avg', 'Upper_Band', 'Lower_Band']],
            on='Date',
            how='inner'
        )

        # Datum sortieren
        self.merged_df.sort_values(by='Date', inplace=True)
        self.merged_df.reset_index(drop=True, inplace=True)

    def create_touches_csv(self, output_file: str = "Bollinger_bands_touches_2.csv"):
        """
        Erstellt die CSV-Datei 'Bollinger_bands_touches_2.csv', die für jedes Datum
        die Anzahl der aufeinanderfolgenden Berührungen der oberen, mittleren und
        unteren Bollinger-Bänder enthält.

        :param output_file: Name der Ausgabedatei
        """
        df = self.merged_df.copy()

        # --------------------------------------------------------
        # Bedingungen für Berührungen definieren:
        # --------------------------------------------------------
        # 1) Upper_Band:
        #    - Entweder Upper_Band zwischen Low und High
        #    - Oder der gesamte Tagesbereich (Low und High) liegt oberhalb des Upper_Band
        df['Upper_Touch'] = df.apply(
            lambda row: (
                pd.notnull(row['Upper_Band']) and
                (
                    (row['Low'] <= row['Upper_Band'] <= row['High']) or
                    (row['Low'] > row['Upper_Band'])
                )
            ),
            axis=1
        )

        # 2) Middle_Band:
        #    - NUR wenn Moving_Avg wirklich zwischen Low und High liegt
        df['Middle_Touch'] = df.apply(
            lambda row: (
                pd.notnull(row['Moving_Avg']) and
                (row['Low'] <= row['Moving_Avg'] <= row['High'])
            ),
            axis=1
        )

        # 3) Lower_Band:
        #    - Entweder Lower_Band zwischen Low und High
        #    - Oder der gesamte Tagesbereich (Low und High) liegt unterhalb des Lower_Band
        df['Lower_Touch'] = df.apply(
            lambda row: (
                pd.notnull(row['Lower_Band']) and
                (
                    (row['Low'] <= row['Lower_Band'] <= row['High']) or
                    (row['High'] < row['Lower_Band'])
                )
            ),
            axis=1
        )

        # Funktion zur Berechnung der aufeinanderfolgenden Berührungen
        def calculate_consecutive_touches(touch_series):
            counts = []
            count = 0
            for touch in touch_series:
                if touch:
                    count += 1
                else:
                    count = 0
                counts.append(count)
            return counts

        # Aufeinanderfolgende Berührungen für jedes Band berechnen
        df['Upper_consecutive_touches'] = calculate_consecutive_touches(df['Upper_Touch'])
        df['Middle_consecutive_touches'] = calculate_consecutive_touches(df['Middle_Touch'])
        df['Lower_consecutive_touches'] = calculate_consecutive_touches(df['Lower_Touch'])

        # Auswahl der relevanten Spalten
        touches_df = df[['Date',
                         'Upper_consecutive_touches',
                         'Middle_consecutive_touches',
                         'Lower_consecutive_touches']].copy()

        # Speichern der CSV
        file_path = self.output_directory / output_file
        touches_df.to_csv(file_path, index=False)

        print(f"'{output_file}' wurde erfolgreich erstellt unter: {file_path}")


# -------------------------------------------------------------------------
# Hauptteil: CSV-Dateien laden und Klasse aufrufen
# -------------------------------------------------------------------------
def main():
    # CSV-Dateien aus dem Projekt laden
    bollinger_df = load_data("bollinger_bands_2.csv")
    historical_df = load_data("SP500_Index_Historical_Data.csv")

    # Instanz der Klasse erstellen
    touch_creator = BollingerBandsTouchCreator(
        bollinger_df=bollinger_df,
        historical_df=historical_df,
        output_directory=r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\BollingerBands"
    )

    # CSV-Erstellung anstoßen
    touch_creator.create_touches_csv()

    print("Bollinger_bands_touches_2.csv wurde erfolgreich erstellt.")


if __name__ == "__main__":
    main()
