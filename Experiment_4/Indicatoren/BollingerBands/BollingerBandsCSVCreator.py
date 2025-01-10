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
# Klasse, die sich um die Erstellung der im Konzept beschriebenen CSV-Dateien kümmert
# -------------------------------------------------------------------------
class BollingerBandsCSVCreator:
    """
    Diese Klasse kümmert sich ausschließlich um die Erstellung der im Konzept
    beschriebenen CSV-Dateien. Sie geht davon aus, dass bollinger_df und historical_df
    bereits eingelesen und als DataFrames vorliegen.

    WICHTIG:
    - Die eigentliche Logik zur Identifikation von Berührungen, Berechnung von Toleranzen,
      Kennzeichnung von Event-Typen (Upper_Touch, Lower_Touch etc.) oder der Future_Returns
      wird hier noch NICHT implementiert. Dies folgt erst im nächsten Schritt!
    - Hier werden nur die CSV-Dateien im gewünschten Format und mit den benötigten Spalten
      vorbereitet und geschrieben (Platzhalter).
    - Neu: 'Consecutive_Touches' als Spalte hinzugefügt. State Machine (2 Tage Toleranz)
      wird final in BollingerBandsFiller.py abgebildet.
    """

    def __init__(self,
                 bollinger_df: pd.DataFrame,
                 historical_df: pd.DataFrame,
                 output_directory: str = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\a_project_one\Indicatoren\BollingerBands\analysis_results"):
        """
        :param bollinger_df: DataFrame mit Bollinger-Band-Daten
        :param historical_df: DataFrame mit den historischen S&P 500 Daten
        :param output_directory: Zielordner, in dem die CSV-Dateien abgelegt werden sollen
        """
        self.bollinger_df = bollinger_df.copy()
        self.historical_df = historical_df.copy()
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Optionales Mergen beider DataFrames auf 'Date', falls notwendig
        self.merged_df = pd.merge(
            self.historical_df,
            self.bollinger_df,
            on='Date',
            how='inner'
        )

        # Datum sortieren (optional)
        self.merged_df.sort_values(by='Date', inplace=True)

    def create_separate_csv_files(self, band_types=None, tolerances=None):
        """
        Erstellt für jede Kombination aus Bandart (Upper, Lower, Middle) und Toleranzstufe
        (z.B. 0.25%, 0.375%, 0.5%, 0.75%, 1%) eine eigene CSV-Datei.

        Aktuell nur Platzhalter-Spalten. Die tatsächliche Logik zur Befüllung erfolgt später.
        """
        if band_types is None:
            band_types = ["Upper_Touch", "Lower_Touch", "Middle_Break_Up", "Middle_Break_Down"]
        if tolerances is None:
            tolerances = ["0.25%", "0.375%", "0.5%", "0.75%", "1%"]

        for band_type in band_types:
            for tol in tolerances:
                placeholder_df = pd.DataFrame({
                    "Date": self.merged_df["Date"],
                    "Event_Type": [band_type] * len(self.merged_df),
                    "Close_Price": self.merged_df.get("Close_x", self.merged_df.get("Close", None)),
                    "Distance_Upper": None,      # Wird später berechnet
                    "Distance_Lower": None,      # Wird später berechnet
                    "Distance_Middle": None,     # Wird später berechnet
                    "Consecutive_Touches": None, # Neu: Zählt aufeinanderfolgende Berührungen (Placeholder)
                    "Touch_Type": None,          # Real/Apparent, später berechnet
                    "Future_Return_1": None,     # Dezimalwert, später berechnet
                    "Future_Return_5": None,     # Dezimalwert, später berechnet
                    "Future_Move_1": None,       # Up oder Down
                    "Future_Move_5": None        # Up oder Down
                })

                file_name = f"{band_type}_{tol}.csv"
                file_path = self.output_directory / file_name
                placeholder_df.to_csv(file_path, index=False)

    def create_combined_csv_file(self, tolerances=None):
        """
        Erstellt eine kombinierte CSV-Datei, die alle Events (Upper, Lower, Middle)
        und alle Toleranzstufen enthält. Spalte 'Tolerance' wird hinzugefügt.
        """
        if tolerances is None:
            tolerances = ["0.25%", "0.375%", "0.5%", "0.75%", "1%"]

        event_types = ["Upper_Touch", "Lower_Touch", "Middle_Break_Up", "Middle_Break_Down"]
        combined_dfs = []

        for event_type in event_types:
            for tol in tolerances:
                placeholder_df = pd.DataFrame({
                    "Date": self.merged_df["Date"],
                    "Event_Type": [event_type] * len(self.merged_df),
                    "Tolerance": [tol] * len(self.merged_df),
                    "Close_Price": self.merged_df.get("Close_x", self.merged_df.get("Close", None)),
                    "Distance_Upper": None,
                    "Distance_Lower": None,
                    "Distance_Middle": None,
                    "Consecutive_Touches": None, # Neu: Placeholder
                    "Touch_Type": None,
                    "Future_Return_1": None,
                    "Future_Return_5": None,
                    "Future_Move_1": None,
                    "Future_Move_5": None
                })
                combined_dfs.append(placeholder_df)

        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_df.sort_values(by=["Date", "Event_Type", "Tolerance"], inplace=True)

        file_path = self.output_directory / "Touches_Combined.csv"
        combined_df.to_csv(file_path, index=False)

    def create_distance_csv(self):
        """
        Erstellt eine CSV mit den prozentualen Abständen zu den Bollinger Bändern:
        Spalten: Date, Upper_Distance, Middle_Distance, Lower_Distance
        """
        distance_df = pd.DataFrame({
            "Date": self.merged_df["Date"],
            "Upper_Distance": None,
            "Middle_Distance": None,
            "Lower_Distance": None
        })

        file_path = self.output_directory / "Distance_Data.csv"
        distance_df.to_csv(file_path, index=False)

    def run_all_csv_creations(self):
        """
        Führt die Erstellung aller CSV-Dateien aus.
        """
        self.create_separate_csv_files()
        self.create_combined_csv_file()
        self.create_distance_csv()


# -------------------------------------------------------------------------
# Hauptteil: CSV-Dateien laden und Klasse aufrufen
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV-Dateien aus dem Projekt laden
    bollinger_df = load_data("bollinger_bands_2.csv")
    historical_df = load_data("SP500_Index_Historical_Data.csv")

    # Instanz der Klasse erstellen
    csv_creator = BollingerBandsCSVCreator(
        bollinger_df=bollinger_df,
        historical_df=historical_df,
        output_directory=r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\a_project_one\Indicatoren\BollingerBands\analysis_results"
    )

    # CSV-Erstellung anstoßen
    csv_creator.run_all_csv_creations()

    print("CSV-Dateien wurden erfolgreich erstellt.")
