import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

class PriceIncreaseAnalyzer:
    """
    Eine Klasse zur Analyse der S&P 500 historischen Daten.
    Sie zählt die Gesamtanzahl der Datensätze ab dem ersten vollständigen Datensatz
    und berechnet, wie oft der Close-Kurs im Vergleich zum vorherigen Tag gestiegen ist.
    Zusätzlich werden aufeinanderfolgende Streaks von steigenden und fallenden Tagen analysiert.
    Die Analyse wird für Trainings- und Testdaten getrennt durchgeführt.
    """

    def __init__(self, subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv"):
        """
        Initialisiert die Analyzer-Klasse.

        Parameters:
        - subfolder (str): Der Name des Unterordners, in dem sich die CSV-Datei befinden könnte.
        - file_name (str): Der Name der CSV-Datei.
        """
        self.subfolder = subfolder
        self.file_name = file_name
        self.file_path = self.find_csv_file()
        self.df = None
        self.df_filtered = None
        self.df_excluded = None
        self.train_df = None
        self.test_df = None
        self.train_results = {}
        self.test_results = {}

    def find_csv_file(self):
        """
        Sucht rekursiv nach der CSV-Datei im gesamten Projektverzeichnis und allen Unterordnern
        innerhalb des "Unternehmenssoftware"-Verzeichnisses.

        Returns:
        - Path: Der Pfad zur CSV-Datei, wenn gefunden.
        - None: Wenn die Datei nicht gefunden wird.
        """
        current_path = Path(__file__).resolve()

        # Suche im "Unternehmenssoftware"-Verzeichnis
        while current_path.name != "Unternehmenssoftware":
            if current_path.parent == current_path:
                print("Das Verzeichnis 'Unternehmenssoftware' wurde nicht gefunden.")
                return None
            current_path = current_path.parent

        for root, _, files in os.walk(current_path):
            for file in files:
                if file == self.file_name:
                    return Path(root) / file

        print(f"Die Datei '{self.file_name}' wurde im 'Unternehmenssoftware'-Verzeichnis nicht gefunden.")
        return None

    def load_and_clean_data(self):
        """
        Lädt die CSV-Datei und filtert die unvollständigen Datensätze.
        Der erste vollständige Datensatz ist definiert als der erste Datensatz,
        bei dem sowohl 'High', 'Low' als auch 'Close' nicht 0.0 sind.
        Anschließend wird die Datenmenge in Trainings- und Testdaten aufgeteilt.
        """
        if self.file_path is None:
            print("Keine gültige Datei gefunden. Bitte überprüfen Sie den Dateipfad.")
            return

        try:
            # Einlesen der CSV-Datei
            self.df = pd.read_csv(self.file_path)

            # Überprüfen, ob die notwendigen Spalten vorhanden sind
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(column in self.df.columns for column in required_columns):
                raise ValueError(f"Die CSV-Datei muss die folgenden Spalten enthalten: {required_columns}")

            # Konvertieren der 'Date'-Spalte in datetime
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

            # Entfernen von Zeilen mit ungültigen Datumsangaben
            self.df = self.df.dropna(subset=['Date'])

            # Sortieren der Daten nach Datum, falls nicht bereits sortiert
            self.df.sort_values('Date', inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            start_date = pd.Timestamp('1980-01-01')
            self.df = self.df[self.df['Date'] >= start_date].copy()

            # Identifizieren des ersten vollständigen Datensatzes (High, Low und Close nicht 0.0)
            filter_condition = (self.df['High'] != 0.0) & (self.df['Low'] != 0.0) & (self.df['Close'] != 0.0)
            first_valid_index = self.df[filter_condition].index.min()

            if pd.isna(first_valid_index):
                raise ValueError("Keine vollständigen Datensätze gefunden (High, Low und Close sind nicht 0.0).")

            # Aufteilen in eingeschlossene und ausgeschlossene Datensätze
            self.df_excluded = self.df.loc[:first_valid_index - 1].copy()
            self.df_filtered = self.df.loc[first_valid_index:].copy()

            # Berechnung von 'Close_Increased'
            self.df_filtered['Previous_Close'] = self.df_filtered['Close'].shift(1)
            self.df_filtered['Close_Increased'] = self.df_filtered['Close'] > self.df_filtered['Previous_Close']

            # Aufteilen in Trainings- und Testdaten
            train_start_date = pd.Timestamp('1980-01-01')
            train_end_date = pd.Timestamp('2015-12-18')
            test_start_date = pd.Timestamp('2015-12-19')
            test_end_date = pd.Timestamp('2024-12-31')

            self.train_df = self.df_filtered[
                (self.df_filtered['Date'] >= train_start_date) &
                (self.df_filtered['Date'] <= train_end_date)
            ].copy()
            self.test_df = self.df_filtered[
                (self.df_filtered['Date'] >= test_start_date) &
                (self.df_filtered['Date'] <= test_end_date)
            ].copy()

            # Überprüfen, ob beide Datensätze ausreichend Daten enthalten
            if self.train_df.empty:
                raise ValueError("Trainingsdaten sind leer. Überprüfen Sie das Datumsformat und die Daten.")
            if self.test_df.empty:
                raise ValueError("Testdaten sind leer. Überprüfen Sie das Datumsformat und die Daten.")

        except pd.errors.EmptyDataError:
            print("Die CSV-Datei ist leer.")
        except pd.errors.ParserError:
            print("Es gab einen Fehler beim Parsen der CSV-Datei.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten beim Laden und Bereinigen der Daten: {e}")

    def analyze_data(self):
        """
        Analysiert die Trainings- und Testdaten, um die Gesamtanzahl der Datensätze,
        die Anzahl der Tage mit gestiegenem Close-Kurs und den Prozentsatz zu berechnen.
        """
        if self.train_df is None or self.test_df is None:
            print("Daten sind nicht geladen. Bitte führen Sie 'load_and_clean_data()' zuerst aus.")
            return

        try:
            # Analyse der Trainingsdaten
            total_train = len(self.train_df)
            increased_train = self.train_df['Close_Increased'].sum()
            decreased_train = (self.train_df['Close_Increased'] == False).sum()
            percentage_increased_train = (increased_train / total_train) * 100
            percentage_decreased_train = (decreased_train / total_train) * 100

            self.train_results = {
                'Total_Data_Points': total_train,
                'Increased_Close_Days': int(increased_train),
                'Decreased_Close_Days': int(decreased_train),
                'Percentage_Increased': round(percentage_increased_train, 2),
                'Percentage_Decreased': round(percentage_decreased_train, 2)
            }

            # Analyse der Testdaten
            total_test = len(self.test_df)
            increased_test = self.test_df['Close_Increased'].sum()
            decreased_test = (self.test_df['Close_Increased'] == False).sum()
            percentage_increased_test = (increased_test / total_test) * 100
            percentage_decreased_test = (decreased_test / total_test) * 100

            self.test_results = {
                'Total_Data_Points': total_test,
                'Increased_Close_Days': int(increased_test),
                'Decreased_Close_Days': int(decreased_test),
                'Percentage_Increased': round(percentage_increased_test, 2),
                'Percentage_Decreased': round(percentage_decreased_test, 2)
            }

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten bei der Datenanalyse: {e}")

    def print_results(self):
        """
        Gibt die Analyseergebnisse für Trainings- und Testdaten aus.
        """
        if not self.train_results or not self.test_results:
            print("Keine Ergebnisse vorhanden. Bitte führen Sie zuerst die Analyse durch.")
            return

        # Trainingsdaten
        train_start_date = self.train_df['Date'].iloc[0].date()
        train_end_date = self.train_df['Date'].iloc[-1].date()
        print("Analyse der S&P 500 historischen Daten - Trainingsdaten:")
        print(f"Zeitraum: {train_start_date} bis {train_end_date}")
        print(f"Alle Tage mit vollständigen Datensätzen: {self.train_results['Total_Data_Points']} Tage")
        print(f"Anzahl der Tage mit gestiegenem Close-Kurs: {self.train_results['Increased_Close_Days']}")
        print(f"Anzahl der Tage mit gefallenem Close-Kurs: {self.train_results['Decreased_Close_Days']}")
        print(f"Prozentsatz der Tage mit gestiegenem Close-Kurs: {self.train_results['Percentage_Increased']}%")
        print(f"Prozentsatz der Tage mit gefallenem Close-Kurs: {self.train_results['Percentage_Decreased']}%\n")

        # Testdaten
        test_start_date = self.test_df['Date'].iloc[0].date()
        test_end_date = self.test_df['Date'].iloc[-1].date()
        print("Analyse der S&P 500 historischen Daten - Testdaten:")
        print(f"Zeitraum: {test_start_date} bis {test_end_date}")
        print(f"Alle Tage mit vollständigen Datensätzen: {self.test_results['Total_Data_Points']} Tage")
        print(f"Anzahl der Tage mit gestiegenem Close-Kurs: {self.test_results['Increased_Close_Days']}")
        print(f"Anzahl der Tage mit gefallenem Close-Kurs: {self.test_results['Decreased_Close_Days']}")
        print(f"Prozentsatz der Tage mit gestiegenem Close-Kurs: {self.test_results['Percentage_Increased']}%")
        print(f"Prozentsatz der Tage mit gefallenem Close-Kurs: {self.test_results['Percentage_Decreased']}%")

    def plot_movement_distribution(self):
        """
        Erstellt vier separate Balkendiagramme der Verteilung von steigenden und fallenden Tagen
        für Trainings- und Testdaten in absoluter und relativer Häufigkeit.
        """
        if not self.train_results or not self.test_results:
            print("Keine Ergebnisse vorhanden. Bitte führen Sie zuerst die Analyse durch.")
            return

        # Daten für Trainingsdaten – Absolute Häufigkeit
        train_labels = ['Gestiegen', 'Gefallen']
        train_counts = [self.train_results['Increased_Close_Days'], self.train_results['Decreased_Close_Days']]
        train_colors = ['green', 'red']

        # Daten für Testdaten – Absolute Häufigkeit
        test_labels = ['Gestiegen', 'Gefallen']
        test_counts = [self.test_results['Increased_Close_Days'], self.test_results['Decreased_Close_Days']]
        test_colors = ['green', 'red']

        # Daten für Trainingsdaten – Relative Häufigkeit
        train_percentages = [self.train_results['Percentage_Increased'], self.train_results['Percentage_Decreased']]

        # Daten für Testdaten – Relative Häufigkeit
        test_percentages = [self.test_results['Percentage_Increased'], self.test_results['Percentage_Decreased']]

        # Funktion zum Erstellen und Speichern der Diagramme
        def create_and_save_plot(labels, counts, colors, title, ylabel, filename, is_percentage=False):
            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, counts, color=colors)
            plt.title(title)
            plt.ylabel(ylabel)
            if is_percentage:
                plt.ylim(0, 100)
            else:
                # Einheitliche Y-Achse basierend auf den maximalen Werten
                plt.ylim(0, max(train_counts + test_counts) * 1.1)

            # Werte über den Balken anzeigen
            for bar in bars:
                height = bar.get_height()
                if is_percentage:
                    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}%', ha='center', va='bottom', fontsize=10)
                else:
                    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.savefig(filename)
            plt.show()

        # Erstellen und Speichern der Trainingsdaten – Absolute Häufigkeit
        create_and_save_plot(
            labels=train_labels,
            counts=train_counts,
            colors=train_colors,
            title='Verteilung der Kursbewegungen (Trainingsdaten) - Absolute Häufigkeit',
            ylabel='Anzahl der Tage',
            filename='train_movement_absolute.png',
            is_percentage=False
        )

        # Erstellen und Speichern der Testdaten – Absolute Häufigkeit
        create_and_save_plot(
            labels=test_labels,
            counts=test_counts,
            colors=test_colors,
            title='Verteilung der Kursbewegungen (Testdaten) - Absolute Häufigkeit',
            ylabel='Anzahl der Tage',
            filename='test_movement_absolute.png',
            is_percentage=False
        )

        # Erstellen und Speichern der Trainingsdaten – Relative Häufigkeit
        create_and_save_plot(
            labels=train_labels,
            counts=train_percentages,
            colors=train_colors,
            title='Verteilung der Kursbewegungen (Trainingsdaten) - Relative Häufigkeit',
            ylabel='Prozentsatz der Tage (%)',
            filename='train_movement_relative.png',
            is_percentage=True
        )

        # Erstellen und Speichern der Testdaten – Relative Häufigkeit
        create_and_save_plot(
            labels=test_labels,
            counts=test_percentages,
            colors=test_colors,
            title='Verteilung der Kursbewegungen (Testdaten) - Relative Häufigkeit',
            ylabel='Prozentsatz der Tage (%)',
            filename='test_movement_relative.png',
            is_percentage=True
        )

# Beispielverwendung
def main():
    analyzer = PriceIncreaseAnalyzer(subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv")
    analyzer.load_and_clean_data()
    analyzer.analyze_data()
    analyzer.print_results()
    analyzer.plot_movement_distribution()

if __name__ == "__main__":
    main()
