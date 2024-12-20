import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

class PriceIncreaseAnalyzer:
    """
    Eine Klasse zur Analyse der S&P 500 historischen Daten.
    Sie zählt die Gesamtanzahl der Datensätze ab dem ersten vollständigen Datensatz
    und berechnet, wie oft der Open-Kurs im Vergleich zum vorherigen Tag gestiegen ist.
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
        self.results = {}

    def find_csv_file(self):
        """
        Durchsucht die Verzeichnisse nach dem angegebenen Unterordner und der CSV-Datei.

        Returns:
        - Path: Der Pfad zur CSV-Datei, wenn gefunden.
        - None: Wenn die Datei nicht gefunden wird.
        """
        current_path = Path(__file__).parent.resolve()
        root = Path(current_path.root)

        while current_path != root:
            potential_folder = current_path / self.subfolder
            potential_file = potential_folder / self.file_name
            if potential_file.is_file():
                return potential_file
            current_path = current_path.parent

        print(f"Die Datei '{self.file_name}' im Unterordner '{self.subfolder}' wurde nicht gefunden.")
        return None

    def load_and_clean_data(self):
        """
        Lädt die CSV-Datei und filtert die unvollständigen Datensätze.
        Der erste vollständige Datensatz ist definiert als der erste Datensatz,
        bei dem sowohl 'High', 'Low' als auch 'Open' nicht 0.0 sind.
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

            # Identifizieren des ersten vollständigen Datensatzes (High, Low und Open nicht 0.0)
            filter_condition = (self.df['High'] != 0.0) & (self.df['Low'] != 0.0) & (self.df['Open'] != 0.0)
            first_valid_index = self.df[filter_condition].index.min()

            if pd.isna(first_valid_index):
                raise ValueError("Keine vollständigen Datensätze gefunden (High, Low und Open sind nicht 0.0).")

            # Aufteilen in eingeschlossene und ausgeschlossene Datensätze
            self.df_excluded = self.df.loc[:first_valid_index - 1].copy()
            self.df_filtered = self.df.loc[first_valid_index:].copy()

            # Konvertieren der 'Date'-Spalte in datetime für beide DataFrames
            self.df_filtered['Date'] = pd.to_datetime(self.df_filtered['Date'])
            self.df_excluded['Date'] = pd.to_datetime(self.df_excluded['Date'])

            # Sortieren der Daten nach Datum, falls nicht bereits sortiert
            self.df_filtered.sort_values('Date', inplace=True)
            self.df_filtered.reset_index(drop=True, inplace=True)

            self.df_excluded.sort_values('Date', inplace=True)
            self.df_excluded.reset_index(drop=True, inplace=True)

        except pd.errors.EmptyDataError:
            print("Die CSV-Datei ist leer.")
        except pd.errors.ParserError:
            print("Es gab einen Fehler beim Parsen der CSV-Datei.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten beim Laden und Bereinigen der Daten: {e}")

    def analyze_data(self, compare_column='Open'):
        """
        Analysiert die Daten, um die Gesamtanzahl der Datensätze,
        die Anzahl der Tage mit gestiegenem Open-Kurs und den Prozentsatz zu berechnen.

        Parameters:
        - compare_column (str): Die Spalte, die zum Vergleich verwendet wird ('Open' oder 'Close').
        """
        if self.df_filtered is None:
            print("Daten sind nicht geladen. Bitte führen Sie 'load_and_clean_data()' zuerst aus.")
            return

        try:
            # Gesamtanzahl der gültigen Datensätze
            total_records = len(self.df_filtered)

            if total_records < 2:
                raise ValueError("Nicht genügend Datenpunkte nach der Filterung vorhanden, um Vergleiche anzustellen.")

            # Vergleich der Open-Kurse
            self.df_filtered['Previous_Open'] = self.df_filtered[compare_column].shift(1)
            self.df_filtered['Open_Increased'] = self.df_filtered[compare_column] > self.df_filtered['Previous_Open']

            # Anzahl der Tage mit gestiegenem Open-Wert
            increased_days = self.df_filtered['Open_Increased'].sum()

            # Berechnung des Prozentsatzes
            percentage_increased = (increased_days / (total_records - 1)) * 100  # -1 wegen shift

            # Ergebnisse zusammenfassen
            self.results = {
                'Total_Data_Points': total_records,
                'Increased_Open_Days': int(increased_days),
                'Percentage_Increased': round(percentage_increased, 2)
            }

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten bei der Datenanalyse: {e}")

    def get_results(self):
        """
        Gibt die Analyseergebnisse zurück.

        Returns:
        - dict: Ein Dictionary mit den Analyseergebnissen.
        """
        return self.results

    def print_results(self):
        """
        Druckt die Analyseergebnisse in einer lesbaren Form.
        """
        if not self.results:
            print("Keine Ergebnisse vorhanden. Bitte führen Sie zuerst die Analyse durch.")
            return

        start_date = self.df_filtered['Date'].iloc[0].date()
        print("Analyse der S&P 500 historischen Daten:")
        print(f"Alle Tage mit vollständigen Datensätzen ab dem {start_date}: {self.results['Total_Data_Points']} Tage")
        print(f"Anzahl der Tage mit gestiegenem Open-Kurs: {self.results['Increased_Open_Days']}")
        print(f"Prozentsatz der Tage mit gestiegenem Open-Kurs: {self.results['Percentage_Increased']}%")

    def plot_data_quality(self):
        """
        Plottet ein Balkendiagramm, das die Anzahl der ausgeschlossenen und eingeschlossenen Datenpunkte zeigt.
        """
        labels = ['Ausgeschlossene Datenpunkte', 'Eingeschlossene Datenpunkte']
        counts = [len(self.df_excluded), len(self.df_filtered)]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color=['red', 'green'])
        plt.title('Datenqualitätsfilterung')
        plt.ylabel('Anzahl der Datenpunkte')

        # Werte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_open_price_movement(self):
        """
        Plottet ein Balkendiagramm, das die Anzahl der Tage mit gestiegenem und gefallenem Open-Kurs zeigt.
        """
        increased = self.results['Increased_Open_Days']
        decreased = self.results['Total_Data_Points'] - increased

        labels = ['Gestiegener Open-Kurs', 'Gesunkener Open-Kurs']
        counts = [increased, decreased]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color=['blue', 'orange'])
        plt.title('Vergleich der Open-Kursbewegungen')
        plt.ylabel('Anzahl der Tage')

        # Werte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_yearly_increases(self):
        """
        Plottet ein Balkendiagramm der jährlichen Anzahl der Tage, an denen der Open-Kurs gestiegen oder gesunken ist.
        """
        # Hinzufügen einer Jahres-Spalte
        self.df_filtered['Year'] = self.df_filtered['Date'].dt.year

        # Gruppieren nach Jahr und Bewegung
        yearly_movement = self.df_filtered.groupby(['Year', 'Open_Increased']).size().unstack(fill_value=0)

        # Plot
        yearly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
        plt.title('Jährliche Anzahl der Tage mit gestiegenem und gesunkenem Open-Kurs')
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl der Tage')
        plt.legend(['Gesunken', 'Gestiegen'])
        plt.tight_layout()
        plt.show()

    def plot_monthly_increases(self):
        """
        Plottet ein Balkendiagramm der monatlichen Anzahl der Monate,
        in denen der Open-Kurs am ersten Tag des Monats gestiegen oder gefallen ist.
        """
        # Sortieren der Daten nach Datum
        self.df_filtered.sort_values('Date', inplace=True)

        # Extrahieren des ersten Tages jedes Monats
        first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)

        # Vergleich des Open-Kurses mit dem vorherigen Monat
        first_days['Previous_Open'] = first_days['Open'].shift(1)
        first_days['Monthly_Increased'] = first_days['Open'] > first_days['Previous_Open']

        # Entfernen des ersten Monats, da kein vorheriger Monat vorhanden ist
        first_days = first_days.dropna(subset=['Previous_Open'])

        # Anzahl der monatlichen Steigerungen und Senkungen
        monthly_increased = first_days['Monthly_Increased'].sum()
        monthly_decreased = len(first_days) - monthly_increased
        percentage_increased = (monthly_increased / len(first_days)) * 100

        # Plot des Balkendiagramms (statt des Tortendiagramms)
        plt.figure(figsize=(8, 6))
        labels = ['Gesunkener Open-Kurs', 'Gestiegener Open-Kurs']
        counts = [monthly_decreased, monthly_increased]
        plt.bar(labels, counts, color=['orange', 'green'])
        plt.title('Monatliche Open-Kursbewegungen')
        plt.ylabel('Anzahl der Monate')

        # Werte über den Balken anzeigen
        for i, count in enumerate(counts):
            plt.text(i, count, f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Plot der prozentualen Verteilung als Balkendiagramm (anstatt eines Tortendiagramms)
        plt.figure(figsize=(8, 6))
        plt.bar(labels, [100 - percentage_increased, percentage_increased], color=['orange', 'green'])
        plt.title('Prozentuale Verteilung der monatlichen Open-Kursbewegungen')
        plt.ylabel('Prozentuale Verteilung (%)')

        # Prozentwerte über den Balken anzeigen
        for i, percentage in enumerate([100 - percentage_increased, percentage_increased]):
            plt.text(i, [100 - percentage_increased, percentage_increased][i], f'{percentage:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_monthly_increases_yearly(self):
        """
        Plottet ein Balkendiagramm der jährlichen Anzahl der Monate, in denen der Open-Kurs gestiegen oder gefallen ist.
        """
        # Sortieren der Daten nach Datum
        self.df_filtered.sort_values('Date', inplace=True)

        # Extrahieren des ersten Tages jedes Monats
        first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)

        # Vergleich des Open-Kurses mit dem vorherigen Monat
        first_days['Previous_Open'] = first_days['Open'].shift(1)
        first_days['Monthly_Increased'] = first_days['Open'] > first_days['Previous_Open']

        # Entfernen des ersten Monats, da kein vorheriger Monat vorhanden ist
        first_days = first_days.dropna(subset=['Previous_Open'])

        # Hinzufügen einer Jahres-Spalte
        first_days['Year'] = first_days['Date'].dt.year

        # Gruppieren nach Jahr und Bewegung
        yearly_monthly_movement = first_days.groupby(['Year', 'Monthly_Increased']).size().unstack(fill_value=0)

        # Plot
        yearly_monthly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
        plt.title('Jährliche Anzahl der Monate mit gestiegenem und gesunkenem Open-Kurs')
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl der Monate')
        plt.legend(['Gesunken', 'Gestiegen'])
        plt.tight_layout()
        plt.show()

def main():
    """
    Hauptfunktion, die den Analyseprozess steuert.
    """
    # Initialisieren der Analyzer-Klasse
    analyzer = PriceIncreaseAnalyzer()

    # Überprüfen, ob die CSV-Datei existiert und laden der Daten
    analyzer.load_and_clean_data()

    # Analysieren der Daten (Standard: Vergleich der 'Open'-Spalte)
    analyzer.analyze_data(compare_column='Open')

    # Ausgabe der Ergebnisse
    analyzer.print_results()

    # Plot der Datenqualitätsfilterung
    analyzer.plot_data_quality()

    # Plot der Open-Kursbewegungen
    analyzer.plot_open_price_movement()

    # Plot der jährlichen Kursbewegungen
    analyzer.plot_yearly_increases()

    # Plot der jährlichen monatlichen Kursbewegungen
    analyzer.plot_monthly_increases_yearly()

    # Plot der monatlichen Kursbewegungen
    analyzer.plot_monthly_increases()

if __name__ == "__main__":
    main()
