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
        self.streak_results = {}

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
        bei dem sowohl 'High', 'Low' als auch 'Close' nicht 0.0 sind.
        (Angepasst auf Close anstatt Open)
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

            # Identifizieren des ersten vollständigen Datensatzes (High, Low und Close nicht 0.0)
            filter_condition = (self.df['High'] != 0.0) & (self.df['Low'] != 0.0) & (self.df['Close'] != 0.0)
            first_valid_index = self.df[filter_condition].index.min()

            if pd.isna(first_valid_index):
                raise ValueError("Keine vollständigen Datensätze gefunden (High, Low und Close sind nicht 0.0).")

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

            # Berechnung von 'Close_Increased'
            self.df_filtered['Previous_Close'] = self.df_filtered['Close'].shift(1)
            self.df_filtered['Close_Increased'] = self.df_filtered['Close'] > self.df_filtered['Previous_Close']

            # Berechnung der Streaks
            self.calculate_streaks()

        except pd.errors.EmptyDataError:
            print("Die CSV-Datei ist leer.")
        except pd.errors.ParserError:
            print("Es gab einen Fehler beim Parsen der CSV-Datei.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten beim Laden und Bereinigen der Daten: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def load_and_clean_data(self):
    #     """
    #     Lädt die CSV-Datei und filtert die unvollständigen Datensätze.
    #     Der erste vollständige Datensatz ist definiert als der erste Datensatz,
    #     bei dem sowohl 'High', 'Low' als auch 'Open' nicht 0.0 sind.
    #     """
    #     if self.file_path is None:
    #         print("Keine gültige Datei gefunden. Bitte überprüfen Sie den Dateipfad.")
    #         return
    #
    #     try:
    #         # Einlesen der CSV-Datei
    #         self.df = pd.read_csv(self.file_path)
    #
    #         # Überprüfen, ob die notwendigen Spalten vorhanden sind
    #         required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    #         if not all(column in self.df.columns for column in required_columns):
    #             raise ValueError(f"Die CSV-Datei muss die folgenden Spalten enthalten: {required_columns}")
    #
    #         # Identifizieren des ersten vollständigen Datensatzes (High, Low und Open nicht 0.0)
    #         filter_condition = (self.df['High'] != 0.0) & (self.df['Low'] != 0.0) & (self.df['Open'] != 0.0)
    #         first_valid_index = self.df[filter_condition].index.min()
    #
    #         if pd.isna(first_valid_index):
    #             raise ValueError("Keine vollständigen Datensätze gefunden (High, Low und Open sind nicht 0.0).")
    #
    #         # Aufteilen in eingeschlossene und ausgeschlossene Datensätze
    #         self.df_excluded = self.df.loc[:first_valid_index - 1].copy()
    #         self.df_filtered = self.df.loc[first_valid_index:].copy()
    #
    #         # Konvertieren der 'Date'-Spalte in datetime für beide DataFrames
    #         self.df_filtered['Date'] = pd.to_datetime(self.df_filtered['Date'])
    #         self.df_excluded['Date'] = pd.to_datetime(self.df_excluded['Date'])
    #
    #         # Sortieren der Daten nach Datum, falls nicht bereits sortiert
    #         self.df_filtered.sort_values('Date', inplace=True)
    #         self.df_filtered.reset_index(drop=True, inplace=True)
    #
    #         self.df_excluded.sort_values('Date', inplace=True)
    #         self.df_excluded.reset_index(drop=True, inplace=True)
    #
    #         # Berechnung von 'Open_Increased'
    #         self.df_filtered['Previous_Open'] = self.df_filtered['Open'].shift(1)
    #         self.df_filtered['Open_Increased'] = self.df_filtered['Open'] > self.df_filtered['Previous_Open']
    #
    #         # Berechnung der Streaks
    #         self.calculate_streaks()
    #
    #     except pd.errors.EmptyDataError:
    #         print("Die CSV-Datei ist leer.")
    #     except pd.errors.ParserError:
    #         print("Es gab einen Fehler beim Parsen der CSV-Datei.")
    #     except Exception as e:
    #         print(f"Ein Fehler ist aufgetreten beim Laden und Bereinigen der Daten: {e}")

    def calculate_streaks(self, max_increasing=13, max_decreasing=10):
        """
        Berechnet die Anzahl und den prozentualen Anteil der aufeinanderfolgenden steigenden und fallenden Tage
        basierend auf dem 'Close_Increased'-Wert.
        (Angepasst auf Close_Increased anstatt Open_Increased)
        """
        # Initialisieren der Streak-Listen
        streaks = []
        current_streak_length = 1
        current_streak_type = None  # 'increasing' oder 'decreasing'

        # Iteriere über die Close_Increased-Spalte
        for idx, row in self.df_filtered.iterrows():
            if pd.isna(row['Close_Increased']):
                continue  # Überspringe den ersten Datensatz ohne vorherigen Wert

            movement = 'increasing' if row['Close_Increased'] else 'decreasing'

            if current_streak_type is None:
                # Beginn der ersten Streak
                current_streak_type = movement
                current_streak_length = 1
            elif movement == current_streak_type:
                # Fortsetzung der aktuellen Streak
                current_streak_length += 1
            else:
                # Abschluss der aktuellen Streak und Start einer neuen
                limit = max_increasing if current_streak_type == 'increasing' else max_decreasing
                streaks.append((current_streak_type, min(current_streak_length, limit)))
                current_streak_type = movement
                current_streak_length = 1

        # Füge die letzte Streak hinzu
        if current_streak_type is not None:
            limit = max_increasing if current_streak_type == 'increasing' else max_decreasing
            streaks.append((current_streak_type, min(current_streak_length, limit)))

        # Separate Listen für steigende und fallende Streaks
        increasing_streaks = [length for stype, length in streaks if stype == 'increasing']
        decreasing_streaks = [length for stype, length in streaks if stype == 'decreasing']

        # Zählen der Streaks
        increasing_counts = pd.Series(increasing_streaks).value_counts().sort_index()
        decreasing_counts = pd.Series(decreasing_streaks).value_counts().sort_index()

        # Begrenzen auf maximale Streak-Länge
        increasing_counts = increasing_counts.reindex(range(1, max_increasing + 1), fill_value=0)
        decreasing_counts = decreasing_counts.reindex(range(1, max_decreasing + 1), fill_value=0)

        # Gesamtanzahl der Streaks
        total_streaks = len(increasing_streaks) + len(decreasing_streaks)

        # Gesamtanzahl der Tage in den Streaks
        total_streak_days = sum(increasing_streaks) + sum(decreasing_streaks)

        # Gesamtanzahl der analysierten Tage
        total_days = len(self.df_filtered) - 1  # -1 wegen des verschobenen vorherigen Werts

        # Überprüfung
        print(f"Überprüfung: Gesamtanzahl der Tage in Streaks = {total_streak_days}, Gesamtanzahl der analysierten Tage = {total_days}")
        if total_streak_days != total_days:
            print("Warnung: Die Summe der Streak-Tage stimmt nicht mit der Gesamtanzahl der analysierten Tage überein.")
        else:
            print("Überprüfung erfolgreich: Die Summe der Streak-Tage stimmt mit der Gesamtanzahl der analysierten Tage überein.")

        # Berechnung der prozentualen Anteile
        if total_streaks > 0:
            increasing_percentages = (increasing_counts / total_streaks * 100).round(2)
            decreasing_percentages = (decreasing_counts / total_streaks * 100).round(2)
        else:
            increasing_percentages = increasing_counts
            decreasing_percentages = decreasing_counts

        # Ergebnisse speichern
        self.streak_results['increasing'] = {
            'counts': increasing_counts,
            'percentages': increasing_percentages
        }
        self.streak_results['decreasing'] = {
            'counts': decreasing_counts,
            'percentages': decreasing_percentages
        }
        self.streak_results['total_streaks'] = total_streaks
        self.streak_results['total_streak_days'] = total_streak_days

    # Originalfassung (auskommentiert, basiert auf Open_Increased):
    # def calculate_streaks(self, max_increasing=13, max_decreasing=10):
    #     """
    #     Berechnet die Anzahl und den prozentualen Anteil der aufeinanderfolgenden steigenden und fallenden Tage.
    #     """
    #     # Initialisieren der Streak-Listen
    #     streaks = []
    #     current_streak_length = 1
    #     current_streak_type = None  # 'increasing' oder 'decreasing'
    #
    #     # Iteriere über die Open_Increased-Spalte
    #     for idx, row in self.df_filtered.iterrows():
    #         if pd.isna(row['Open_Increased']):
    #             continue
    #
    #         movement = 'increasing' if row['Open_Increased'] else 'decreasing'
    #
    #         if current_streak_type is None:
    #             current_streak_type = movement
    #             current_streak_length = 1
    #         elif movement == current_streak_type:
    #             current_streak_length += 1
    #         else:
    #             limit = max_increasing if current_streak_type == 'increasing' else max_decreasing
    #             streaks.append((current_streak_type, min(current_streak_length, limit)))
    #             current_streak_type = movement
    #             current_streak_length = 1
    #
    #     if current_streak_type is not None:
    #         limit = max_increasing if current_streak_type == 'increasing' else max_decreasing
    #         streaks.append((current_streak_type, min(current_streak_length, limit)))
    #
    #     increasing_streaks = [length for stype, length in streaks if stype == 'increasing']
    #     decreasing_streaks = [length for stype, length in streaks if stype == 'decreasing']
    #
    #     increasing_counts = pd.Series(increasing_streaks).value_counts().sort_index()
    #     decreasing_counts = pd.Series(decreasing_streaks).value_counts().sort_index()
    #
    #     increasing_counts = increasing_counts.reindex(range(1, max_increasing + 1), fill_value=0)
    #     decreasing_counts = decreasing_counts.reindex(range(1, max_decreasing + 1), fill_value=0)
    #
    #     total_streaks = len(increasing_streaks) + len(decreasing_streaks)
    #     total_streak_days = sum(increasing_streaks) + sum(decreasing_streaks)
    #     total_days = len(self.df_filtered) - 1
    #
    #     print(f"Überprüfung: Gesamtanzahl der Tage in Streaks = {total_streak_days}, Gesamtanzahl der analysierten Tage = {total_days}")
    #     if total_streak_days != total_days:
    #         print("Warnung: Die Summe der Streak-Tage stimmt nicht mit der Gesamtanzahl der analysierten Tage überein.")
    #     else:
    #         print("Überprüfung erfolgreich: Die Summe der Streak-Tage stimmt mit der Gesamtanzahl der analysierten Tage überein.")
    #
    #     if total_streaks > 0:
    #         increasing_percentages = (increasing_counts / total_streaks * 100).round(2)
    #         decreasing_percentages = (decreasing_counts / total_streaks * 100).round(2)
    #     else:
    #         increasing_percentages = increasing_counts
    #         decreasing_percentages = decreasing_counts
    #
    #     self.streak_results['increasing'] = {
    #         'counts': increasing_counts,
    #         'percentages': increasing_percentages
    #     }
    #     self.streak_results['decreasing'] = {
    #         'counts': decreasing_counts,
    #         'percentages': decreasing_percentages
    #     }
    #     self.streak_results['total_streaks'] = total_streaks
    #     self.streak_results['total_streak_days'] = total_streak_days

    def analyze_data(self, compare_column='Close'):
        """
        Analysiert die Daten, um die Gesamtanzahl der Datensätze,
        die Anzahl der Tage mit gestiegenem Close-Kurs und den Prozentsatz zu berechnen.
        (Angepasst auf Close anstatt Open)
        """
        if self.df_filtered is None:
            print("Daten sind nicht geladen. Bitte führen Sie 'load_and_clean_data()' zuerst aus.")
            return

        try:
            # Gesamtanzahl der gültigen Datensätze
            total_records = len(self.df_filtered)

            if total_records < 2:
                raise ValueError("Nicht genügend Datenpunkte nach der Filterung vorhanden, um Vergleiche anzustellen.")

            # Anzahl der Tage mit gestiegenem Close-Wert (sum of 'Close_Increased')
            increased_days = self.df_filtered['Close_Increased'].sum()

            # Prozentsatz
            percentage_increased = (increased_days / (total_records - 1)) * 100  # -1 wegen shift

            # Ergebnisse
            self.results = {
                'Total_Data_Points': total_records,
                'Increased_Close_Days': int(increased_days),
                'Percentage_Increased': round(percentage_increased, 2)
            }

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten bei der Datenanalyse: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def analyze_data(self, compare_column='Open'):
    #     """
    #     Analysiert die Daten, um die Gesamtanzahl der Datensätze,
    #     die Anzahl der Tage mit gestiegenem Open-Kurs und den Prozentsatz zu berechnen.
    #     """
    #     if self.df_filtered is None:
    #         print("Daten sind nicht geladen. Bitte führen Sie 'load_and_clean_data()' zuerst aus.")
    #         return
    #
    #     try:
    #         total_records = len(self.df_filtered)
    #         if total_records < 2:
    #             raise ValueError("Nicht genügend Datenpunkte nach der Filterung vorhanden, um Vergleiche anzustellen.")
    #
    #         increased_days = self.df_filtered['Open_Increased'].sum()
    #         percentage_increased = (increased_days / (total_records - 1)) * 100
    #
    #         self.results = {
    #             'Total_Data_Points': total_records,
    #             'Increased_Open_Days': int(increased_days),
    #             'Percentage_Increased': round(percentage_increased, 2)
    #         }
    #     except Exception as e:
    #         print(f"Ein Fehler ist aufgetreten bei der Datenanalyse: {e}")

    def get_results(self):
        """
        Gibt die Analyseergebnisse zurück.
        """
        return self.results

    def print_results(self):
        """
        Druckt die Analyseergebnisse in einer lesbaren Form.
        (Angepasst auf Close-Werte)
        """
        if not self.results:
            print("Keine Ergebnisse vorhanden. Bitte führen Sie zuerst die Analyse durch.")
            return

        start_date = self.df_filtered['Date'].iloc[0].date()
        print("Analyse der S&P 500 historischen Daten (basierend auf Close-Kursen):")
        print(f"Alle Tage mit vollständigen Datensätzen ab dem {start_date}: {self.results['Total_Data_Points']} Tage")
        print(f"Anzahl der Tage mit gestiegenem Close-Kurs: {self.results['Increased_Close_Days']}")
        print(f"Prozentsatz der Tage mit gestiegenem Close-Kurs: {self.results['Percentage_Increased']}%")

    # Originalfassung (auskommentiert, basierend auf Open-Werten):
    # def print_results(self):
    #     if not self.results:
    #         print("Keine Ergebnisse vorhanden. Bitte führen Sie zuerst die Analyse durch.")
    #         return
    #
    #     start_date = self.df_filtered['Date'].iloc[0].date()
    #     print("Analyse der S&P 500 historischen Daten:")
    #     print(f"Alle Tage mit vollständigen Datensätzen ab dem {start_date}: {self.results['Total_Data_Points']} Tage")
    #     print(f"Anzahl der Tage mit gestiegenem Open-Kurs: {self.results['Increased_Open_Days']}")
    #     print(f"Prozentsatz der Tage mit gestiegenem Open-Kurs: {self.results['Percentage_Increased']}%")

    def plot_data_quality(self):
        """
        Plottet ein Balkendiagramm, das die Anzahl der ausgeschlossenen und eingeschlossenen Datenpunkte zeigt.
        (Unverändert, da sich nichts an Open/Close ändert.)
        """
        labels = ['Ausgeschlossene Datenpunkte', 'Eingeschlossene Datenpunkte']
        counts = [len(self.df_excluded), len(self.df_filtered)]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color=['red', 'green'])
        plt.title('Datenqualitätsfilterung')
        plt.ylabel('Anzahl der Datenpunkte')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_close_price_movement(self):
        """
        Plottet ein Balkendiagramm, das die Anzahl der Tage mit gestiegenem und gefallenem Close-Kurs zeigt.
        (Angepasst auf Close anstatt Open)
        """
        try:
            increased = self.results['Increased_Close_Days']
            decreased = self.results['Total_Data_Points'] - increased

            labels = ['Gestiegener Close-Kurs', 'Gesunkener Close-Kurs']
            counts = [increased, decreased]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, counts, color=['blue', 'orange'])
            plt.title('Vergleich der Close-Kursbewegungen')
            plt.ylabel('Anzahl der Tage')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der Close-Kursbewegungen: {e}")

    # Originalfassung (auskommentiert, basierend auf Open-Kursen):
    # def plot_open_price_movement(self):
    #     try:
    #         increased = self.results['Increased_Open_Days']
    #         decreased = self.results['Total_Data_Points'] - increased
    #
    #         labels = ['Gestiegener Open-Kurs', 'Gesunkener Open-Kurs']
    #         counts = [increased, decreased]
    #
    #         plt.figure(figsize=(8, 6))
    #         bars = plt.bar(labels, counts, color=['blue', 'orange'])
    #         plt.title('Vergleich der Open-Kursbewegungen')
    #         plt.ylabel('Anzahl der Tage')
    #
    #         for bar in bars:
    #             height = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der Open-Kursbewegungen: {e}")

    def plot_consecutive_close_streaks(self, max_increasing=13, max_decreasing=10):
        """
        Plottet separate Histogramme, die zeigen, wie oft der Close-Kurs an aufeinanderfolgenden Tagen
        gestiegen oder gefallen ist.
        (Angepasst auf Close anstatt Open)
        """
        try:
            increasing_streaks = self.streak_results['increasing']['counts']
            decreasing_streaks = self.streak_results['decreasing']['counts']

            increasing_streaks = increasing_streaks.reindex(range(1, max_increasing + 1), fill_value=0)
            decreasing_streaks = decreasing_streaks.reindex(range(1, max_decreasing + 1), fill_value=0)

            if increasing_streaks.iloc[-1] == 0:
                increasing_streaks = increasing_streaks.iloc[:-1]
            if decreasing_streaks.iloc[-1] == 0:
                decreasing_streaks = decreasing_streaks.iloc[:-1]

            increasing_streaks = increasing_streaks.reset_index()
            increasing_streaks.columns = ['Streak_Length', 'Count']
            increasing_streaks['Streak_Length'] = increasing_streaks['Streak_Length'].astype(str)

            decreasing_streaks = decreasing_streaks.reset_index()
            decreasing_streaks.columns = ['Streak_Length', 'Count']
            decreasing_streaks['Streak_Length'] = decreasing_streaks['Streak_Length'].astype(str)

            # Plot für steigende Streaks
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Streak_Length',
                y='Count',
                data=increasing_streaks,
                color='green',
                edgecolor='black',
                order=[str(i) for i in range(1, len(increasing_streaks)+1)]
            )
            plt.title('Histogramm der aufeinanderfolgenden steigenden Close-Kurs-Tage')
            plt.xlabel('Anzahl aufeinanderfolgender Tage')
            plt.ylabel('Häufigkeit')

            for index, row in increasing_streaks.iterrows():
                if row['Count'] > 0:
                    plt.text(index, row['Count'], f"{int(row['Count'])}", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()

            # Plot für fallende Streaks
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Streak_Length',
                y='Count',
                data=decreasing_streaks,
                color='red',
                edgecolor='black',
                order=[str(i) for i in range(1, len(decreasing_streaks)+1)]
            )
            plt.title('Histogramm der aufeinanderfolgenden fallenden Close-Kurs-Tage')
            plt.xlabel('Anzahl aufeinanderfolgender Tage')
            plt.ylabel('Häufigkeit')

            for index, row in decreasing_streaks.iterrows():
                if row['Count'] > 0:
                    plt.text(index, row['Count'], f"{int(row['Count'])}", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der aufeinanderfolgenden Streaks: {e}")

    # Originalfassung (auskommentiert, basierend auf Open):
    # def plot_consecutive_open_streaks(self, max_increasing=13, max_decreasing=10):
    #     try:
    #         increasing_streaks = self.streak_results['increasing']['counts']
    #         decreasing_streaks = self.streak_results['decreasing']['counts']
    #
    #         increasing_streaks = increasing_streaks.reindex(range(1, max_increasing + 1), fill_value=0)
    #         decreasing_streaks = decreasing_streaks.reindex(range(1, max_decreasing + 1), fill_value=0)
    #
    #         if increasing_streaks.iloc[-1] == 0:
    #             increasing_streaks = increasing_streaks.iloc[:-1]
    #         if decreasing_streaks.iloc[-1] == 0:
    #             decreasing_streaks = decreasing_streaks.iloc[:-1]
    #
    #         increasing_streaks = increasing_streaks.reset_index()
    #         increasing_streaks.columns = ['Streak_Length', 'Count']
    #         increasing_streaks['Streak_Length'] = increasing_streaks['Streak_Length'].astype(str)
    #
    #         decreasing_streaks = decreasing_streaks.reset_index()
    #         decreasing_streaks.columns = ['Streak_Length', 'Count']
    #         decreasing_streaks['Streak_Length'] = decreasing_streaks['Streak_Length'].astype(str)
    #
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Streak_Length', y='Count', data=increasing_streaks, color='green', edgecolor='black', order=[str(i) for i in range(1, len(increasing_streaks)+1)])
    #         plt.title('Histogramm der aufeinanderfolgenden steigenden Open-Kurs-Tage')
    #         plt.xlabel('Anzahl aufeinanderfolgender Tage')
    #         plt.ylabel('Häufigkeit')
    #
    #         for index, row in increasing_streaks.iterrows():
    #             if row['Count'] > 0:
    #                 plt.text(index, row['Count'], f"{int(row['Count'])}", ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Streak_Length', y='Count', data=decreasing_streaks, color='red', edgecolor='black', order=[str(i) for i in range(1, len(decreasing_streaks)+1)])
    #         plt.title('Histogramm der aufeinanderfolgenden fallenden Open-Kurs-Tage')
    #         plt.xlabel('Anzahl aufeinanderfolgender Tage')
    #         plt.ylabel('Häufigkeit')
    #
    #         for index, row in decreasing_streaks.iterrows():
    #             if row['Count'] > 0:
    #                 plt.text(index, row['Count'], f"{int(row['Count'])}", ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der aufeinanderfolgenden Streaks: {e}")

    def plot_percentage_distribution_consecutive_streaks(self, max_increasing=13, max_decreasing=10):
        """
        Plottet die prozentuale Verteilung der aufeinanderfolgenden steigenden und fallenden Close-Kurs-Tage
        in separaten Balkendiagrammen.
        (Angepasst auf Close anstatt Open)
        """
        try:
            # Prozentuale Verteilung der steigenden Streaks
            increasing_percentages = self.streak_results['increasing']['percentages']
            increasing_percentages = increasing_percentages.reindex(range(1, max_increasing + 1), fill_value=0)
            if increasing_percentages.iloc[-1] == 0:
                increasing_percentages = increasing_percentages.iloc[:-1]

            increasing_percentages = increasing_percentages.reset_index()
            increasing_percentages.columns = ['Streak_Length', 'Percentage']
            increasing_percentages['Streak_Length'] = increasing_percentages['Streak_Length'].astype(str)

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Streak_Length',
                y='Percentage',
                data=increasing_percentages,
                color='green',
                edgecolor='black',
                order=[str(i) for i in range(1, len(increasing_percentages)+1)]
            )
            plt.title('Prozentuale Verteilung der steigenden Close-Kurs-Streaks')
            plt.xlabel('Anzahl aufeinanderfolgender Tage')
            plt.ylabel('Prozentuale Verteilung (%)')

            for index, row in increasing_percentages.iterrows():
                if row['Percentage'] > 0:
                    plt.text(index, row['Percentage'] + 0.5, f"{row['Percentage']}%", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()

            # Prozentuale Verteilung der fallenden Streaks
            decreasing_percentages = self.streak_results['decreasing']['percentages']
            decreasing_percentages = decreasing_percentages.reindex(range(1, max_decreasing + 1), fill_value=0)
            if decreasing_percentages.iloc[-1] == 0:
                decreasing_percentages = decreasing_percentages.iloc[:-1]

            decreasing_percentages = decreasing_percentages.reset_index()
            decreasing_percentages.columns = ['Streak_Length', 'Percentage']
            decreasing_percentages['Streak_Length'] = decreasing_percentages['Streak_Length'].astype(str)

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='Streak_Length',
                y='Percentage',
                data=decreasing_percentages,
                color='red',
                edgecolor='black',
                order=[str(i) for i in range(1, len(decreasing_percentages)+1)]
            )
            plt.title('Prozentuale Verteilung der fallenden Close-Kurs-Streaks')
            plt.xlabel('Anzahl aufeinanderfolgender Tage')
            plt.ylabel('Prozentuale Verteilung (%)')

            for index, row in decreasing_percentages.iterrows():
                if row['Percentage'] > 0:
                    plt.text(index, row['Percentage'] + 0.5, f"{row['Percentage']}%", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der prozentualen Verteilung der Streaks: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_percentage_distribution_consecutive_streaks(self, max_increasing=13, max_decreasing=10):
    #     """
    #     Plottet die prozentuale Verteilung der aufeinanderfolgenden steigenden und fallenden Open-Kurs-Tage.
    #     """
    #     try:
    #         increasing_percentages = self.streak_results['increasing']['percentages']
    #         increasing_percentages = increasing_percentages.reindex(range(1, max_increasing + 1), fill_value=0)
    #         if increasing_percentages.iloc[-1] == 0:
    #             increasing_percentages = increasing_percentages.iloc[:-1]
    #
    #         increasing_percentages = increasing_percentages.reset_index()
    #         increasing_percentages.columns = ['Streak_Length', 'Percentage']
    #         increasing_percentages['Streak_Length'] = increasing_percentages['Streak_Length'].astype(str)
    #
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Streak_Length', y='Percentage', data=increasing_percentages, color='green', edgecolor='black', order=[str(i) for i in range(1, len(increasing_percentages)+1)])
    #         plt.title('Prozentuale Verteilung der steigenden Open-Kurs-Streaks')
    #         plt.xlabel('Anzahl aufeinanderfolgender Tage')
    #         plt.ylabel('Prozentuale Verteilung (%)')
    #
    #         for index, row in increasing_percentages.iterrows():
    #             if row['Percentage'] > 0:
    #                 plt.text(index, row['Percentage'] + 0.5, f"{row['Percentage']}%", ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         decreasing_percentages = self.streak_results['decreasing']['percentages']
    #         decreasing_percentages = decreasing_percentages.reindex(range(1, max_decreasing + 1), fill_value=0)
    #         if decreasing_percentages.iloc[-1] == 0:
    #             decreasing_percentages = decreasing_percentages.iloc[:-1]
    #
    #         decreasing_percentages = decreasing_percentages.reset_index()
    #         decreasing_percentages.columns = ['Streak_Length', 'Percentage']
    #         decreasing_percentages['Streak_Length'] = decreasing_percentages['Streak_Length'].astype(str)
    #
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Streak_Length', y='Percentage', data=decreasing_percentages, color='red', edgecolor='black', order=[str(i) for i in range(1, len(decreasing_percentages)+1)])
    #         plt.title('Prozentuale Verteilung der fallenden Open-Kurs-Streaks')
    #         plt.xlabel('Anzahl aufeinanderfolgender Tage')
    #         plt.ylabel('Prozentuale Verteilung (%)')
    #
    #         for index, row in decreasing_percentages.iterrows():
    #             if row['Percentage'] > 0:
    #                 plt.text(index, row['Percentage'] + 0.5, f"{row['Percentage']}%", ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der prozentualen Verteilung der Streaks: {e}")

    def plot_yearly_increases(self):
        """
        Plottet ein Balkendiagramm der jährlichen Anzahl der Tage,
        an denen der Close-Kurs gestiegen oder gesunken ist.
        (Angepasst auf Close anstatt Open)
        """
        try:
            self.df_filtered['Year'] = self.df_filtered['Date'].dt.year

            yearly_movement = self.df_filtered.groupby(['Year', 'Close_Increased']).size().unstack(fill_value=0)

            yearly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
            plt.title('Jährliche Anzahl der Tage mit gestiegenem und gesunkenem Close-Kurs')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl der Tage')
            plt.legend(['Gesunken', 'Gestiegen'])
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der jährlichen Kursbewegungen: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_yearly_increases(self):
    #     """
    #     Plottet ein Balkendiagramm der jährlichen Anzahl der Tage,
    #     an denen der Open-Kurs gestiegen oder gesunken ist.
    #     """
    #     try:
    #         self.df_filtered['Year'] = self.df_filtered['Date'].dt.year
    #         yearly_movement = self.df_filtered.groupby(['Year', 'Open_Increased']).size().unstack(fill_value=0)
    #
    #         yearly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
    #         plt.title('Jährliche Anzahl der Tage mit gestiegenem und gesunkenem Open-Kurs')
    #         plt.xlabel('Jahr')
    #         plt.ylabel('Anzahl der Tage')
    #         plt.legend(['Gesunken', 'Gestiegen'])
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der jährlichen Kursbewegungen: {e}")

    def plot_monthly_increases(self):
        """
        Plottet ein Balkendiagramm der monatlichen Anzahl der Monate,
        in denen der Close-Kurs am ersten Tag des Monats gestiegen oder gefallen ist.
        (Angepasst auf Close anstatt Open)
        """
        try:
            self.df_filtered.sort_values('Date', inplace=True)

            # Erster Tag jedes Monats
            first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)

            first_days['Previous_Close'] = first_days['Close'].shift(1)
            first_days['Monthly_Increased'] = first_days['Close'] > first_days['Previous_Close']

            first_days = first_days.dropna(subset=['Previous_Close'])

            monthly_increased = first_days['Monthly_Increased'].sum()
            monthly_decreased = len(first_days) - monthly_increased
            percentage_increased = (monthly_increased / len(first_days)) * 100

            # Balkendiagramm 1: absolute Werte
            plt.figure(figsize=(8, 6))
            labels = ['Gesunkener Close-Kurs', 'Gestiegener Close-Kurs']
            counts = [monthly_decreased, monthly_increased]
            plt.bar(labels, counts, color=['orange', 'green'])
            plt.title('Monatliche Close-Kursbewegungen')
            plt.ylabel('Anzahl der Monate')

            for i, count in enumerate(counts):
                if count > 0:
                    plt.text(i, count, f'{count}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()

            # Balkendiagramm 2: prozentuale Verteilung
            plt.figure(figsize=(8, 6))
            plt.bar(labels, [100 - percentage_increased, percentage_increased], color=['orange', 'green'])
            plt.title('Prozentuale Verteilung der monatlichen Close-Kursbewegungen')
            plt.ylabel('Prozentuale Verteilung (%)')

            for i, percentage in enumerate([100 - percentage_increased, percentage_increased]):
                if percentage > 0:
                    plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der monatlichen Kursbewegungen: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_monthly_increases(self):
    #     """
    #     Plottet ein Balkendiagramm der monatlichen Anzahl der Monate,
    #     in denen der Open-Kurs am ersten Tag des Monats gestiegen oder gefallen ist.
    #     """
    #     try:
    #         self.df_filtered.sort_values('Date', inplace=True)
    #         first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)
    #
    #         first_days['Previous_Open'] = first_days['Open'].shift(1)
    #         first_days['Monthly_Increased'] = first_days['Open'] > first_days['Previous_Open']
    #
    #         first_days = first_days.dropna(subset=['Previous_Open'])
    #
    #         monthly_increased = first_days['Monthly_Increased'].sum()
    #         monthly_decreased = len(first_days) - monthly_increased
    #         percentage_increased = (monthly_increased / len(first_days)) * 100
    #
    #         plt.figure(figsize=(8, 6))
    #         labels = ['Gesunkener Open-Kurs', 'Gestiegener Open-Kurs']
    #         counts = [monthly_decreased, monthly_increased]
    #         plt.bar(labels, counts, color=['orange', 'green'])
    #         plt.title('Monatliche Open-Kursbewegungen')
    #         plt.ylabel('Anzahl der Monate')
    #
    #         for i, count in enumerate(counts):
    #             if count > 0:
    #                 plt.text(i, count, f'{count}', ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         plt.figure(figsize=(8, 6))
    #         plt.bar(labels, [100 - percentage_increased, percentage_increased], color=['orange', 'green'])
    #         plt.title('Prozentuale Verteilung der monatlichen Open-Kursbewegungen')
    #         plt.ylabel('Prozentuale Verteilung (%)')
    #
    #         for i, percentage in enumerate([100 - percentage_increased, percentage_increased]):
    #             if percentage > 0:
    #                 plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    #
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der monatlichen Kursbewegungen: {e}")

    def plot_monthly_increases_yearly(self):
        """
        Plottet ein Balkendiagramm der jährlichen Anzahl der Monate, in denen der Close-Kurs gestiegen oder gefallen ist.
        (Angepasst auf Close anstatt Open)
        """
        try:
            self.df_filtered.sort_values('Date', inplace=True)

            first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)

            first_days['Previous_Close'] = first_days['Close'].shift(1)
            first_days['Monthly_Increased'] = first_days['Close'] > first_days['Previous_Close']

            first_days = first_days.dropna(subset=['Previous_Close'])

            first_days['Year'] = first_days['Date'].dt.year

            yearly_monthly_movement = first_days.groupby(['Year', 'Monthly_Increased']).size().unstack(fill_value=0)

            yearly_monthly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
            plt.title('Jährliche Anzahl der Monate mit gestiegenem und gesunkenem Close-Kurs')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl der Monate')
            plt.legend(['Gesunken', 'Gestiegen'])
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der jährlichen monatlichen Kursbewegungen: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_monthly_increases_yearly(self):
    #     """
    #     Plottet ein Balkendiagramm der jährlichen Anzahl der Monate, in denen der Open-Kurs gestiegen oder gefallen ist.
    #     """
    #     try:
    #         self.df_filtered.sort_values('Date', inplace=True)
    #         first_days = self.df_filtered.groupby([self.df_filtered['Date'].dt.year, self.df_filtered['Date'].dt.month]).first().reset_index(drop=True)
    #
    #         first_days['Previous_Open'] = first_days['Open'].shift(1)
    #         first_days['Monthly_Increased'] = first_days['Open'] > first_days['Previous_Open']
    #         first_days = first_days.dropna(subset=['Previous_Open'])
    #
    #         first_days['Year'] = first_days['Date'].dt.year
    #
    #         yearly_monthly_movement = first_days.groupby(['Year', 'Monthly_Increased']).size().unstack(fill_value=0)
    #
    #         yearly_monthly_movement.plot(kind='bar', stacked=False, figsize=(12, 8), color=['red', 'green'])
    #         plt.title('Jährliche Anzahl der Monate mit gestiegenem und gesunkenem Open-Kurs')
    #         plt.xlabel('Jahr')
    #         plt.ylabel('Anzahl der Monate')
    #         plt.legend(['Gesunken', 'Gestiegen'])
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der jährlichen monatlichen Kursbewegungen: {e}")

    def plot_cumulative_returns(self):
        """
        Plottet die kumulativen Renditen der Close-Kurse über die Zeit.
        (Angepasst auf Close anstatt Open)
        """
        try:
            self.df_filtered['Daily_Return'] = self.df_filtered['Close'].pct_change()
            self.df_filtered['Cumulative_Return'] = (1 + self.df_filtered['Daily_Return']).cumprod()

            plt.figure(figsize=(14, 7))
            plt.plot(self.df_filtered['Date'], self.df_filtered['Cumulative_Return'], label='Kumulative Rendite', color='purple')
            plt.title('Kumulative Renditen der Close-Kurse')
            plt.xlabel('Datum')
            plt.ylabel('Kumulative Rendite')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der kumulativen Renditen: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_cumulative_returns(self):
    #     """
    #     Plottet die kumulativen Renditen der Open-Kurse über die Zeit.
    #     """
    #     try:
    #         self.df_filtered['Daily_Return'] = self.df_filtered['Open'].pct_change()
    #         self.df_filtered['Cumulative_Return'] = (1 + self.df_filtered['Daily_Return']).cumprod()
    #
    #         plt.figure(figsize=(14, 7))
    #         plt.plot(self.df_filtered['Date'], self.df_filtered['Cumulative_Return'], label='Kumulative Rendite', color='purple')
    #         plt.title('Kumulative Renditen der Open-Kurse')
    #         plt.xlabel('Datum')
    #         plt.ylabel('Kumulative Rendite')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der kumulativen Renditen: {e}")

    def plot_rolling_statistics(self, window=50):
        """
        Plottet rollende Mittelwerte und Standardabweichungen der Close-Kurse.
        (Angepasst auf Close anstatt Open)
        Parameters:
        - window (int): Das Fenster für die rollende Berechnung.
        """
        try:
            self.df_filtered['Rolling_Mean'] = self.df_filtered['Close'].rolling(window=window).mean()
            self.df_filtered['Rolling_Std'] = self.df_filtered['Close'].rolling(window=window).std()

            plt.figure(figsize=(14, 7))
            plt.plot(self.df_filtered['Date'], self.df_filtered['Close'], label='Close-Kurs', color='blue', alpha=0.5)
            plt.plot(self.df_filtered['Date'], self.df_filtered['Rolling_Mean'], label=f'{window}-Tage gleitender Durchschnitt', color='red')
            plt.fill_between(
                self.df_filtered['Date'],
                self.df_filtered['Rolling_Mean'] - self.df_filtered['Rolling_Std'],
                self.df_filtered['Rolling_Mean'] + self.df_filtered['Rolling_Std'],
                color='gray', alpha=0.2, label='Rolling Std Dev'
            )

            plt.title(f'Rolling {window}-Day Mean und Standardabweichung des Close-Kurses')
            plt.xlabel('Datum')
            plt.ylabel('Close-Kurs')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der rollenden Statistiken: {e}")

    # Originalfassung (auskommentiert, basiert auf Open):
    # def plot_rolling_statistics(self, window=50):
    #     """
    #     Plottet rollende Mittelwerte und Standardabweichungen der Open-Kurse.
    #     """
    #     try:
    #         self.df_filtered['Rolling_Mean'] = self.df_filtered['Open'].rolling(window=window).mean()
    #         self.df_filtered['Rolling_Std'] = self.df_filtered['Open'].rolling(window=window).std()
    #
    #         plt.figure(figsize=(14, 7))
    #         plt.plot(self.df_filtered['Date'], self.df_filtered['Open'], label='Open-Kurs', color='blue', alpha=0.5)
    #         plt.plot(self.df_filtered['Date'], self.df_filtered['Rolling_Mean'], label=f'{window}-Tage gleitender Durchschnitt', color='red')
    #         plt.fill_between(self.df_filtered['Date'],
    #                          self.df_filtered['Rolling_Mean'] - self.df_filtered['Rolling_Std'],
    #                          self.df_filtered['Rolling_Mean'] + self.df_filtered['Rolling_Std'],
    #                          color='gray', alpha=0.2, label='Rolling Std Dev')
    #
    #         plt.title(f'Rolling {window}-Day Mean and Standard Deviation of Open-Kurs')
    #         plt.xlabel('Datum')
    #         plt.ylabel('Open-Kurs')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #     except KeyError as e:
    #         print(f"Ein Fehler ist aufgetreten beim Plotten der rollenden Statistiken: {e}")

    def plot_correlation_heatmap(self):
        """
        Plottet eine Heatmap der Korrelationen zwischen den verschiedenen Kurskennzahlen.
        (Unverändert, da wir alle Spalten gleich behandeln und nur Auswertung via DF)
        """
        try:
            corr = self.df_filtered[['Close', 'High', 'Low', 'Open', 'Volume']].corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Korrelationsmatrix der Kurskennzahlen')
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der Korrelationsmatrix: {e}")

    def plot_seasonal_decomposition(self, model='additive', period=None):
        """
        Führt eine saisonale Zerlegung der Close-Kurse durch und plottet die Komponenten.
        (Angepasst auf Close anstatt Open)
        """
        try:
            decomposition = seasonal_decompose(self.df_filtered['Close'], model=model, period=period)

            fig = decomposition.plot()
            fig.set_size_inches(14, 10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten der saisonalen Zerlegung: {e}")

    def plot_volume_vs_movement(self):
        """
        Plottet das Handelsvolumen im Vergleich zu den Kursbewegungen (Close).
        (Angepasst auf Close_Increased)
        """
        try:
            plt.figure(figsize=(14, 7))
            sns.scatterplot(x='Volume', y='Close_Increased', data=self.df_filtered, alpha=0.5)
            plt.title('Handelsvolumen vs. Close-Kursbewegungen')
            plt.xlabel('Volume')
            plt.ylabel('Close-Kurs Gestiegen (1) / Gesunken (0)')
            plt.tight_layout()
            plt.show()
        except KeyError as e:
            print(f"Ein Fehler ist aufgetreten beim Plotten von Volume vs Movement: {e}")

    def main(self):
        """
        Hauptfunktion, die den Analyseprozess steuert.
        (Angepasst auf Close anstatt Open)
        """
        # Laden und Bereinigen der Daten
        self.load_and_clean_data()

        # Analysieren der Daten (Standard: Vergleich der 'Close'-Spalte)
        self.analyze_data(compare_column='Close')

        # Ergebnisse ausgeben
        self.print_results()

        # Datenqualitätsfilter
        self.plot_data_quality()

        # Plot der Close-Kursbewegungen
        self.plot_close_price_movement()

        # Plot der aufeinanderfolgenden steigenden und fallenden Streaks (Close)
        self.plot_consecutive_close_streaks(max_increasing=13, max_decreasing=10)

        # Prozentuale Verteilung der Streaks (Close)
        self.plot_percentage_distribution_consecutive_streaks(max_increasing=13, max_decreasing=10)

        # Jährliche Kursbewegungen (Close)
        self.plot_yearly_increases()

        # Jährliche monatliche Kursbewegungen (Close)
        self.plot_monthly_increases_yearly()

        # Monatliche Kursbewegungen (Close)
        self.plot_monthly_increases()

        # Optional: Weitere Analysen und Plots
        # self.plot_cumulative_returns()
        # self.plot_rolling_statistics(window=50)
        # self.plot_correlation_heatmap()
        # self.plot_seasonal_decomposition(model='additive', period=252)
        # self.plot_volume_vs_movement()

# Beispielverwendung
def main():
    """
    Hauptfunktion, die den Analyseprozess steuert.
    (Angepasst auf Close)
    """
    csv_path = "sp500_data/SP500_Index_Historical_Data.csv"
    analyzer = PriceIncreaseAnalyzer(subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv")
    analyzer.main()

if __name__ == "__main__":
    main()
