import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ConsecutiveStreakAnalyzer:
    def __init__(self, subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv"):
        """
        Initialisiert den Analyzer und sucht nach der CSV-Datei im gesamten Projektverzeichnis.

        Parameters:
        - subfolder (str): Der Name des Unterordners, in dem sich die CSV-Datei befinden könnte.
        - file_name (str): Der Name der CSV-Datei.
        """
        self.subfolder = subfolder
        self.file_name = file_name
        self.csv_path = self.find_csv_file()

        if self.csv_path is None:
            raise FileNotFoundError(
                f"Die Datei '{self.file_name}' wurde im Verzeichnis 'Unternehmenssoftware' nicht gefunden.")

        self.df = None
        self.streak_results = {
            'Open': {},
            'Close': {}
        }

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

        # Rekursive Suche nach der Datei innerhalb des "Unternehmenssoftware"-Verzeichnisses
        for root, _, files in os.walk(current_path):
            for file in files:
                if file == self.file_name:
                    return Path(root) / file

        print(f"Die Datei '{self.file_name}' wurde im 'Unternehmenssoftware'-Verzeichnis nicht gefunden.")
        return None

    def load_and_prepare_data(self):
        """
        Lädt die CSV-Datei und bereitet die Daten vor, indem unvollständige Datensätze entfernt werden.
        Berechnet die Indikatoren für steigende und fallende Open- und Close-Kurse.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Die Datei {self.csv_path} wurde nicht gefunden.")

        # CSV laden
        self.df = pd.read_csv(self.csv_path)

        # Überprüfen, ob die notwendigen Spalten vorhanden sind
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(column in self.df.columns for column in required_columns):
            raise ValueError(f"Die CSV-Datei muss die folgenden Spalten enthalten: {required_columns}")

        # Datum konvertieren und sortieren
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)

        # Identifizieren des ersten vollständigen Datensatzes (High, Low, Open und Close nicht 0.0)
        filter_condition = (
                (self.df['High'] != 0.0) &
                (self.df['Low'] != 0.0) &
                (self.df['Open'] != 0.0) &
                (self.df['Close'] != 0.0)
        )
        first_valid_index = self.df[filter_condition].index.min()

        if pd.isna(first_valid_index):
            raise ValueError("Keine vollständigen Datensätze gefunden (High, Low, Open und Close sind nicht 0.0).")

        # Aufteilen in eingeschlossene und ausgeschlossene Datensätze
        df_excluded = self.df.loc[:first_valid_index - 1].copy()
        self.df_filtered = self.df.loc[first_valid_index:].copy()

        # Sortieren der Daten nach Datum, falls nicht bereits sortiert
        self.df_filtered.sort_values('Date', inplace=True)
        self.df_filtered.reset_index(drop=True, inplace=True)

        df_excluded.sort_values('Date', inplace=True)
        df_excluded.reset_index(drop=True, inplace=True)

        # Berechne Indikatoren für Open
        self.df_filtered['Previous_Open'] = self.df_filtered['Open'].shift(1)
        self.df_filtered['Open_Increased'] = self.df_filtered['Open'] > self.df_filtered['Previous_Open']

        # Berechne Indikatoren für Close
        self.df_filtered['Previous_Close'] = self.df_filtered['Close'].shift(1)
        self.df_filtered['Close_Increased'] = self.df_filtered['Close'] > self.df_filtered['Previous_Close']

    def calculate_streaks_for_column(self, column_name, key, max_streak=30):
        """
        Berechnet die Anzahl und den prozentualen Anteil der aufeinanderfolgenden steigenden und fallenden Tage für eine gegebene Spalte.

        Parameters:
        - column_name (str): Die Spalte, die analysiert werden soll ('Open_Increased' oder 'Close_Increased').
        - key (str): Der Schlüssel unter dem die Ergebnisse gespeichert werden sollen ('Open' oder 'Close').
        - max_streak (int): Die maximale Länge der zu analysierenden Streaks.
        """
        streaks = []
        current_streak_length = 1
        current_streak_type = None  # 'increasing' oder 'decreasing'

        # Iteriere über die angegebene Spalte
        for idx, row in self.df_filtered.iterrows():
            if pd.isna(row[column_name]):
                continue  # Überspringe den ersten Datensatz ohne vorherigen Wert

            movement = 'increasing' if row[column_name] else 'decreasing'

            if current_streak_type is None:
                # Beginn der ersten Streak
                current_streak_type = movement
                current_streak_length = 1
            elif movement == current_streak_type:
                # Fortsetzung der aktuellen Streak
                current_streak_length += 1
            else:
                # Abschluss der aktuellen Streak und Start einer neuen
                streaks.append((current_streak_type, min(current_streak_length, max_streak)))
                current_streak_type = movement
                current_streak_length = 1

        # Füge die letzte Streak hinzu
        if current_streak_type is not None:
            streaks.append((current_streak_type, min(current_streak_length, max_streak)))

        # Separate Listen für steigende und fallende Streaks
        increasing_streaks = [length for stype, length in streaks if stype == 'increasing']
        decreasing_streaks = [length for stype, length in streaks if stype == 'decreasing']

        # Zählen der Streaks
        increasing_counts = pd.Series(increasing_streaks).value_counts().sort_index()
        decreasing_counts = pd.Series(decreasing_streaks).value_counts().sort_index()

        # Begrenzen auf maximale Streak-Länge
        increasing_counts = increasing_counts.reindex(range(1, max_streak + 1), fill_value=0)
        decreasing_counts = decreasing_counts.reindex(range(1, max_streak + 1), fill_value=0)

        # Gesamtanzahl der Streaks berechnen (Anzahl der einzelnen Streaks)
        total_streaks = len(increasing_streaks) + len(decreasing_streaks)

        # Gesamtanzahl der Tage in den Streaks berechnen
        total_streak_days = sum(increasing_streaks) + sum(decreasing_streaks)

        # Gesamtanzahl der analysierten Tage
        total_days = len(self.df_filtered) - 1  # -1 wegen des verschobenen vorherigen Werts

        # Überprüfung
        print(
            f"Überprüfung für {key} - {column_name}: Gesamtanzahl der Tage in Streaks = {total_streak_days}, Gesamtanzahl der analysierten Tage = {total_days}")
        if total_streak_days != total_days:
            print("Warnung: Die Summe der Streak-Tage stimmt nicht mit der Gesamtanzahl der analysierten Tage überein.")
        else:
            print(
                "Überprüfung erfolgreich: Die Summe der Streak-Tage stimmt mit der Gesamtanzahl der analysierten Tage überein.")

        # Berechnung der prozentualen Anteile
        if total_streaks > 0:
            increasing_percentages = (increasing_counts / total_streaks * 100).round(2)
            decreasing_percentages = (decreasing_counts / total_streaks * 100).round(2)
        else:
            increasing_percentages = increasing_counts
            decreasing_percentages = decreasing_counts

        # Ergebnisse speichern
        self.streak_results[key] = {
            'increasing': {
                'counts': increasing_counts,
                'percentages': increasing_percentages
            },
            'decreasing': {
                'counts': decreasing_counts,
                'percentages': decreasing_percentages
            },
            'total_streaks': total_streaks,
            'total_streak_days': total_streak_days
        }

    def calculate_streaks(self, max_streak=30):
        """
        Berechnet die Streaks für sowohl Open- als auch Close-Kurse.

        Parameters:
        - max_streak (int): Die maximale Länge der zu analysierenden Streaks.
        """
        print("\nBerechnung der Streaks für Open-Kurse...")
        self.calculate_streaks_for_column('Open_Increased', 'Open', max_streak)

        print("\nBerechnung der Streaks für Close-Kurse...")
        self.calculate_streaks_for_column('Close_Increased', 'Close', max_streak)

    def print_results(self):
        """
        Gibt die berechneten Streak-Ergebnisse für Open- und Close-Kurse in der Konsole aus.
        """
        for key in ['Open', 'Close']:
            if key not in self.streak_results:
                print(
                    f"\nKeine Ergebnisse für {key}-Kurse vorhanden. Bitte führen Sie zuerst die Streak-Berechnung durch.")
                continue

            print(f"\nAnalyse der aufeinanderfolgenden {key}-Kurs-Tage")
            print(f"Gesamtanzahl der Streaks: {self.streak_results[key]['total_streaks']}")
            print(f"Gesamtanzahl der Tage in den Streaks: {self.streak_results[key]['total_streak_days']}\n")

            print("Steigende Streaks:")
            print("Länge\tAnzahl\tProzentualer Anteil")
            for streak, count in self.streak_results[key]['increasing']['counts'].items():
                percentage = self.streak_results[key]['increasing']['percentages'].get(streak, 0)
                print(f"{streak}\t{count}\t{percentage}%")

            print("\nFallende Streaks:")
            print("Länge\tAnzahl\tProzentualer Anteil")
            for streak, count in self.streak_results[key]['decreasing']['counts'].items():
                percentage = self.streak_results[key]['decreasing']['percentages'].get(streak, 0)
                print(f"{streak}\t{count}\t{percentage}%")

    def plot_streaks(self, key='Open'):
        """
        Plottet die steigenden und fallenden Streaks für den angegebenen Kurstyp.

        Parameters:
        - key (str): 'Open' oder 'Close'.
        """
        if key not in self.streak_results:
            print(f"Keine Ergebnisse für {key}-Kurse vorhanden.")
            return

        data = self.streak_results[key]

        # Steigende Streaks
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=data['increasing']['counts'].index.astype(str),
            y=data['increasing']['counts'].values,
            color='green'
        )
        plt.title(f'Steigende {key}-Kurs-Streaks')
        plt.xlabel('Streak-Länge')
        plt.ylabel('Anzahl')
        plt.tight_layout()
        plt.show()

        # Fallende Streaks
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=data['decreasing']['counts'].index.astype(str),
            y=data['decreasing']['counts'].values,
            color='red'
        )
        plt.title(f'Fallende {key}-Kurs-Streaks')
        plt.xlabel('Streak-Länge')
        plt.ylabel('Anzahl')
        plt.tight_layout()
        plt.show()


# Beispielverwendung
def main():
    try:
        analyzer = ConsecutiveStreakAnalyzer(subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv")
        analyzer.load_and_prepare_data()
        analyzer.calculate_streaks()
        analyzer.print_results()
        analyzer.plot_streaks('Open')
        analyzer.plot_streaks('Close')
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    main()
