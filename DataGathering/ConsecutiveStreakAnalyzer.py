import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ConsecutiveStreakAnalyzer:
    def __init__(self, csv_filename="SP500_Index_Historical_Data.csv", subfolder="sp500_data"):
        """
        Initialisiert den Analyzer mit dem relativen Pfad zur CSV-Datei.

        Parameters:
        - csv_filename (str): Der Name der CSV-Datei.
        - subfolder (str, optional): Der Name des Unterordners im Projektverzeichnis, in dem sich die CSV-Datei befindet.
        """
        # Bestimme das Projektverzeichnis (eine Ebene höher als der Skriptordner)
        script_dir = Path(__file__).parent.resolve()
        project_dir = script_dir.parent  # Eine Ebene höher

        # Setze den Pfad zur CSV-Datei
        if subfolder:
            self.csv_path = project_dir / subfolder / csv_filename
        else:
            self.csv_path = project_dir / csv_filename

        self.df = None
        self.streak_results = {
            'Open': {},
            'Close': {}
        }

    def load_and_prepare_data(self):
        """
        Lädt die CSV-Datei und bereitet die Daten vor, indem unvollständige Datensätze entfernt werden.
        Berechnet die Indikatoren für steigende und fallende Open- und Close-Kurse.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Die Datei {self.csv_path} wurde nicht gefunden.")

        # CSV laden
        self.df = pd.read_csv(self.csv_path)

        # Datum konvertieren und sortieren
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)

        # Filtere vollständige Datensätze (High, Low, Open und Close nicht 0.0)
        self.df = self.df[
            (self.df['High'] != 0.0) &
            (self.df['Low'] != 0.0) &
            (self.df['Open'] != 0.0) &
            (self.df['Close'] != 0.0)
        ].copy()

        # Berechne Indikatoren für Open
        self.df['Previous_Open'] = self.df['Open'].shift(1)
        self.df['Open_Increased'] = self.df['Open'] > self.df['Previous_Open']

        # Berechne Indikatoren für Close
        self.df['Previous_Close'] = self.df['Close'].shift(1)
        self.df['Close_Increased'] = self.df['Close'] > self.df['Previous_Close']

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
        for idx, row in self.df.iterrows():
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
        total_days = len(self.df) - 1  # -1 wegen des verschobenen vorherigen Werts

        # Überprüfung
        print(f"Überprüfung für {key} - {column_name}: Gesamtanzahl der Tage in Streaks = {total_streak_days}, Gesamtanzahl der analysierten Tage = {total_days}")
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
                print(f"\nKeine Ergebnisse für {key}-Kurse vorhanden. Bitte führen Sie zuerst die Streak-Berechnung durch.")
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
if __name__ == "__main__":
    # Initialisiere den Analyzer, der den sp500_data-Ordner im Projektverzeichnis sucht
    analyzer = ConsecutiveStreakAnalyzer(subfolder="sp500_data")  # Setze `subfolder=None`, wenn die CSV im Projektverzeichnis liegt
    analyzer.load_and_prepare_data()
    analyzer.calculate_streaks()
    analyzer.print_results()
    analyzer.plot_streaks('Open')
    analyzer.plot_streaks('Close')
