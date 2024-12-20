import pandas as pd
from pathlib import Path

class ConsecutiveStreakAnalyzer:
    def __init__(self, csv_path):
        """
        Initialisiert den Analyzer mit dem Pfad zur CSV-Datei.

        Parameters:
        - csv_path (str): Der Pfad zur CSV-Datei.
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self.streak_results = {}

    def load_and_prepare_data(self):
        """
        Lädt die CSV-Datei und bereitet die Daten vor, indem unvollständige Datensätze entfernt werden.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Die Datei {self.csv_path} wurde nicht gefunden.")

        # CSV laden
        self.df = pd.read_csv(self.csv_path)

        # Datum konvertieren und sortieren
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)

        # Filtere vollständige Datensätze
        self.df = self.df[(self.df['High'] != 0.0) & (self.df['Low'] != 0.0) & (self.df['Open'] != 0.0)].copy()

        # Shift für die vorherigen Werte
        self.df['Previous_Open'] = self.df['Open'].shift(1)
        self.df['Open_Increased'] = self.df['Open'] > self.df['Previous_Open']

    def calculate_streaks(self, max_streak=30):
        """
        Berechnet die Anzahl und den prozentualen Anteil der aufeinanderfolgenden steigenden und fallenden Tage.

        Parameters:
        - max_streak (int): Die maximale Länge der zu analysierenden Streaks.
        """
        # Initialisieren der Streak-Listen
        streaks = []
        current_streak_length = 1
        current_streak_type = None  # 'increasing' oder 'decreasing'

        # Iteriere über die Open_Increased-Spalte
        for idx, row in self.df.iterrows():
            if pd.isna(row['Open_Increased']):
                continue  # Überspringe den ersten Datensatz ohne vorherigen Wert

            movement = 'increasing' if row['Open_Increased'] else 'decreasing'

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
        print(f"Überprüfung: Gesamtanzahl der Tage in Streaks = {total_streak_days}, Gesamtanzahl der analysierten Tage = {total_days}")
        if total_streak_days != total_days:
            print("Warnung: Die Summe der Streak-Tage stimmt nicht mit der Gesamtanzahl der analysierten Tage überein.")
        else:
            print("Überprüfung erfolgreich: Die Summe der Streak-Tage stimmt mit der Gesamtanzahl der analysierten Tage überein.")

        # Ergebnisse speichern
        self.streak_results['increasing'] = {
            'counts': increasing_counts,
            'percentages': (increasing_counts / total_streaks * 100).round(2)
        }
        self.streak_results['decreasing'] = {
            'counts': decreasing_counts,
            'percentages': (decreasing_counts / total_streaks * 100).round(2)
        }
        self.streak_results['total_streaks'] = total_streaks
        self.streak_results['total_streak_days'] = total_streak_days

    def print_results(self):
        """
        Gibt die berechneten Streak-Ergebnisse in der Konsole aus.
        """
        print("Analyse der aufeinanderfolgenden Open-Kurs-Tage")
        print(f"\nGesamtanzahl der Streaks: {self.streak_results['total_streaks']}")
        print(f"Gesamtanzahl der Tage in den Streaks: {self.streak_results['total_streak_days']}\n")

        print("Steigende Streaks:")
        print("Länge\tAnzahl\tProzentualer Anteil")
        for streak, count in self.streak_results['increasing']['counts'].items():
            percentage = self.streak_results['increasing']['percentages'].get(streak, 0)
            print(f"{streak}\t{count}\t{percentage}%")

        print("\nFallende Streaks:")
        print("Länge\tAnzahl\tProzentualer Anteil")
        for streak, count in self.streak_results['decreasing']['counts'].items():
            percentage = self.streak_results['decreasing']['percentages'].get(streak, 0)
            print(f"{streak}\t{count}\t{percentage}%")

# Beispielverwendung
if __name__ == "__main__":
    csv_path = "C:\\Users\\Anwender\\PycharmProjects\\Unternehmenssoftware\\sp500_data\\SP500_Index_Historical_Data.csv"

    analyzer = ConsecutiveStreakAnalyzer(csv_path)
    analyzer.load_and_prepare_data()
    analyzer.calculate_streaks()
    analyzer.print_results()
