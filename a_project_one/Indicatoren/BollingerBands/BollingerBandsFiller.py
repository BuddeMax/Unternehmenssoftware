import os
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Klasse zum Füllen der bestehenden CSV-Dateien
# -------------------------------------------------------------------------
class BollingerBandsFiller:
    """
    Diese Klasse füllt bestehende CSV-Dateien im 'analysis_results'-Verzeichnis mit den erforderlichen Daten.
    """

    def __init__(self, merged_df: pd.DataFrame, output_directory: Path):
        """
        :param merged_df: Zusammengeführtes DataFrame mit historischen und Bollinger-Daten
        :param output_directory: Verzeichnis, in dem die CSV-Dateien gespeichert sind
        """
        self.df = merged_df.copy()
        self.output_directory = output_directory
        print(f"Verarbeitung im Ausgabe-Verzeichnis: {self.output_directory}")

    def fill_distance_data(self):
        """
        Füllt die 'Distance_Data.csv' mit den berechneten Abständen zu den Bollinger Bändern.
        Überschreibt nur die Spalten Upper_Distance, Middle_Distance, Lower_Distance.
        """
        distance_file = self.output_directory / "Distance_Data.csv"
        if distance_file.exists():
            print(f"Lade Distance_Data.csv: {distance_file}")
            distance_df = pd.read_csv(distance_file, parse_dates=['Date'])

            # Erstelle ein Hilfs-DF nur mit den Spalten, die für die Berechnung gebraucht werden
            calc_df = self.df[['Date', 'Close', 'UpperBand', 'MiddleBand', 'LowerBand']].copy()

            # Mergen, um die Close- und Band-Werte zu erhalten
            merged = pd.merge(distance_df[['Date']], calc_df, on='Date', how='left')

            # Berechnung der Distanzen
            merged["Upper_Distance"] = ((merged["Close"] - merged["UpperBand"]) / merged["UpperBand"]) * 100
            merged["Middle_Distance"] = ((merged["Close"] - merged["MiddleBand"]) / merged["MiddleBand"]) * 100
            merged["Lower_Distance"] = ((merged["Close"] - merged["LowerBand"]) / merged["LowerBand"]) * 100

            # Aktualisierung der Distance_Data.csv
            distance_df['Upper_Distance'] = merged['Upper_Distance']
            distance_df['Middle_Distance'] = merged['Middle_Distance']
            distance_df['Lower_Distance'] = merged['Lower_Distance']

            # Nur die 4 gewünschten Spalten behalten
            distance_df = distance_df[['Date', 'Upper_Distance', 'Middle_Distance', 'Lower_Distance']]

            # Speichern
            distance_df.to_csv(distance_file, index=False)
            print(f"Distance_Data.csv wurde aktualisiert: {distance_file}")
        else:
            print(f"Distance_Data.csv existiert nicht im Ausgabe-Verzeichnis: {distance_file}")

    def fill_touches_combined(self):
        """
        Füllt die 'Touches_Combined.csv' mit den berechneten Events und Toleranzen.
        Hängt neue Events an und sortiert die Datei. Nur die relevanten Spalten werden beibehalten.
        """
        combined_file = self.output_directory / "Touches_Combined.csv"
        if combined_file.exists():
            print(f"Lade Touches_Combined.csv: {combined_file}")
            combined_df = pd.read_csv(combined_file, parse_dates=['Date'])

            # Für jede Toleranz Events erzeugen und anhängen
            for tol in [0.0025, 0.00375, 0.005, 0.0075, 0.01]:
                print(f"Fülle Touches_Combined.csv für Toleranz: {tol*100}%")
                temp_df = self._get_events_for_tolerance(tol)

                # Nur die nötigen Spalten aus temp_df behalten
                temp_df = temp_df[[
                    'Date', 'Event_Type', 'Tolerance', 'Close_Price',
                    'Distance_Upper', 'Distance_Lower', 'Distance_Middle',
                    'Consecutive_Touches', 'Touch_Type',
                    'Future_Return_1', 'Future_Return_5', 'Future_Move_1', 'Future_Move_5'
                ]].copy()

                # Anhängen
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

            # Sortieren und nur die gewünschten Spalten behalten
            combined_df.sort_values(by=["Date", "Event_Type", "Tolerance"], inplace=True)
            combined_df = combined_df[[
                'Date', 'Event_Type', 'Tolerance', 'Close_Price',
                'Distance_Upper', 'Distance_Lower', 'Distance_Middle',
                'Consecutive_Touches', 'Touch_Type',
                'Future_Return_1', 'Future_Return_5', 'Future_Move_1', 'Future_Move_5'
            ]]

            # Entfernen von Duplikaten, falls notwendig
            combined_df.drop_duplicates(inplace=True)

            # Speichern
            combined_df.to_csv(combined_file, index=False)
            print(f"Touches_Combined.csv wurde aktualisiert: {combined_file}")
        else:
            print(f"Touches_Combined.csv existiert nicht im Ausgabe-Verzeichnis: {combined_file}")

    def fill_event_specific_files(self):
        """
        Füllt die event-spezifischen CSV-Dateien (z.B. Lower_Touch_0.25%.csv) mit den berechneten Daten.
        Hängt neue Zeilen an und aktualisiert vorhandene Zeilen. Nur die relevanten Spalten werden beibehalten.
        """
        # Liste der Event-Typen
        event_types = ["Upper_Touch", "Lower_Touch", "Middle_Break_Up", "Middle_Break_Down"]
        # Liste der Toleranzen
        tolerances = [0.0025, 0.00375, 0.005, 0.0075, 0.01]

        for event in event_types:
            for tol in tolerances:
                # Formatieren des Toleranz-Strings ohne unnötige Dezimalstellen
                tolerance_str = f"{tol*100:g}%"
                # Dateiname z. B. "Upper_Touch_0.25%.csv" oder "Upper_Touch_1%.csv"
                file_name = f"{event}_{tolerance_str}.csv"
                event_file = self.output_directory / file_name
                if event_file.exists():
                    print(f"Lade {file_name}: {event_file}")
                    event_df = pd.read_csv(event_file, parse_dates=['Date'])

                    # Hole alle Events für diese Toleranz
                    tolerance_df = self._get_events_for_tolerance(tol)
                    # Filter nur auf den aktuellen Event-Typ
                    filtered_df = tolerance_df[tolerance_df["Event_Type"] == event].copy()

                    # Behalte nur die benötigten Spalten
                    filtered_df = filtered_df[[
                        'Date', 'Event_Type', 'Close_Price',
                        'Distance_Upper', 'Distance_Lower', 'Distance_Middle',
                        'Consecutive_Touches', 'Touch_Type',
                        'Future_Return_1', 'Future_Return_5', 'Future_Move_1', 'Future_Move_5'
                    ]]

                    # Zusammenführen (alle neuen Zeilen anhängen + vorhandene aktualisieren)
                    updated_df = pd.merge(event_df, filtered_df,
                                          on='Date', how='outer', suffixes=('', '_new'))

                    # Für jede Spalte die *_new-Spalte übernehmen, falls vorhanden
                    for col in ["Event_Type", "Close_Price", "Distance_Upper", "Distance_Lower", "Distance_Middle",
                                "Consecutive_Touches", "Touch_Type", "Future_Return_1", "Future_Return_5",
                                "Future_Move_1", "Future_Move_5"]:
                        new_col = f"{col}_new"
                        if new_col in updated_df.columns:
                            # Fülle leere Werte in der alten Spalte mit den neuen Werten
                            updated_df[col] = updated_df[col].fillna(updated_df[new_col])
                            updated_df.drop(columns=[new_col], inplace=True)

                    # Optional: Filtere noch auf Rows, die tatsächlich ein Event haben.
                    updated_df = updated_df[updated_df["Event_Type"].notna()]

                    # Sortieren nach Datum
                    updated_df.sort_values(by="Date", inplace=True)

                    # Nur die gewünschten Spalten behalten
                    updated_df = updated_df[[
                        'Date', 'Event_Type', 'Close_Price', 'Distance_Upper', 'Distance_Lower',
                        'Distance_Middle', 'Consecutive_Touches', 'Touch_Type',
                        'Future_Return_1', 'Future_Return_5', 'Future_Move_1', 'Future_Move_5'
                    ]]

                    # Entfernen von Duplikaten, falls notwendig
                    updated_df.drop_duplicates(inplace=True)

                    # Speichern
                    updated_df.to_csv(event_file, index=False)
                    print(f"{file_name} wurde aktualisiert: {event_file}")
                else:
                    print(f"{file_name} existiert nicht im Ausgabe-Verzeichnis: {event_file}")

    def _get_events_for_tolerance(self, tolerance: float):
        """
        Filtert die Events für eine bestimmte Toleranz und berechnet die notwendigen Spalten.
        Gibt ein DF mit allen Datumszeilen zurück. Nur bei vorhandenen Events sind Spalten gefüllt.
        """
        temp_df = self.df.copy()
        temp_df["Event_Type"] = None
        temp_df["Touch_Type"] = None
        temp_df["Tolerance"] = f"{tolerance*100}%"
        temp_df["Close_Price"] = temp_df["Close"]  # Kopie zur Klarheit

        # Toleranz-Checks: Berechnung basierend auf der Bandbreite
        for i in range(len(temp_df)):
            close_price = temp_df.at[i, "Close"]
            upper = temp_df.at[i, "UpperBand"]
            middle = temp_df.at[i, "MiddleBand"]
            lower = temp_df.at[i, "LowerBand"]

            # Berechnung der Bandbreite
            band_width = upper - lower

            # Toleranz-Werte (relativ zur Bandbreite)
            upper_tol = upper - (tolerance * band_width)
            lower_tol = lower + (tolerance * band_width)
            # Für Middle_Break, basierend auf Abstand zum MiddleBand
            upper_tol_mid = middle + (tolerance * (upper - middle))
            lower_tol_mid = middle - (tolerance * (middle - lower))

            # Event-Erkennung
            # Upper_Touch
            if close_price >= upper:
                temp_df.at[i, "Event_Type"] = "Upper_Touch"
                temp_df.at[i, "Touch_Type"] = "Real"
            elif upper_tol <= close_price < upper:
                temp_df.at[i, "Event_Type"] = "Upper_Touch"
                temp_df.at[i, "Touch_Type"] = "Apparent"
            # Lower_Touch
            elif close_price <= lower:
                temp_df.at[i, "Event_Type"] = "Lower_Touch"
                temp_df.at[i, "Touch_Type"] = "Real"
            elif lower < close_price <= lower_tol:
                temp_df.at[i, "Event_Type"] = "Lower_Touch"
                temp_df.at[i, "Touch_Type"] = "Apparent"
            # Middle_Break
            elif close_price > middle:
                if close_price > upper_tol_mid:
                    temp_df.at[i, "Event_Type"] = "Middle_Break_Up"
                    temp_df.at[i, "Touch_Type"] = "Real"
                else:
                    temp_df.at[i, "Event_Type"] = "Middle_Break_Up"
                    temp_df.at[i, "Touch_Type"] = "Apparent"
            elif close_price < middle:
                if close_price < lower_tol_mid:
                    temp_df.at[i, "Event_Type"] = "Middle_Break_Down"
                    temp_df.at[i, "Touch_Type"] = "Real"
                else:
                    temp_df.at[i, "Event_Type"] = "Middle_Break_Down"
                    temp_df.at[i, "Touch_Type"] = "Apparent"

        # Berechnung der Distanzen in %
        temp_df["Distance_Upper"] = ((temp_df["Close"] - temp_df["UpperBand"]) / temp_df["UpperBand"]) * 100
        temp_df["Distance_Middle"] = ((temp_df["Close"] - temp_df["MiddleBand"]) / temp_df["MiddleBand"]) * 100
        temp_df["Distance_Lower"] = ((temp_df["Close"] - temp_df["LowerBand"]) / temp_df["LowerBand"]) * 100

        # Future Returns & Moves
        for d in [1, 5]:
            ret_col = f"Future_Return_{d}"
            mov_col = f"Future_Move_{d}"
            temp_df[ret_col] = np.nan
            temp_df[mov_col] = None

            for i in range(len(temp_df) - d):
                current_close = temp_df.at[i, "Close"]
                future_close = temp_df.at[i + d, "Close"]
                if pd.notnull(current_close) and pd.notnull(future_close):
                    perc_change_decimal = (future_close - current_close) / current_close
                    temp_df.at[i, ret_col] = round(perc_change_decimal, 3)

            temp_df.loc[temp_df[ret_col] > 0, mov_col] = "Up"
            temp_df.loc[temp_df[ret_col] <= 0, mov_col] = "Down"

        # Berechnung Consecutive_Touches mit 2 Tagen Pause
        temp_df["Consecutive_Touches"] = 0
        current_event = None
        current_count = 0
        days_since_last_touch = 0

        for i in range(len(temp_df)):
            event_today = temp_df.at[i, "Event_Type"]

            if pd.isna(event_today):
                if current_event is not None:
                    days_since_last_touch += 1
                    if days_since_last_touch > 2:
                        current_event = None
                        current_count = 0
                temp_df.at[i, "Consecutive_Touches"] = 0
            else:
                if current_event is None:
                    current_event = event_today
                    current_count = 1
                    days_since_last_touch = 0
                else:
                    if event_today == current_event:
                        current_count += 1
                        days_since_last_touch = 0
                    else:
                        current_event = event_today
                        current_count = 1
                        days_since_last_touch = 0
                temp_df.at[i, "Consecutive_Touches"] = current_count

        return temp_df

    def run_all_fills(self):
        """
        Führt alle Fülloperationen für die vorhandenen CSV-Dateien aus.
        """
        # Fülle Distance_Data.csv
        self.fill_distance_data()
        # Fülle Touches_Combined.csv
        self.fill_touches_combined()
        # Fülle event-spezifische CSV-Dateien
        self.fill_event_specific_files()

def find_csv_file(file_name):
    """
    Durchsucht das Projektverzeichnis (drei Ebenen über diesem Skript) und alle Unterverzeichnisse
    nach einer Datei mit dem Namen file_name. Gibt den Pfad zurück, wenn gefunden.
    """
    project_dir = Path(__file__).resolve().parents[3]  # Geht drei Ebenen hoch – ggf. anpassen
    print(f"Projektverzeichnis: {project_dir}")  # Debugging-Ausgabe
    for file_path in project_dir.rglob(file_name):
        if file_path.is_file():
            print(f"Gefunden: {file_path}")  # Debugging-Ausgabe
            return file_path
    print(f"Datei '{file_name}' wurde im Projektverzeichnis nicht gefunden.")
    return None

def load_data(file_name):
    """
    Lädt die CSV-Datei mit parse_dates=['Date'] und gibt ein pandas DataFrame zurück.
    """
    csv_path = find_csv_file(file_name)
    if csv_path:
        print(f"Lade Daten von: {csv_path}")  # Debugging-Ausgabe
        return pd.read_csv(csv_path, parse_dates=['Date'])
    else:
        raise FileNotFoundError(f"Datei '{file_name}' nicht gefunden.")

def run_main():
    """
    Hauptfunktion, die den gesamten Prozess ausführt.
    """
    # Definiere das Ausgabe-Verzeichnis korrekt
    output_directory = Path(__file__).resolve().parent / "analysis_results"
    print(f"Ausgabe-Verzeichnis: {output_directory}")  # Debugging-Ausgabe

    try:
        # Lade die CSV-Dateien
        bollinger_df = load_data("bollinger_bands.csv")
        historical_df = load_data("SP500_Index_Historical_Data.csv")

        # Zusammenführen der DataFrames auf 'Date'
        merged_df = pd.merge(historical_df, bollinger_df, on='Date', how='inner', suffixes=('_hist', '_boll'))

        # Überprüfe, ob 'Close_hist' und 'Close_boll' gleich sind
        close_diff = (merged_df['Close_hist'] - merged_df['Close_boll']).abs()
        if not (close_diff <= 1e-6).all():
            print("Warnung: 'Close_hist' und 'Close_boll' sind nicht identisch.")
        else:
            print("Info: 'Close_hist' und 'Close_boll' sind identisch.")

        # Entferne die 'Close_boll'-Spalte, da sie redundant ist
        merged_df.drop(columns=['Close_boll'], inplace=True)

        # Benenne 'Close_hist' zu 'Close' um
        merged_df.rename(columns={'Close_hist': 'Close'}, inplace=True)

        # Benenne die Bollinger-Spalten um
        merged_df.rename(columns={
            'Moving_Avg': 'MiddleBand',
            'Upper_Band': 'UpperBand',
            'Lower_Band': 'LowerBand'
        }, inplace=True)

        # Debug: Überprüfe die finalen Spaltennamen
        print("Finale Spalten nach dem Umbenennen:")
        print(merged_df.columns.tolist())

        # Instanz der Klasse erstellen
        filler = BollingerBandsFiller(merged_df, output_directory)

        # Führe alle Fülloperationen aus
        filler.run_all_fills()

    except FileNotFoundError as e:
        print(f"Fehler: {e}")
    except KeyError as e:
        print(f"Fehler beim Zugriff auf eine Spalte: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    run_main()
