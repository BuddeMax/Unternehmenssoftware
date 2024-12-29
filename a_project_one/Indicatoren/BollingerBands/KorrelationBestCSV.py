import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------------
# Hilfsfunktionen zum Finden und Laden der CSV-Dateien
# --------------------------------------------------------

def find_csv_file(file_name):
    """
    Durchsucht das Projektverzeichnis (drei Ebenen über diesem Skript) und alle Unterverzeichnisse
    nach einer Datei mit dem Namen file_name. Gibt den Pfad zurück, wenn gefunden.
    """
    project_dir = Path(__file__).resolve().parents[3]  # Geht drei Ebenen hoch – ggf. anpassen
    for file_path in project_dir.rglob(file_name):
        if file_path.is_file():
            return file_path
    print(f"[DEBUG] Datei '{file_name}' wurde im Projektverzeichnis nicht gefunden.")
    return None


def load_data(file_name):
    """
    Lädt die CSV-Datei mit parse_dates=['Date'] und gibt ein pandas DataFrame zurück.
    """
    csv_path = find_csv_file(file_name)
    if csv_path:
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            if df.empty:
                print(f"[DEBUG] Warnung: Die Datei '{file_name}' ist leer oder konnte nicht korrekt geladen werden.")
            return df
        except Exception as e:
            print(f"[DEBUG] Fehler beim Laden der Datei '{file_name}': {e}")
            return None
    else:
        print(f"[DEBUG] Datei '{file_name}' wurde nicht gefunden.")
        return None


# --------------------------------------------------------
# Funktion zur Bestimmung der relevanten Distanz-Spalte basierend auf dem Dateinamen
# --------------------------------------------------------

def get_relevant_distance_column(file_name):
    """
    Gibt die relevante Distanz-Spalte basierend auf dem Dateinamen zurück.
    """
    if "Upper_Touch" in file_name:
        return "Distance_Upper"
    elif "Lower_Touch" in file_name:
        return "Distance_Lower"
    elif "Middle_Break_Up" in file_name or "Middle_Break_Down" in file_name:
        return "Distance_Middle"
    else:
        return None  # Für nicht relevante Dateien wie Touches_Combined.csv oder Distance_Data.csv


# --------------------------------------------------------
# Funktion zur Identifikation der letzten Berührungspunkte
# --------------------------------------------------------

def identify_last_touches(df):
    """
    Identifiziert die letzten Berührungspunkte in der DataFrame.
    Eine Zeile ist ein letzter Berührungspunkt, wenn die Consecutive_Touches
    größer sind als in der nächsten Zeile oder wenn es keine nächste Zeile gibt.
    """
    # Sortiere nach Datum
    df = df.sort_values('Date').reset_index(drop=True)

    # Shift Consecutive_Touches nach oben, um die nächste Zeile zu erhalten
    df['Next_Consecutive_Touches'] = df['Consecutive_Touches'].shift(-1)

    # Markiere Zeilen, die letzte Berührungspunkte sind
    last_touch_mask = (df['Consecutive_Touches'] > df['Next_Consecutive_Touches']) | (
        df['Next_Consecutive_Touches'].isna())

    # Filtere nur die letzten Berührungspunkte
    last_touches = df[last_touch_mask].copy()

    return last_touches


# --------------------------------------------------------
# Hauptlogik für die Korrelationsanalyse
# --------------------------------------------------------

def calculate_correlations_for_files():
    """
    Lädt alle relevanten CSV-Dateien und berechnet Pearson- und Spearman-Korrelationen
    zwischen der relevanten Distanz-Spalte und den Future_Return-Spalten.
    Gibt das Ergebnis in der Konsole aus.
    """

    # Liste aller Dateien, die untersucht werden sollen.
    # Hier NUR die Dateinamen (ohne Pfade). Sie müssen über find_csv_file gefunden werden.
    csv_files = [
        "Distance_Data.csv",  # Wird übersprungen, da keine Future_Return-Spalten

        "Lower_Touch_0.25%.csv", "Lower_Touch_0.375%.csv", "Lower_Touch_0.5%.csv",
        "Lower_Touch_0.75%.csv", "Lower_Touch_1%.csv",

        "Middle_Break_Down_0.25%.csv", "Middle_Break_Down_0.375%.csv", "Middle_Break_Down_0.5%.csv",
        "Middle_Break_Down_0.75%.csv", "Middle_Break_Down_1%.csv",

        "Middle_Break_Up_0.25%.csv", "Middle_Break_Up_0.375%.csv", "Middle_Break_Up_0.5%.csv",
        "Middle_Break_Up_0.75%.csv", "Middle_Break_Up_1%.csv",

        "Upper_Touch_0.25%.csv", "Upper_Touch_0.375%.csv", "Upper_Touch_0.5%.csv",
        "Upper_Touch_0.75%.csv", "Upper_Touch_1%.csv",

        "Touches_Combined.csv"  # Wird übersprungen, da mehrere Berührungstypen
    ]

    # Diese Spalten (falls vorhanden) sind unsere "Zielspalten" (Y):
    future_return_columns = ["Future_Return_1", "Future_Return_5"]

    # Iteration über alle Dateien
    for file_name in csv_files:
        df = load_data(file_name)
        if df is None:
            # Datei konnte nicht geladen werden oder ist leer → weiter zur nächsten
            continue

        # Bestimme die relevante Distanz-Spalte basierend auf dem Dateinamen
        relevant_distance_col = get_relevant_distance_column(file_name)

        if relevant_distance_col is None:
            # Datei ist entweder nicht relevant oder enthält mehrere Berührungstypen → überspringen
            print(
                f"\n===== Datei '{file_name}' ist nicht relevant für einzelne Berührungstypen oder enthält mehrere Berührungstypen. Korrelationen werden nicht berechnet =====")
            continue

        # Prüfen, ob die Datei die Future_Return-Spalten enthält
        has_future_returns = all(col in df.columns for col in future_return_columns)

        if not has_future_returns:
            # Datei enthält keine Future_Return-Spalten (z.B. Distance_Data.csv) → überspringen
            print(
                f"\n===== Datei '{file_name}' enthält keine Future_Return-Spalten. Korrelationen werden nicht berechnet =====")
            continue

        # Identifiziere die letzten Berührungspunkte
        last_touches = identify_last_touches(df)

        if last_touches.empty:
            print(
                f"\n===== Datei '{file_name}' enthält keine letzten Berührungspunkte mit gültigen Future_Return-Werten =====")
            continue

        # Entferne Zeilen ohne Future_Return-Werte
        last_touches = last_touches.dropna(subset=future_return_columns)

        if last_touches.empty:
            print(
                f"\n===== Datei '{file_name}' enthält keine letzten Berührungspunkte mit gültigen Future_Return-Werten =====")
            continue

        # Extrahiere die relevante Distanz-Spalte
        if relevant_distance_col not in last_touches.columns:
            print(
                f"\n[DEBUG] Relevante Distanz-Spalte '{relevant_distance_col}' existiert nicht in Datei '{file_name}'.")
            continue

        # Iteriere nur über die Future_Return-Spalten
        print(f"\n===== Korrelationsergebnisse für Datei: {file_name} =====")
        for fut_col in future_return_columns:
            if fut_col not in last_touches.columns:
                print(f"[DEBUG] Spalte '{fut_col}' existiert nicht in '{file_name}'.")
                continue

            # Prüfe, ob die Spalten numerisch sind
            if not pd.api.types.is_numeric_dtype(last_touches[relevant_distance_col]):
                print(
                    f"[DEBUG] Spalte '{relevant_distance_col}' in Datei '{file_name}' ist nicht numerisch. Überspringe.")
                continue
            if not pd.api.types.is_numeric_dtype(last_touches[fut_col]):
                print(f"[DEBUG] Spalte '{fut_col}' in Datei '{file_name}' ist nicht numerisch. Überspringe.")
                continue

            # Entferne Zeilen mit NaN-Werten in den relevanten Spalten
            valid_data = last_touches[[relevant_distance_col, fut_col]].dropna()
            if valid_data.empty:
                print(
                    f"[DEBUG] Keine validen Werte für '{relevant_distance_col}' und '{fut_col}' in Datei '{file_name}'.")
                continue

            # Pearson-Korrelation
            try:
                pearson_corr, pearson_pvalue = pearsonr(valid_data[relevant_distance_col], valid_data[fut_col])
            except Exception as e:
                print(
                    f"[DEBUG] Fehler bei Pearson-Korrelation zwischen '{relevant_distance_col}' und '{fut_col}' in Datei '{file_name}': {e}")
                pearson_corr, pearson_pvalue = (None, None)

            # Spearman-Korrelation
            try:
                spearman_corr, spearman_pvalue = spearmanr(valid_data[relevant_distance_col], valid_data[fut_col])
            except Exception as e:
                print(
                    f"[DEBUG] Fehler bei Spearman-Korrelation zwischen '{relevant_distance_col}' und '{fut_col}' in Datei '{file_name}': {e}")
                spearman_corr, spearman_pvalue = (None, None)

            # Ausgabe der Ergebnisse
            if pearson_corr is not None and spearman_corr is not None:
                print(f"-> {relevant_distance_col} vs. {fut_col}")
                print(f"   Pearson:  r = {pearson_corr:.4f}")
                print(f"   Spearman: r = {spearman_corr:.4f}")
            else:
                print(
                    f"[DEBUG] Korrelation konnte für '{relevant_distance_col}' vs. '{fut_col}' in Datei '{file_name}' nicht berechnet werden.")


if __name__ == "__main__":
    calculate_correlations_for_files()
