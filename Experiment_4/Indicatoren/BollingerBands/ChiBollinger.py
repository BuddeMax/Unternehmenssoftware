import pandas as pd
from pathlib import Path
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic


# --------------------------------------------------------
# Hilfsfunktionen zum Finden und Laden der CSV-Dateien
# --------------------------------------------------------
def find_csv_file(file_name):
    """
    Durchsucht das Projektverzeichnis (drei Ebenen über diesem Skript) und alle Unterordner
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

    # Aufräumen
    df.drop(columns=['Next_Consecutive_Touches'], inplace=True, errors='ignore')
    last_touches.drop(columns=['Next_Consecutive_Touches'], inplace=True, errors='ignore')

    return last_touches


# --------------------------------------------------------
# Funktion zur Durchführung des Chi-Quadrat-Tests
# --------------------------------------------------------
def perform_chi_square_test(up_count, down_count, expected_up=0.5, expected_down=0.5):
    """
    Führt einen Chi-Quadrat-Goodness-of-Fit-Test durch.

    :param up_count: Anzahl der Up-Bewegungen
    :param down_count: Anzahl der Down-Bewegungen
    :param expected_up: Erwartete Häufigkeit für Up (Standard: 0.5)
    :param expected_down: Erwartete Häufigkeit für Down (Standard: 0.5)
    :return: chi2, p
    """
    total = up_count + down_count
    expected = [expected_up * total, expected_down * total]
    chi2, p = chisquare(f_obs=[up_count, down_count], f_exp=expected)
    return chi2, p


# --------------------------------------------------------
# Berechnung von Chi²max
# --------------------------------------------------------
def calculate_chi2_max(up_count, down_count):
    """
    Berechnet den maximal möglichen Chi²-Wert basierend auf den beobachteten Häufigkeiten.
    Für eine binäre Verteilung bei erwarteter 50-50-Verteilung ist Chi²max gleich der Gesamtzahl der Beobachtungen.

    :param up_count: Anzahl der Up-Bewegungen
    :param down_count: Anzahl der Down-Bewegungen
    :return: Maximaler Chi²-Wert
    """
    total = up_count + down_count
    return total


# --------------------------------------------------------
# Funktion zum Plotten der Kontingenztabelle als Mosaikplot
# --------------------------------------------------------
def plot_mosaic(contingency, file_name, future_move_col):
    """
    Plottet die Kontingenztabelle als Mosaikplot.

    :param contingency: Pandas Crosstab DataFrame
    :param file_name: Name der CSV-Datei (für den Titel)
    :param future_move_col: Zukunftige Bewegungs-Spalte (z.B. Future_Move_1)
    """
    # Konvertiere die Kontingenztabelle in ein Dictionary mit Tupel-Schlüsseln
    contingency_dict = contingency.to_dict(orient='index')
    mosaic_dict = {}
    for row_key, row in contingency_dict.items():
        for col_key, value in row.items():
            mosaic_dict[(row_key, col_key)] = value

    plt.figure(figsize=(8, 6))
    mosaic(
        mosaic_dict,
        title=f'Mosaikplot: {file_name} | {future_move_col}',
        properties=lambda key: {'color': 'green' if key[1] == 'Up' else 'red'},
        labelizer=lambda key: str(mosaic_dict[key])
    )
    plt.tight_layout()

    # Speichern der Abbildung
    plot_filename = f"{file_name.replace('.csv', '')}_{future_move_col}_mosaic.png"
    plt.savefig(plot_filename)
    plt.close()


# --------------------------------------------------------
# Funktion zum Drucken der Kontingenztabelle in der Konsole
# --------------------------------------------------------
def print_contingency_table(contingency):
    """
    Druckt die Kontingenztabelle mit Counts und Prozenten in der Konsole.
    Die Prozentwerte sind über die gesamte Tabelle normalisiert, sodass sie insgesamt 100% ergeben.

    :param contingency: Pandas Crosstab DataFrame
    """
    total = contingency.values.sum()
    contingency_percent = contingency / total * 100

    # Kombiniere Counts und Prozent
    combined = contingency.astype(str) + " (" + contingency_percent.round(1).astype(str) + "%)"

    print("\nKontingenztabelle (Counts and Percentages):")
    print(combined.to_string())


# --------------------------------------------------------
# Hauptlogik für den Chi-Quadrat-Test und Plotting
# --------------------------------------------------------
def calculate_chi_square_and_plot():
    """
    Lädt alle relevanten CSV-Dateien und berechnet den Chi-Quadrat-Test,
    indem die Häufigkeit von Up und Down nach einer Berührung überprüft wird.
    Ignoriert den Touch_Type und fasst Real und Apparent zusammen.
    Erstellt Mosaikplots und druckt die Kontingenztabellen in der Konsole.
    """

    # Liste aller Dateien, die untersucht werden sollen.
    csv_files = [
        "Lower_Touch_0.25%.csv", "Lower_Touch_0.375%.csv", "Lower_Touch_0.5%.csv",
        "Lower_Touch_0.75%.csv", "Lower_Touch_1%.csv",

        "Middle_Break_Down_0.25%.csv", "Middle_Break_Down_0.375%.csv", "Middle_Break_Down_0.5%.csv",
        "Middle_Break_Down_0.75%.csv", "Middle_Break_Down_1%.csv",

        "Middle_Break_Up_0.25%.csv", "Middle_Break_Up_0.375%.csv", "Middle_Break_Up_0.5%.csv",
        "Middle_Break_Up_0.75%.csv", "Middle_Break_Up_1%.csv",

        "Upper_Touch_0.25%.csv", "Upper_Touch_0.375%.csv", "Upper_Touch_0.5%.csv",
        "Upper_Touch_0.75%.csv", "Upper_Touch_1%.csv",
    ]

    future_move_col = "Future_Move_1"  # Nur Future_Move_1 betrachten

    for file_name in csv_files:
        df = load_data(file_name)
        if df is None or df.empty:
            print(f"\n===== Datei '{file_name}' konnte nicht geladen werden oder ist leer. Überspringe. =====")
            continue

        # Identifiziere die letzten Berührungspunkte
        last_touches = identify_last_touches(df)

        if last_touches.empty:
            print(f"\n===== Datei '{file_name}' enthält keine letzten Berührungspunkte. Überspringe. =====")
            continue

        # Zähle die Anzahl von Up und Down
        move_counts = last_touches[future_move_col].value_counts()
        up_count = move_counts.get("Up", 0)
        down_count = move_counts.get("Down", 0)
        total = up_count + down_count

        if total == 0:
            print(f"\n===== Datei '{file_name}' enthält keine gültigen Future_Move_1 Werte. Überspringe. =====")
            continue

        # Führe den Chi-Quadrat-Goodness-of-Fit-Test durch
        chi2, p = perform_chi_square_test(up_count, down_count)

        # Berechne Chi²max
        chi2_max = calculate_chi2_max(up_count, down_count)

        # Berechne die Chi²-normalisiert
        chi2_normalized = chi2 / chi2_max if chi2_max != 0 else 0

        # Berechne die prozentuale Verteilung (über die gesamte Tabelle)
        up_percent = (up_count / total) * 100
        down_percent = (down_count / total) * 100

        # Ausgabe der Ergebnisse
        print(f"\n===== Chi-Quadrat-Test für Datei: {file_name} | {future_move_col} =====")
        print(f"Beobachtete Häufigkeiten: Up = {up_count}, Down = {down_count}")
        print(f"Prozentuale Verteilung: Up = {up_percent:.2f}%, Down = {down_percent:.2f}%")
        print(f"Chi² = {chi2:.4f}, Chi²max = {chi2_max:.4f}")
        print(f"Chi²-normalisiert = {chi2_normalized:.4f}")

        # Erstelle eine Kontingenztabelle (Touch_Type vs. Future_Move_1)
        contingency = pd.crosstab(last_touches["Touch_Type"], last_touches[future_move_col])

        # Drucke die Kontingenztabelle in der Konsole
        print_contingency_table(contingency)

        # Plot der Kontingenztabelle als Mosaikplot
        try:
            plot_mosaic(contingency, file_name, future_move_col)
        except Exception as e:
            print(f"[DEBUG] Fehler beim Plotten der Mosaikplot für '{file_name}': {e}")


# --------------------------------------------------------
# Hauptfunktion
# --------------------------------------------------------
def main():
    calculate_chi_square_and_plot()


if __name__ == "__main__":
    main()
