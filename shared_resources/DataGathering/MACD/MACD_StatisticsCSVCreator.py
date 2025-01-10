import pandas as pd
from pathlib import Path
import numpy as np


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


def main():
    """
    Hauptfunktion zur Verarbeitung der MACD-Daten und Erstellung der neuen CSV mit zusätzlichen Features.
    """
    # 1. CSV-Datei laden
    #    Hier wird nach der Datei 'SP500_Index_Historical_Data_with_RSI_MACD.csv' im Projekt gesucht.
    df = load_data('SP500_Index_Historical_Data_with_RSI_MACD.csv')

    # 2. DataFrame nach Datum sortieren (wichtig für alle zeitabhängigen Berechnungen)
    df.sort_values(by='Date', inplace=True)

    # 3. MACD-Histogramm erstellen
    #    MACD-Histogramm = MACD - Signal_Line
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # 4. Kreuzungspunkte (Crossovers) bestimmen:
    #    - 1, wenn MACD von unten nach oben durch Signal kreuzt (bullisch)
    #    - -1, wenn MACD von oben nach unten durch Signal kreuzt (bärisch)
    #    - 0, sonst
    df['MACD_Crossover'] = 0
    for i in range(1, len(df)):
        prev_macd = df.iloc[i - 1]['MACD']
        prev_signal = df.iloc[i - 1]['Signal_Line']
        curr_macd = df.iloc[i]['MACD']
        curr_signal = df.iloc[i]['Signal_Line']

        # Bullische Kreuzung
        if prev_macd < prev_signal and curr_macd > curr_signal:
            df.at[df.index[i], 'MACD_Crossover'] = 1

        # Bärische Kreuzung
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            df.at[df.index[i], 'MACD_Crossover'] = -1

    # 5. Abstand zwischen MACD- und Signal-Linie (MACD Gap)
    #    MACD_Gap = MACD - Signal_Line
    df['MACD_Gap'] = df['MACD_Histogram']

    # 6. Histogramm-Dynamik:
    #    MACD_Histogram_Change = MACD_Histogram[t] - MACD_Histogram[t-1]
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()

    # 7. Stärke der Kreuzung:
    #    = |MACD - Signal_Line| am Tag der Kreuzung (ansonsten 0)
    df['MACD_Crossover_Strength'] = 0
    crossover_indices = df[df['MACD_Crossover'] != 0].index
    df.loc[crossover_indices, 'MACD_Crossover_Strength'] = (
                df.loc[crossover_indices, 'MACD'] - df.loc[crossover_indices, 'Signal_Line']).abs()

    # 8. MACD-Trend (Slope) über ein gleitendes Fenster n=5
    #    MACD_Slope_5 = MACD[t] - MACD[t-5]
    n = 5
    df['MACD_Slope_5'] = df['MACD'] - df['MACD'].shift(n)
    df['MACD_Slope_5'].fillna(0, inplace=True)

    # 9. Divergenzen zwischen MACD und Kurs (Close):
    #    Wenn MACD und Close in entgegengesetzte Richtungen gehen (tagesbasierter Vergleich)
    df['Close_Change'] = df['Close'].diff()
    df['MACD_Change'] = df['MACD'].diff()

    def check_divergence(row):
        # Divergenz, wenn Vorzeichen unterschiedlich:
        if pd.notnull(row['Close_Change']) and pd.notnull(row['MACD_Change']):
            if row['Close_Change'] * row['MACD_Change'] < 0:
                return 1
        return 0

    df['MACD_Price_Divergence'] = df.apply(check_divergence, axis=1)
    # Aufräumen
    df.drop(columns=['Close_Change', 'MACD_Change'], inplace=True)

    # 10. Extremwerte des MACD:
    #     Identifikation von Extremwerten über 2 * std in einem 20-Tage-Fenster.
    window_extreme = 20
    df['MACD_RollingMean'] = df['MACD'].rolling(window_extreme).mean()
    df['MACD_RollingStd'] = df['MACD'].rolling(window_extreme).std()

    df['MACD_Extreme'] = 0
    # +2 STD
    df.loc[df['MACD'] > df['MACD_RollingMean'] + 2 * df['MACD_RollingStd'], 'MACD_Extreme'] = 1
    # -2 STD
    df.loc[df['MACD'] < df['MACD_RollingMean'] - 2 * df['MACD_RollingStd'], 'MACD_Extreme'] = -1

    # 11. Durchschnittliche MACD-Änderung (Beispiel: n=5)
    #     MACD_Mean_Change_5 = Mittelwert der täglichen Differenzen in einem 5er-Fenster
    macd_diff = df['MACD'].diff()
    df['MACD_Mean_Change_5'] = macd_diff.rolling(5).mean().fillna(0)

    # Aufräumen von Hilfsspalten
    df.drop(columns=['MACD_RollingMean', 'MACD_RollingStd'], inplace=True)

    # 12. Finale Spaltenauswahl
    #     Wir wollen NICHT die RSI-Spalte übernehmen. Nur:
    #     Date, Open, High, Low, Close, Volume, MACD, Signal_Line,
    #     plus unsere neuen Spalten.
    final_cols = [
        'Date',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'MACD',
        'Signal_Line',
        'MACD_Histogram',
        'MACD_Crossover',
        'MACD_Gap',
        'MACD_Histogram_Change',
        'MACD_Crossover_Strength',
        'MACD_Slope_5',
        'MACD_Price_Divergence',
        'MACD_Extreme',
        'MACD_Mean_Change_5'
    ]
    df_final = df[final_cols].copy()

    # 13. CSV speichern:
    """
    Speichern der finalen CSV mit allen relevanten MACD-Features.
    Diese soll ins Projektverzeichnis:
    'Unternehmenssoftware' > 'sp500_data' > 'MACD'.
    Der Code soll auf jedem Rechner funktionieren, deshalb mit Path-Logik.
    """
    # Bestimmen des Basisverzeichnisses (drei Ebenen hoch)
    base_dir = Path(__file__).resolve().parents[3]
    # Navigieren zum vorhandenen 'sp500_data/MACD' Verzeichnis
    save_dir = base_dir / 'sp500_data' / 'MACD'

    if not save_dir.exists():
        raise FileNotFoundError(f"Speicherverzeichnis '{save_dir}' existiert nicht.")

    # Definieren des Speicherpfads
    save_path = save_dir / 'Final_MACD_Stats.csv'

    # Speichern der CSV
    df_final.to_csv(save_path, index=False)
    print(f"Datei erfolgreich erstellt unter: {save_path}")


if __name__ == "__main__":
    main()
