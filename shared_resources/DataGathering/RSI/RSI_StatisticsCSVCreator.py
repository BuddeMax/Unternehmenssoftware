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
    # 1. CSV-Datei laden
    #    Hier wird nach der Datei 'SP500_Index_Historical_Data_with_RSI_MACD.csv' im Projekt gesucht.
    df = load_data('SP500_Index_Historical_Data_with_RSI_MACD.csv')

    # 2. DataFrame nach Datum sortieren (wichtig für alle zeitabhängigen Berechnungen)
    df.sort_values(by='Date', inplace=True)

    # 3. RSI-DwellTime (Überschreitungsdauer) für RSI>70 und RSI<30 berechnen
    #    Wir erzeugen hier zwei Spalten:
    #      - RSI_DwellTime_Over70: Wie viele Tage in Folge RSI > 70
    #      - RSI_DwellTime_Under30: Wie viele Tage in Folge RSI < 30
    df['RSI_DwellTime_Over70'] = 0
    df['RSI_DwellTime_Under30'] = 0

    # Um die Überschreitungsdauer zu berechnen, iterieren wir durch den DataFrame
    over70_count = 0
    under30_count = 0
    for i in range(len(df)):
        rsi_value = df.iloc[i]['RSI']

        if rsi_value > 70:
            over70_count += 1
        else:
            over70_count = 0

        if rsi_value < 30:
            under30_count += 1
        else:
            under30_count = 0

        df.at[df.index[i], 'RSI_DwellTime_Over70'] = over70_count
        df.at[df.index[i], 'RSI_DwellTime_Under30'] = under30_count

    # 4. RSI-Trend (Slope) – Beispielhaft über ein festes Zeitfenster n=5
    #    RSI_Slope[t] = RSI[t] - RSI[t-5]
    n = 5
    df['RSI_Slope_5'] = df['RSI'] - df['RSI'].shift(n)
    # NaN-Werte am Anfang auffüllen (z. B. mit 0)
    df['RSI_Slope_5'].fillna(0, inplace=True)

    # 5. Relative Nähe zu 50 (Mean Reversion Signal)
    #    Distance to 50 = |RSI - 50|
    df['RSI_DistanceTo50'] = (df['RSI'] - 50).abs()

    # 6. (Einfache) Divergenz (RSI Divergence)
    #    Hier als binäres Signal, wenn RSI und Preis in entgegengesetzte Richtung gehen
    #    Beispielhaft über n=1: vergleicht die Änderung von RSI und Close zum Vortag.
    df['Close_Change'] = df['Close'] - df['Close'].shift(1)
    df['RSI_Change'] = df['RSI'] - df['RSI'].shift(1)

    def check_divergence(row):
        # Divergenz, wenn Vorzeichen unterschiedlich: (Close_Change * RSI_Change < 0)
        if row['Close_Change'] * row['RSI_Change'] < 0:
            return 1
        else:
            return 0

    df['RSI_Divergence'] = df.apply(check_divergence, axis=1)
    # Aufräumen (könnte man beibehalten, wenn man die Changes später braucht)
    df.drop(columns=['Close_Change', 'RSI_Change'], inplace=True)

    # 7. Häufigkeit der Grenzüberschreitungen (z. B. in den letzten 14 Tagen)
    #    Rolling-Summe, wie oft RSI > 70 oder RSI < 30 in einem 14-Tage-Fenster war
    window_size = 14
    df['RSI_OverboughtCount_14'] = df['RSI'].gt(70).rolling(window_size).sum()
    df['RSI_OversoldCount_14'] = df['RSI'].lt(30).rolling(window_size).sum()

    # 8. RSI-Korrelation mit Kursbewegungen (z. B. rollende Korrelation über 14 Tage)
    #    Wir nehmen hier die rollende Korrelation zwischen RSI und Close
    df['RSI_Close_Corr_14'] = df['RSI'].rolling(window_size).corr(df['Close'])

    # 9. RSI-Volatilität (Standardabweichung über 14 Tage)
    df['RSI_Volatility_14'] = df['RSI'].rolling(window_size).std()

    # 10. RSI-Stärke und Häufigkeit von Extremen (z. B. RSI > 80 oder RSI < 20),
    #     wiederum in einem 14-Tage-Fenster
    df['RSI_ExtremeCount_14'] = ((df['RSI'] > 80) | (df['RSI'] < 20)).rolling(window_size).sum()

    # Hinweis: Da wir keine Bollinger-Band-Daten hier einlesen, lassen wir das Thema "RSI-BB_Convergence" weg.

    # 11. Finale Spaltenauswahl
    #     Gewünscht: Date, Close, RSI + neu erzeugte Spalten
    final_cols = [
        'Date',
        'Close',
        'RSI',
        'RSI_DwellTime_Over70',
        'RSI_DwellTime_Under30',
        'RSI_Slope_5',
        'RSI_DistanceTo50',
        'RSI_Divergence',
        'RSI_OverboughtCount_14',
        'RSI_OversoldCount_14',
        'RSI_Close_Corr_14',
        'RSI_Volatility_14',
        'RSI_ExtremeCount_14'
    ]
    df_final = df[final_cols].copy()

    # 12. CSV speichern
    """
    Speichern der finalen CSV mit allen relevanten RSI-Features.
    """
    save_path = Path(
        r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\RSI\Final_RSI_Stats.csv"
    )
    df_final.to_csv(save_path, index=False)
    print(f"Datei erfolgreich erstellt unter: {save_path}")


if __name__ == "__main__":
    main()
