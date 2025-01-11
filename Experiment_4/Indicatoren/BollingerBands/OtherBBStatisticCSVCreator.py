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
    # 1. Daten laden
    df_bbands = load_data('bollinger_bands_2.csv')  # Enthält: Date, Close, Moving_Avg, Upper_Band, Lower_Band
    df_sp500 = load_data('SP500_Index_Historical_Data.csv')  # Enthält: Date, Open, High, Low, Close, Volume

    # 2. Mergen beider DataFrames (auf 'Date'), um spätere Nutzung der SP500-Daten zu ermöglichen
    #    'inner' oder 'outer' je nach Bedarf – hier 'inner' als Beispiel
    df_merged = pd.merge(df_bbands, df_sp500, on='Date', how='inner', suffixes=('_bb', '_sp500'))

    # 3. Sortieren nach Datum, damit das Shift für Slope und Divergence stimmt
    df_merged.sort_values(by='Date', inplace=True)

    # 4. Bandwidth (Breite der Bollinger Bänder)
    """
    Bandwidth (Breite der Bollinger Bänder)

    Beschreibung: Die Bandbreite ist der relative Abstand zwischen dem oberen und unteren Bollinger-Band, berechnet als:
        Bandwidth = (Upper_Band - Lower_Band) / Moving_Avg

    Bedeutung: Sie gibt an, wie volatil der Markt ist. Eine größere Bandbreite deutet auf höhere Volatilität hin, während eine kleinere auf eine Seitwärtsbewegung schließen lässt.

    Nutzung:
       Bandbreite als Feature im LSTM.
    """
    df_merged['Bandwidth'] = (df_merged['Upper_Band'] - df_merged['Lower_Band']) / df_merged['Moving_Avg']

    # 5. %B (Prozentuale Position innerhalb der Bänder)
    """
    %B (Prozentuale Position innerhalb der Bänder)

    Beschreibung: %B misst, wo sich der Kurs relativ zu den Bollinger-Bändern befindet, berechnet als:
        %B = (Close - Lower_Band) / (Upper_Band - Lower_Band)

    Bedeutung: Werte nahe 1 zeigen, dass der Kurs am oberen Band ist, Werte nahe 0 deuten auf das untere Band hin.

    Nutzung:
        Als eigenständiges Feature im LSTM.
    """
    df_merged['Percent_B'] = (df_merged['Close_bb'] - df_merged['Lower_Band']) / (
            df_merged['Upper_Band'] - df_merged['Lower_Band']
    )

    # 6. Preisabweichung vom Mittelband
    """
    Preisabweichung vom Mittelband

    Beschreibung: Differenz zwischen dem Schlusskurs und dem Mittelband:
        Price_Deviation = Close - Moving_Avg

    Bedeutung: Positive Werte zeigen an, dass der Kurs über dem Mittelband liegt (bullish), negative Werte deuten auf eine Schwäche hin (bearish).

    Nutzung:
        Direkte Integration als numerisches Feature.
    """
    df_merged['Price_Deviation'] = df_merged['Close_bb'] - df_merged['Moving_Avg']

    # 7. Trendrichtung der Bänder (Slope)
    """
    Trendrichtung der Bänder (Slope)

    Beschreibung: Untersuche die Steigung des Mittelbands über Zeitfenster:
        Slope[t] = Moving_Avg[t] - Moving_Avg[t-1]

    Bedeutung: Zeigt die generelle Trendrichtung (aufwärts, abwärts, seitwärts).

    Nutzung:
        Füge die Steigung als Feature ins LSTM ein.
    """
    df_merged['Slope'] = df_merged['Moving_Avg'].diff()

    # 8. Divergenz von Bollinger-Bändern
    """
    Divergenz von Bollinger-Bändern

    Beschreibung: Untersuche, wie weit sich die Bollinger-Bänder im Vergleich zu einem vorherigen Zeitpunkt auseinander- oder zusammengezogen haben:
        Divergence = (Bandwidth[t] - Bandwidth[t-1]) / Bandwidth[t-1]

    Bedeutung: Gibt Hinweise auf eine sich verstärkende oder abschwächende Volatilität.

    Nutzung:
        Als numerisches Feature für die Dynamik der Marktbewegung.
    """
    df_merged['Divergence'] = df_merged['Bandwidth'].diff() / df_merged['Bandwidth'].shift(1)
    # Optional: Falls NaN-Werte stören (z. B. erste Zeile), könnten diese aufgefüllt werden:
    df_merged['Divergence'].fillna(0, inplace=True)

    # Hinweis: Hier könnten weitere Statistiken folgen (z. B. Band Hits Frequency, Strength, etc.)

    # 9. Relevante Spalten auswählen
    """
    Auswahl der relevanten Spalten für die finale CSV.
    Beinhaltet die originalen Bollinger-Band-Daten sowie die neu berechneten Features.
    """
    final_cols = [
        'Date',
        'Close_bb', 'Moving_Avg', 'Upper_Band', 'Lower_Band',  # Original Bollinger-Spalten
        'Bandwidth', 'Percent_B', 'Price_Deviation', 'Slope', 'Divergence',
        # SP500-spezifische Spalten (optional je nach Bedarf)
        'Close_sp500', 'Volume'
    ]
    df_final = df_merged[final_cols]

    # 10. CSV speichern
    """
    Speichern der finalen CSV mit allen relevanten Bollinger-Features.
    Der Speicherpfad ist festgelegt auf:
        C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\BollingerBands\Final_Bollinger_Stats.csv
    """
    save_path = Path(
        r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\BollingerBands\Final_Bollinger_Stats.csv")
    df_final.to_csv(save_path, index=False)
    print(f"Datei erfolgreich erstellt unter: {save_path}")


if __name__ == "__main__":
    main()
