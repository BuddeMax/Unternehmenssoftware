import pandas as pd
import numpy as np
from pathlib import Path


def calculate_bollinger_bands(data, periods=20, std_dev=2):
    """
    Berechnet die Bollinger Bands für einen DataFrame.

    Parameters:
    data (pandas.DataFrame): DataFrame mit OHLCV Daten.
    periods (int): Anzahl der Perioden für die Berechnung des gleitenden Durchschnitts (Standard: 20).
    std_dev (int): Anzahl der Standardabweichungen für die Bollinger Bands (Standard: 2).

    Returns:
    pandas.DataFrame: Original DataFrame mit zusätzlichen Spalten für BB_Middle, BB_Upper und BB_Lower.
    """
    df = data.copy()
    df['Close'] = df['Close'].astype(float)

    df['BB_Middle'] = df['Close'].rolling(window=periods).mean()
    df['BB_Std'] = df['Close'].rolling(window=periods).std()

    df['BB_Upper'] = df['BB_Middle'] + (std_dev * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * df['BB_Std'])

    return df.drop(columns=['BB_Std'])  # Die temporäre Spalte wird entfernt


def main():
    # Dynamische Pfade basierend auf dem aktuellen Skriptverzeichnis
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]  # Gehe zwei Ebenen nach oben zum Hauptprojektverzeichnis

    input_path = project_dir / 'sp500_data' / 'SP500_Index_Historical_Data.csv'
    output_path = project_dir / 'sp500_data' / 'BollingerBands' / 'SP500_Index_Historical_Data_with_BB.csv'

    # Prüfen, ob die Eingabedatei existiert
    if not input_path.exists():
        raise FileNotFoundError(f"Die Datei {input_path.name} wurde nicht im Pfad {input_path} gefunden.")

    # CSV-Datei einlesen
    df = pd.read_csv(input_path)

    # Bollinger Bands berechnen (Standard: 20 Perioden, 2 Standardabweichungen)
    df_with_bb = calculate_bollinger_bands(df)

    # Ausgabeordner erstellen, falls er nicht existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_bb.to_csv(output_path, index=False)

    print(f"Bollinger Bands wurden berechnet und in {output_path} gespeichert.")


if __name__ == "__main__":
    main()
