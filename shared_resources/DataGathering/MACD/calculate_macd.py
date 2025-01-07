import pandas as pd
import numpy as np
from pathlib import Path


def calculate_macd(data, span_short=12, span_long=26, span_signal=9):
    """
    Berechnet den MACD (Moving Average Convergence Divergence) für einen DataFrame.

    Parameters:
    data (pandas.DataFrame): DataFrame mit OHLCV Daten.
    span_short (int): Zeitraum für den kurzfristigen EMA (Standard: 12).
    span_long (int): Zeitraum für den langfristigen EMA (Standard: 26).
    span_signal (int): Zeitraum für die Signallinie (Standard: 9).

    Returns:
    pandas.DataFrame: Original DataFrame mit zusätzlichen MACD und Signal_Line Spalten.
    """
    df = data.copy()
    df['Close'] = df['Close'].astype(float)

    short_ema = df['Close'].ewm(span=span_short, adjust=False).mean()
    long_ema = df['Close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()

    return df


def main():
    # Dynamische Pfade basierend auf dem aktuellen Skriptverzeichnis
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]  # Gehe zwei Ebenen nach oben zum Hauptprojektverzeichnis

    input_path = project_dir / 'sp500_data' / 'SP500_Index_Historical_Data.csv'
    output_path = project_dir / 'sp500_data' / 'MACD' / 'SP500_Index_Historical_Data_with_MACD.csv'

    # Prüfen, ob die Eingabedatei existiert
    if not input_path.exists():
        raise FileNotFoundError(f"Die Datei {input_path.name} wurde nicht im Pfad {input_path} gefunden.")

    # CSV-Datei einlesen
    df = pd.read_csv(input_path)

    # MACD berechnen (Standard: span_short=12, span_long=26, span_signal=9)
    df_with_macd = calculate_macd(df)

    # Ausgabeordner erstellen, falls er nicht existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_macd.to_csv(output_path, index=False)

    print(f"MACD wurde berechnet und in {output_path} gespeichert.")


if __name__ == "__main__":
    main()
