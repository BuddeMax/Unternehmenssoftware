import pandas as pd
import numpy as np
from pathlib import Path


def calculate_rsi(data, periods=14):
    """
    Berechnet den RSI (Relative Strength Index) für einen DataFrame.

    Parameters:
    data (pandas.DataFrame): DataFrame mit OHLCV Daten.
    periods (int): Anzahl der Perioden für die RSI-Berechnung (Standard: 14).

    Returns:
    pandas.DataFrame: Original DataFrame mit zusätzlicher RSI-Spalte.
    """
    df = data.copy()
    df['Close'] = df['Close'].astype(float)
    delta = df['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / periods, min_periods=periods).mean()
    avg_loss = loss.ewm(alpha=1 / periods, min_periods=periods).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def main():
    # Dynamische Pfade basierend auf dem aktuellen Skriptverzeichnis
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]  # Gehe zwei Ebenen nach oben zum Hauptprojektverzeichnis

    input_path = project_dir / 'sp500_data' / 'SP500_Index_Historical_Data.csv'
    output_path = project_dir / 'sp500_data' / 'RSI' / 'SP500_Index_Historical_Data_with_RSI.csv'

    # Prüfen, ob die Eingabedatei existiert
    if not input_path.exists():
        raise FileNotFoundError(f"Die Datei {input_path.name} wurde nicht im Pfad {input_path} gefunden.")

    # CSV-Datei einlesen
    df = pd.read_csv(input_path)

    # RSI berechnen (Standard: 14 Perioden)
    df_with_rsi = calculate_rsi(df)

    # Ausgabeordner erstellen, falls er nicht existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_rsi.to_csv(output_path, index=False)

    print(f"RSI wurde berechnet und in {output_path} gespeichert.")


if __name__ == "__main__":
    main()
