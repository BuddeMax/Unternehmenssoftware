import pandas as pd
import numpy as np
from pathlib import Path

def calculate_rsi(data, periods=14):
    """
    Berechnet den RSI (Relative Strength Index) für einen DataFrame

    Parameters:
    data (pandas.DataFrame): DataFrame mit OHLCV Daten
    periods (int): Anzahl der Perioden für RSI Berechnung (Standard: 14)

    Returns:
    pandas.DataFrame: Original DataFrame mit zusätzlicher RSI Spalte
    """
    df = data.copy()
    df['Close'] = df['Close'].astype(float)
    delta = df['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/periods, min_periods=periods).mean()
    avg_loss = loss.ewm(alpha=1/periods, min_periods=periods).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def calculate_macd(data, span_short=12, span_long=26, span_signal=9):
    """
    Berechnet den MACD (Moving Average Convergence Divergence) für einen DataFrame

    Parameters:
    data (pandas.DataFrame): DataFrame mit OHLCV Daten
    span_short (int): Zeitraum für den kurzfristigen EMA (Standard: 12)
    span_long (int): Zeitraum für den langfristigen EMA (Standard: 26)
    span_signal (int): Zeitraum für die Signallinie (Standard: 9)

    Returns:
    pandas.DataFrame: Original DataFrame mit zusätzlichen MACD und Signal_Line Spalten
    """
    df = data.copy()
    df['Close'] = df['Close'].astype(float)

    short_ema = df['Close'].ewm(span=span_short, adjust=False).mean()
    long_ema = df['Close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()

    return df

def main():
    # Pfad zum aktuellen Skript ermitteln
    script_dir = Path(__file__).resolve().parent
    # Ein Verzeichnis nach oben gehen, um zum Hauptverzeichnis zu kommen
    project_dir = script_dir.parent

    # Eingabedatei
    input_filename = 'SP500_Index_Historical_Data.csv'
    input_path = project_dir / 'sp500_data' / input_filename

    # Prüfen ob die Eingabedatei existiert
    if not input_path.exists():
        raise FileNotFoundError(f"Die Datei {input_filename} wurde nicht im Pfad {input_path} gefunden.")

    # Ausgabedatei im gleichen Verzeichnis erstellen
    output_filename = 'SP500_Index_Historical_Data_with_RSI_MACD.csv'
    output_path = input_path.parent / output_filename

    # CSV Datei einlesen
    df = pd.read_csv(input_path)

    # RSI berechnen (Standard: 14 Perioden)
    df_with_rsi = calculate_rsi(df)

    # MACD berechnen (Standard: span_short=12, span_long=26, span_signal=9)
    df_with_rsi_macd = calculate_macd(df_with_rsi)

    # Ergebnis in neue CSV Datei speichern
    df_with_rsi_macd.to_csv(output_path, index=False)

    print(f"RSI und MACD wurden berechnet und in {output_path} gespeichert.")

if __name__ == "__main__":
    main()
