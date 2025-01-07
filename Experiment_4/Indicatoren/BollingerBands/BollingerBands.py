import pandas as pd
import os
import talib
from pathlib import Path


class BollingerBands:
    r"""
    Eine Klasse zur Berechnung von Bollinger-Bändern basierend auf den historischen Daten des S&P 500.

    **Bollinger-Bänder (BBands):**
    Bollinger-Bänder sind ein technischer Indikator, der die Volatilität eines Finanzinstruments misst und hilft, potenzielle Überkauf- oder Überverkaufssituationen zu identifizieren. Sie bestehen aus drei Bändern:

    1. **Mittleres Band (Moving Average):**
       Ein einfacher gleitender Durchschnitt (SMA) der Schlusskurse über einen definierten Zeitraum (z. B. 20 Tage).
       Mathematisch: SMA = Summe der Schlusskurse / Anzahl der Schlusskurse.
    2. **Oberes Band (Upper Band):**
       Das mittlere Band plus ein Vielfaches der Standardabweichung.
       Mathematisch: Upper Band = SMA + (k * Standardabweichung).
    3. **Unteres Band (Lower Band):**
       Das mittlere Band minus ein Vielfaches der Standardabweichung.
       Mathematisch: Lower Band = SMA - (k * Standardabweichung).

    **Input:**
    - Historische Schlusskurse ("Close") aus der CSV-Datei mit S&P 500-Daten.

    **Verarbeitung:**
    - Berechnung des gleitenden Durchschnitts (SMA): Summe der Schlusskurse / Anzahl der Schlusskurse.
    - Berechnung der Standardabweichung (STD) für denselben Zeitraum: STD = Quadratwurzel aus (Summe der quadrierten Abweichungen der Schlusskurse vom SMA / Anzahl der Schlusskurse).
    - Ableitung des oberen und unteren Bands durch Hinzufügen bzw. Subtrahieren eines Vielfachen der Standardabweichung vom gleitenden Durchschnitt.

    **Output:**
    - Drei neue Spalten in den Daten:
      - `Moving_Avg`: Das mittlere Band (gleitender Durchschnitt).
      - `Upper_Band`: Das obere Band.
      - `Lower_Band`: Das untere Band.

    **Speicherung in der CSV:**
    Die berechneten Bollinger-Bänder werden zusammen mit den ursprünglichen Daten in einer neuen CSV-Datei gespeichert, die im Unterordner `BollingerBands` abgelegt wird.
    """

    def __init__(self, subfolder="sp500_data", file_name="SP500_Index_Historical_Data.csv",
                 output_folder="BollingerBands"):
        self.subfolder = subfolder
        self.file_name = file_name
        self.output_folder = output_folder
        self.file_path = self._find_csv_file()

    def _find_csv_file(self):
        current_path = Path(__file__).resolve().parent
        for parent in current_path.parents:
            potential_path = parent / self.subfolder / self.file_name
            if potential_path.exists():
                return potential_path
        print(f"Die Datei {self.file_name} wurde im Projektverzeichnis nicht gefunden.")
        return None

    def calculate_bollinger_bands(self, timeperiod=20, nbdevup=2, nbdevdn=2):
        if not self.file_path:
            print("Keine gültige Datei gefunden. Berechnung abgebrochen.")
            return

        try:
            # CSV-Datei laden
            df = pd.read_csv(self.file_path, parse_dates=['Date'])

            # Überprüfen, ob die notwendigen Spalten existieren
            if 'Close' not in df.columns:
                raise ValueError("Die Spalte 'Close' fehlt in den Daten.")

            # Bollinger-Bänder mit TA-Lib berechnen
            df['Upper_Band'], df['Moving_Avg'], df['Lower_Band'] = talib.BBANDS(
                df['Close'],
                timeperiod=timeperiod,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
                matype=0  # SMA
            )

            # Nur relevante Spalten behalten
            df = df[['Date', 'Close', 'Moving_Avg', 'Upper_Band', 'Lower_Band']]

            # Ordner erstellen, falls nicht vorhanden
            output_path = Path(self.file_path).parent / self.output_folder
            output_path.mkdir(parents=True, exist_ok=True)

            # Ergebnisse speichern
            output_file = output_path / "bollinger_bands.csv"
            df.to_csv(output_file, index=False)

            print(f"Bollinger-Bänder wurden erfolgreich berechnet und in {output_file} gespeichert.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    bollinger = BollingerBands()
    bollinger.calculate_bollinger_bands()
