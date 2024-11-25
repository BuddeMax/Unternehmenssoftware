import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime

# Funktion zum Laden der Ticker aus der hochgeladenen CSV
def load_sp500_tickers_from_csv(csv_path):
    try:
        tickers_data = pd.read_csv(csv_path)
        tickers = tickers_data['Symbol'].tolist()  # Annahme: Spalte heißt "Symbol"
        return tickers
    except Exception as e:
        print(f"Fehler beim Laden der Ticker aus der CSV-Datei: {e}")
        return []

# Funktion zum Herunterladen der Daten für einen Ticker mit Wiederholungen bei Fehlern
def download_ticker_data(ticker, start_date, end_date, max_retries=5):
    attempt = 0
    delay = 0.25
    # Initiale Verzögerung in Sekunden
    while attempt < max_retries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"Keine Daten für {ticker} abgerufen.")
                return None
            else:
                data['Ticker'] = ticker  # Ticker als zusätzliche Spalte hinzufügen
                return data
        except Exception as e:
            print(f"Fehler beim Abrufen der Daten für {ticker}: {e}")
            attempt += 1
            time.sleep(delay)
            delay *= 2  # Exponentieller Backoff
    print(f"Fehler: Daten für {ticker} konnten nach {max_retries} Versuchen nicht abgerufen werden.")
    return None

# Hauptfunktion
def main():
    # CSV-Pfad mit aktuellen Tickersymbolen
    csv_path = r"C:\Users\b_mas\PycharmProjects\Unternehmenssoftware\sp500_data\sp500_current_members.csv"

    # Speicherort für die endgültige CSV
    output_file = r"C:\Users\b_mas\PycharmProjects\Unternehmenssoftware\sp500_data\sp500_historical_data_current_members.csv"

    # Ticker laden
    tickers = load_sp500_tickers_from_csv(csv_path)

    # Zeitrahmen für die Daten (anpassen nach Bedarf)
    start_date = '1980-01-01'  # Anfangsdatum
    end_date = datetime.today().strftime('%Y-%m-%d')  # Heutiges Datum

    # Prüfen, ob die Ausgabedatei bereits existiert, um den Header entsprechend zu behandeln
    if os.path.exists(output_file):
        os.remove(output_file)  # Vorhandene Datei löschen, um neu zu starten

    # Für jeden Ticker Daten abrufen und direkt in die CSV-Datei schreiben
    for idx, ticker in enumerate(tickers):
        print(f"Rufe Daten für {ticker} ab ({start_date} bis {end_date})...")
        data = download_ticker_data(ticker, start_date, end_date)

        if data is not None and not data.empty:
            # Datenvalidierung
            if data.isnull().values.any():
                print(f"Daten für {ticker} enthalten fehlende Werte.")
            else:
                print(f"Daten für {ticker} enthalten keine fehlenden Werte.")

            # Daten direkt in die CSV-Datei schreiben
            if idx == 0:
                # Für den ersten Ticker Header einschließen
                data.to_csv(output_file, mode='w', header=True, index=True)
            else:
                # Für nachfolgende Ticker ohne Header anhängen
                data.to_csv(output_file, mode='a', header=False, index=True)
        else:
            print(f"Speichern der Daten für {ticker} aufgrund von Fehlern beim Abruf übersprungen.")

        # Dynamische Verzögerung zur Vermeidung von API-Limits
        time.sleep(1)  # Initiale Verzögerung

    print(f"Alle historischen Daten gespeichert in {output_file}")

if __name__ == "__main__":
    main()
