import yfinance as yf
import pandas as pd

# Definition des S&P 500 Symbols bei Yahoo Finance
SP500_SYMBOL = "^GSPC"

# Zeitraum für die Datenabfrage
START_DATE = "1980-01-01"  # Startdatum
END_DATE = "2024-11-27"    # Enddatum (aktuell)

# Herunterladen der historischen Daten des S&P 500
print("Lade historische Daten des S&P 500 Index...")
sp500_data = yf.download(SP500_SYMBOL, start=START_DATE, end=END_DATE)

# Sicherstellen, dass Daten erfolgreich geladen wurden
if sp500_data.empty:
    print("Keine Daten gefunden. Bitte überprüfen Sie das Symbol oder den Zeitraum.")
else:
    print("Daten erfolgreich geladen.")

    # MultiIndex entfernen, falls vorhanden
    sp500_data.reset_index(inplace=True)

    # Nur relevante Spalten behalten
    print("\nFiltere relevante Spalten...")
    sp500_filtered = sp500_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Sicherstellen, dass das Datum die korrekte Struktur hat
    sp500_filtered['Date'] = pd.to_datetime(sp500_filtered['Date'])

    # Speichern der gefilterten Daten in eine CSV-Datei
    csv_filename = "SP500_Index_Historical_Data.csv"
    sp500_filtered.to_csv(csv_filename, index=False)
    print(f"Daten wurden erfolgreich in '{csv_filename}' gespeichert.")