import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Liste von Aktien aus verschiedenen Branchen
stocks = {
    "Technology": ["AAPL", "MSFT", "GOOGL"],
    "Healthcare": ["JNJ", "PFE", "MRK"],
    "Finance": ["JPM", "BAC", "C"],
    "Energy": ["XOM", "CVX", "BP"],
    "Consumer": ["PG", "KO", "PEP"],
    "Industrials": ["GE", "BA", "CAT"]
}

# Zeitrahmen der Daten
start_date = "1980-01-02"
end_date = datetime.today().strftime('%Y-%m-%d')

# *S&P500-Daten laden und Datum als Spalte, nicht Index*
sp500 = yf.download("^GSPC", start=start_date, end=end_date)
sp500.reset_index(inplace=True)
sp500["Date"] = sp500["Date"].dt.date  # Nur das Datum, ohne Uhrzeit
sp500 = sp500.rename(columns={
    "Date": "Date (S&P500)",
    "Open": "Open (S&P500)",
    "High": "High (S&P500)",
    "Low": "Low (S&P500)",
    "Close": "Close (S&P500)",
    "Volume": "Volume (S&P500)"
})

# Haupt-DataFrame für den Export
combined_data = sp500.copy()

# *Aktien aus verschiedenen Branchen abrufen und integrieren*
for sector, tickers in stocks.items():
    sector_closes = []
    for ticker in tickers:
        print(f"Fetching data for {ticker} in {sector} sector...")

        # *Aktienkursdaten abrufen*
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data["Date"] = data["Date"].dt.date  # Nur Datum
        data = data[["Date", "Close"]]
        data.rename(columns={"Close": f"Close {ticker}"}, inplace=True)

        # *Merge OHNE Indexkonflikte*
        combined_data = combined_data.merge(
            data,
            how="left",
            left_on="Date (S&P500)",
            right_on="Date"
        )

        # *Doppelte Datumsangaben entfernen*
        combined_data.drop(columns=["Date"], inplace=True)
        sector_closes.append(f"Close {ticker}")

    # *Sektor-Durchschnitt berechnen*
    combined_data[f"Durchschnitt Close {sector}"] = combined_data[sector_closes].mean(axis=1)

# **Dynamischer Pfad für die CSV-Datei erstellen**
base_path = os.path.expanduser("~/Unternehmenssoftware/shared_resources/sp500_data")
os.makedirs(base_path, exist_ok=True)  # Erstellt Ordner, falls er nicht existiert

output_file_path = os.path.join(base_path, "sp500_sector_data.csv")

# *CSV speichern*
combined_data.to_csv(output_file_path, index=False)

print(f"Data successfully saved to {output_file_path}")
