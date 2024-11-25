import pandas as pd
import os

def main():
    # Pfad zum Ausgabeverzeichnis
    output_dir = "../sp500_data"

    # Überprüfen, ob das Ausgabeverzeichnis existiert
    if not os.path.exists(output_dir):
        print(f"Das Verzeichnis {output_dir} existiert nicht.")
        return

    # Pfade zu den CSV-Dateien
    current_members_file = os.path.join(output_dir, "sp500_current_members.csv")
    added_tickers_file = os.path.join(output_dir, "sp500_added_tickers.csv")
    removed_tickers_file = os.path.join(output_dir, "sp500_removed_tickers.csv")

    # Überprüfen, ob die Dateien existieren
    for file in [current_members_file, added_tickers_file, removed_tickers_file]:
        if not os.path.isfile(file):
            print(f"Datei nicht gefunden: {file}")
            return

    # Aktuelle Mitglieder laden
    current_members = pd.read_csv(current_members_file)

    # Ticker-Spalte identifizieren
    if 'Symbol' in current_members.columns:
        current_tickers = set(current_members['Symbol'].dropna().unique())
    else:
        print("Spalte 'Symbol' in sp500_current_members.csv nicht gefunden.")
        return

    # Hinzugefügte Ticker laden
    added_tickers_df = pd.read_csv(added_tickers_file)
    if 'Added_Ticker' in added_tickers_df.columns:
        added_tickers = set(added_tickers_df['Added_Ticker'].dropna().unique())
    else:
        print("Spalte 'Added_Ticker' in sp500_added_tickers.csv nicht gefunden.")
        return

    # Entfernte Ticker laden
    removed_tickers_df = pd.read_csv(removed_tickers_file)
    if 'Removed_Ticker' in removed_tickers_df.columns:
        removed_tickers = set(removed_tickers_df['Removed_Ticker'].dropna().unique())
    else:
        print("Spalte 'Removed_Ticker' in sp500_removed_tickers.csv nicht gefunden.")
        return

    # Alle Ticker sammeln
    all_tickers = current_tickers.union(added_tickers).union(removed_tickers)

    # Duplikate entfernen und sortieren
    unique_tickers = sorted(all_tickers)

    # Ticker in der Konsole ausgeben
    print("Alle jemals im S&P 500 enthaltenen Ticker:")
    for ticker in unique_tickers:
        print(ticker)

    # DataFrame erstellen und als CSV speichern
    all_tickers_df = pd.DataFrame({'Symbol': unique_tickers})

    # Speichern
    all_tickers_file = os.path.join(output_dir, "sp500_all_tickers.csv")
    all_tickers_df.to_csv(all_tickers_file, index=False)
    print(f"Alle Ticker wurden gespeichert in {all_tickers_file}")

if __name__ == "__main__":
    main()
