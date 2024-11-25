import pandas as pd
import os
import platform
import subprocess

# Wikipedia-URL mit den Tabellen
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_changes_table(url):
    """
    Lädt die Änderungen-Tabelle (ID: 'changes') von der Wikipedia-Seite.
    """
    try:
        # Lade nur die Tabelle mit der ID 'changes'
        tables = pd.read_html(url, attrs={"id": "changes"})
        if not tables:
            print("Änderungen-Tabelle mit ID 'changes' nicht gefunden.")
            return None

        # Nehme die erste (und einzige) Tabelle
        changes_table = tables[0]

        # Überprüfen, ob die Tabelle MultiIndex oder einfache Spaltenstruktur hat
        if isinstance(changes_table.columns, pd.MultiIndex):
            # MultiIndex: Hierarchische Struktur
            # Erstelle neue Spaltennamen durch intelligente Kombination der Ebenen
            new_columns = []
            for col in changes_table.columns:
                # Wenn die Ebenen identisch sind oder die zweite Ebene leer ist
                if col[0] == col[1] or not col[1]:
                    new_columns.append(col[0])
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            changes_table.columns = new_columns
        else:
            # Einfache Struktur: Behalte die vorhandenen Spaltennamen
            changes_table.columns = changes_table.columns

        # Spalten zur Überprüfung ausgeben
        print("Spalten in changes_table:", changes_table.columns)

        return changes_table

    except Exception as e:
        print(f"Fehler beim Laden der Änderungen-Tabelle: {e}")
        return None

def save_to_csv(df, filename):
    """
    Speichert einen DataFrame als CSV.
    """
    try:
        df.to_csv(filename, index=False)
        print(f"Datei gespeichert: {filename}")
    except Exception as e:
        print(f"Fehler beim Speichern der Datei {filename}: {e}")

def open_output_folder(path):
    """
    Öffnet den Ordnerpfad im Dateimanager.
    """
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux/Unix
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"Fehler beim Öffnen des Ordners: {e}")

def main():
    # Erstelle Ausgabeverzeichnis, falls nicht vorhanden
    output_dir = "../sp500_data"
    os.makedirs(output_dir, exist_ok=True)

    # Änderungen-Tabelle laden
    changes_table = fetch_changes_table(url)

    if changes_table is not None:
        # Überprüfe, ob die benötigten Spalten vorhanden sind
        required_columns = ['Date', 'Added_Ticker', 'Added_Security', 'Removed_Ticker', 'Removed_Security', 'Reason']
        missing_columns = [col for col in required_columns if col not in changes_table.columns]
        if missing_columns:
            print(f"Fehlende Spalten in changes_table: {missing_columns}")
            return

        # Verarbeite die Spalten für Added und Removed
        added_tickers = changes_table[['Date', 'Added_Ticker', 'Added_Security']].dropna(subset=['Added_Ticker'])
        removed_tickers = changes_table[['Date', 'Removed_Ticker', 'Removed_Security']].dropna(subset=['Removed_Ticker'])

        # Speichern der getrennten Dateien
        save_to_csv(changes_table, os.path.join(output_dir, "sp500_historical_changes.csv"))
        save_to_csv(added_tickers, os.path.join(output_dir, "sp500_added_tickers.csv"))
        save_to_csv(removed_tickers, os.path.join(output_dir, "sp500_removed_tickers.csv"))

        print("Daten erfolgreich extrahiert und gespeichert.")
    else:
        print("Keine Änderungen-Tabelle gefunden. Prozess abgebrochen.")

    # Ordnerpfad öffnen
    print(f"Öffne den Ordner: {os.path.abspath(output_dir)}")
    open_output_folder(os.path.abspath(output_dir))

if __name__ == "__main__":
    main()
