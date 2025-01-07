import pandas as pd


def extract_symbol_column():
    # Pfade zu den CSV-Dateien
    input_csv = r"C:\Users\b_mas\PycharmProjects\Unternehmenssoftware\sp500_data\sp500_current_members_complete_header.csv"
    output_csv = r"C:\Users\b_mas\PycharmProjects\Unternehmenssoftware\sp500_data\sp500_current_members.csv"

    try:
        # Lesen der Eingabe-CSV-Datei
        data = pd.read_csv(input_csv)
        print("Spalten in der Eingabedatei:", data.columns.tolist())

        # Überprüfen, ob die Spalte 'Symbol' vorhanden ist
        if 'Symbol' not in data.columns:
            print("Fehler: Die Spalte 'Symbol' wurde nicht in der CSV-Datei gefunden.")
            return

        # Extrahieren der 'Symbol'-Spalte
        symbols = data[['Symbol']].copy()

        # Entfernen von möglichen Leerzeichen in den Ticker-Symbolen
        symbols['Symbol'] = symbols['Symbol'].str.strip()

        # Speichern der 'Symbol'-Spalte in die neue CSV-Datei
        symbols.to_csv(output_csv, index=False)

        print(f"Die Datei '{output_csv}' wurde erfolgreich erstellt.")
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Dateien: {e}")


if __name__ == "__main__":
    extract_symbol_column()
