import pandas as pd
import os

def combine_csv_files():
    # Pfade der Eingabedateien
    file1 = r"C:\\Users\\Anwender\\PycharmProjects\\Unternehmenssoftware\\shared_resources\\sp500_data\\SP500_Index_Historical_Data.csv"
    file2 = r"C:\\Users\\Anwender\\PycharmProjects\\Unternehmenssoftware\\shared_resources\\sp500_data\\CombinedIndicators\\CombinedIndicators1.csv"

    # CSV-Dateien einlesen
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Spalte "Close+1" aus der zweiten Datei entfernen, falls vorhanden
    if "Close+1" in df2.columns:
        df2.drop(columns=["Close+1"], inplace=True)

    # Zusammenführen der Dateien anhand der Spalten
    combined_df = pd.concat([df1, df2], axis=1)

    # Doppelte Spalten entfernen
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Speicherort der Ausgabe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "Combined_SP500_Data.csv")

    # Ergebnis speichern
    combined_df.to_csv(output_file, index=False)
    print(f"Die zusammengeführte Datei wurde gespeichert unter: {output_file}")

if __name__ == "__main__":
    combine_csv_files()
