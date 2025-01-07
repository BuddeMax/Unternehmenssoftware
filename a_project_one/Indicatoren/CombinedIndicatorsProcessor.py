import os
import glob
import pandas as pd

class CombinedIndicatorsProcessor:
    """
    Korrigierte Klasse zum Einlesen der finalen CSV-Dateien (RSI1_final.csv, MACD1_final.csv, BollingerBands1_final.csv)
    und der historischen CSV (SP500_Index_Historical_Data_with_RSI_MACD.csv),
    um eine kombinierte CSV-Datei (CombinedIndicators1.csv) mit exakt den gewünschten Spalten zu erstellen.
    """

    def __init__(self, project_root):
        """
        :param project_root: Wurzelverzeichnis des Projekts, unter dem rekursiv nach den Dateien gesucht wird.
        """
        self.project_root = project_root
        # Zielordner für die neue kombinierte CSV-Datei
        self.combined_output_path = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\CombinedIndicators"
        self.combined_output_file = "CombinedIndicators1.csv"

        # Keyword, um die historische CSV zu finden
        self.historical_data_keyword = "SP500_Index_Historical_Data_with_RSI_MACD"

    def find_csv_file(self, filename_keyword):
        """
        Durchsucht rekursiv das Projektverzeichnis nach einer CSV,
        deren Pfad/Basename 'filename_keyword' enthält.
        """
        pattern = os.path.join(self.project_root, "**", "*.csv")
        for file in glob.iglob(pattern, recursive=True):
            if filename_keyword.lower() in os.path.basename(file).lower():
                return file
        return None

    def load_csv(self, file_path):
        """
        Lädt eine CSV-Datei als DataFrame. Sortiert nach 'Date' (falls vorhanden).
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.sort_values(by='Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def create_combined_file(self):
        """
        1. Findet und lädt die Indikator-CSV-Dateien:
           - RSI1_final.csv (enthält u.a. RSI_Overbought, RSI_Slope, etc.)
           - MACD1_final.csv (enthält MACD_BullishCrossover, MACD_Slope, etc.)
           - BollingerBands1_final.csv (enthält BB_UpperBreak, BB_Squeeze, etc.)
        2. Findet und lädt die historische Daten-CSV (SP500_Index_Historical_Data_with_RSI_MACD.csv)
           und nimmt daraus nur die Spalten 'Date', 'Volume', 'RSI', 'MACD', 'Signal_Line' sowie
           den (shifted) 'Close' als 'Close+1'.
        3. Berechnet eine zusätzliche Spalte 'Close_Direction', die angibt, ob 'Close+1' größer als 'Close' ist.
        4. Führt alle DataFrames auf Basis der 'Date'-Spalte zusammen.
        5. Speichert die kombinierte Datei mit nur den geforderten Spalten.
        """

        # --- Indikator-Dateien suchen und laden ---
        rsi_file = self.find_csv_file("RSI1_final")
        macd_file = self.find_csv_file("MACD1_final")
        bb_file = self.find_csv_file("BollingerBands1_final")

        if not all([rsi_file, macd_file, bb_file]):
            missing = []
            if not rsi_file:
                missing.append("RSI1_final.csv")
            if not macd_file:
                missing.append("MACD1_final.csv")
            if not bb_file:
                missing.append("BollingerBands1_final.csv")
            raise FileNotFoundError(f"Folgende Indikator-Dateien wurden nicht gefunden: {', '.join(missing)}")

        print(f"RSI-Datei gefunden: {rsi_file}")
        print(f"MACD-Datei gefunden: {macd_file}")
        print(f"Bollinger-Datei gefunden: {bb_file}")

        # DataFrames laden und auf die geforderten Spalten reduzieren
        rsi_df = self.load_csv(rsi_file)[[
            'Date',
            'RSI_Overbought',
            'RSI_Oversold',
            'RSI_Overbought_Streak',
            'RSI_Oversold_Streak',
            'RSI_CrossOver_70',
            'RSI_CrossUnder_30',
            'RSI_Slope'
        ]]

        macd_df = self.load_csv(macd_file)[[
            'Date',
            'MACD_BullishCrossover',
            'MACD_BearishCrossover',
            'MACD_Bullish_Streak',
            'MACD_Bearish_Streak',
            'MACD_ZeroLine_Crossover',
            'MACD_Slope'
        ]]

        bb_df = self.load_csv(bb_file)[[
            'Date',
            'BB_UpperBreak',
            'BB_LowerBreak',
            'BB_UpperBreak_Streak',
            'BB_LowerBreak_Streak',
            'BB_Squeeze'
        ]]

        # --- Historische Daten finden und laden ---
        historical_file = self.find_csv_file(self.historical_data_keyword)
        if historical_file is None:
            raise FileNotFoundError(f"Historische Daten-Quelldatei mit Keyword '{self.historical_data_keyword}' nicht gefunden.")
        print(f"Historische Daten-Datei gefunden: {historical_file}")
        hist_df = self.load_csv(historical_file)

        # Nur die gewünschten Spalten übernehmen
        # Dabei 'Close' um 1 Tag verschieben und als 'Close+1' einfügen
        if not all(col in hist_df.columns for col in ['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']):
            raise ValueError("Die historischen Daten enthalten nicht alle benötigten Spalten: "
                             "Close, Volume, RSI, MACD, Signal_Line.")

        hist_df['Close+1'] = hist_df['Close'].shift(-1)
        hist_df['Close_Direction'] = (hist_df['Close+1'] > hist_df['Close']).astype(int)  # Neue Spalte
        hist_df = hist_df[[
            'Date',
            'Close+1',
            'Close_Direction',
            'Volume',
            'RSI',
            'MACD',
            'Signal_Line'
        ]]

        # --- Alle DataFrames zusammenführen ---
        combined_df = hist_df \
            .merge(rsi_df, on='Date', how='inner') \
            .merge(macd_df, on='Date', how='inner') \
            .merge(bb_df, on='Date', how='inner')

        # --- Zielordner sicherstellen ---
        os.makedirs(self.combined_output_path, exist_ok=True)

        # --- Kombinierte CSV speichern ---
        combined_output_full_path = os.path.join(self.combined_output_path, self.combined_output_file)
        combined_df.to_csv(combined_output_full_path, index=False, float_format="%.5f")
        print(f"Kombinierte CSV-Datei erstellt: {combined_output_full_path}")

# ----------------------------------------------------
# Beispielhafter Aufruf (wenn das Skript direkt ausgeführt wird):
if __name__ == "__main__":
    project_dir = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware"  # Anpassen an dein Projekt
    combined_processor = CombinedIndicatorsProcessor(project_root=project_dir)
    combined_processor.create_combined_file()
