import os
import glob
import pandas as pd
import numpy as np

class IndicatorDataProcessor:
    """
    Klasse zum Einlesen der vorhandenen CSV-Dateien:
      - Final_RSI_Stats.csv
      - Final_MACD_Stats.csv
      - Final_Bollinger_Stats.csv
      - (optional) Bollinger_bands_touches.csv

    Und zum Berechnen der erweiterten Indikatorspalten:
      - RSI: RSI_Overbought, RSI_Oversold, RSI_CrossOver_70, etc.
      - MACD: MACD_BullishCrossover, MACD_BearishCrossover, MACD_ZeroLine_Crossover, etc.
      - Bollinger: BB_UpperBreak, BB_LowerBreak, BB_Squeeze, etc.

    Ergebnis: Neue CSV-Dateien in den angegebenen Ordnern, wobei nur 'Date' und die neu erzeugten Indikator-Spalten enthalten sind.
    """

    def __init__(self, project_root):
        """
        :param project_root: Wurzelverzeichnis des Projekts, unter dem rekursiv nach den Dateien gesucht wird.
        """
        self.project_root = project_root

        # Zielordner für die neuen CSV-Dateien (von dir vorgegeben)
        self.rsi_output_path = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\RSI"
        self.macd_output_path = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\MACD"
        self.bb_output_path = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware\sp500_data\BollingerBands"

        # Neue Dateinamen (jeweils mit "1" am Ende, plus Zusatz, um Überschneidungen zu vermeiden)
        self.rsi_output_file = "RSI1_final.csv"
        self.macd_output_file = "MACD1_final.csv"
        self.bb_output_file = "BollingerBands1_final.csv"

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

    # ----------------------------------------------------
    # Berechnungen der neuen Indikatorspalten
    # ----------------------------------------------------

    def compute_macd_indicators(self, df):
        """
        Aus Final_MACD_Stats.csv werden 'MACD' und 'Signal_Line' genutzt, um:
         - MACD_BullishCrossover
         - MACD_BearishCrossover
         - MACD_ZeroLine_Crossover
         - MACD_Slope
         - (Bullish/Bearish) Crossover_Streak
        zu berechnen.
        """
        if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
            raise ValueError("Spalten 'MACD' oder 'Signal_Line' fehlen im MACD-DataFrame.")

        df['MACD_BullishCrossover'] = 0
        df['MACD_BearishCrossover'] = 0
        df['MACD_ZeroLine_Crossover'] = 0
        df['MACD_Slope'] = 0.0
        df['MACD_Bullish_Streak'] = 0
        df['MACD_Bearish_Streak'] = 0

        for i in range(1, len(df)):
            macd_t = df.loc[i, 'MACD']
            macd_t_minus_1 = df.loc[i-1, 'MACD']
            sig_t = df.loc[i, 'Signal_Line']
            sig_t_minus_1 = df.loc[i-1, 'Signal_Line']

            # Bullish Crossover
            if macd_t >= sig_t and macd_t_minus_1 < sig_t_minus_1:
                df.loc[i, 'MACD_BullishCrossover'] = 1

            # Bearish Crossover
            if macd_t <= sig_t and macd_t_minus_1 > sig_t_minus_1:
                df.loc[i, 'MACD_BearishCrossover'] = 1

            # ZeroLine Crossover
            if macd_t >= 0 and macd_t_minus_1 < 0:
                df.loc[i, 'MACD_ZeroLine_Crossover'] = 1

            # MACD-Slope
            df.loc[i, 'MACD_Slope'] = macd_t - macd_t_minus_1

            # Bullish Streak
            if df.loc[i, 'MACD_BullishCrossover'] == 1:
                df.loc[i, 'MACD_Bullish_Streak'] = df.loc[i-1, 'MACD_Bullish_Streak'] + 1
            else:
                if macd_t > sig_t:
                    df.loc[i, 'MACD_Bullish_Streak'] = df.loc[i-1, 'MACD_Bullish_Streak'] + 1
                else:
                    df.loc[i, 'MACD_Bullish_Streak'] = 0

            # Bearish Streak
            if df.loc[i, 'MACD_BearishCrossover'] == 1:
                df.loc[i, 'MACD_Bearish_Streak'] = df.loc[i-1, 'MACD_Bearish_Streak'] + 1
            else:
                if macd_t < sig_t:
                    df.loc[i, 'MACD_Bearish_Streak'] = df.loc[i-1, 'MACD_Bearish_Streak'] + 1
                else:
                    df.loc[i, 'MACD_Bearish_Streak'] = 0

        return df

    def compute_rsi_indicators(self, df):
        """
        Aus Final_RSI_Stats.csv wird v.a. die Spalte 'RSI' genutzt, um:
         - RSI_Overbought
         - RSI_Oversold
         - RSI_Overbought_Streak
         - RSI_Oversold_Streak
         - RSI_CrossOver_70
         - RSI_CrossUnder_30
         - RSI_Slope
        zu generieren.
        """
        if 'RSI' not in df.columns:
            raise ValueError("Spalte 'RSI' fehlt im RSI-DataFrame.")

        df['RSI_Overbought'] = 0
        df['RSI_Oversold'] = 0
        df['RSI_Overbought_Streak'] = 0
        df['RSI_Oversold_Streak'] = 0
        df['RSI_CrossOver_70'] = 0
        df['RSI_CrossUnder_30'] = 0
        df['RSI_Slope'] = 0.0

        for i in range(1, len(df)):
            rsi_t = df.loc[i, 'RSI']
            rsi_t_minus_1 = df.loc[i-1, 'RSI']

            # Overbought / Oversold
            if rsi_t > 70:
                df.loc[i, 'RSI_Overbought'] = 1
            if rsi_t < 30:
                df.loc[i, 'RSI_Oversold'] = 1

            # Overbought / Oversold Streak
            if df.loc[i, 'RSI_Overbought'] == 1:
                df.loc[i, 'RSI_Overbought_Streak'] = df.loc[i-1, 'RSI_Overbought_Streak'] + 1
            else:
                df.loc[i, 'RSI_Overbought_Streak'] = 0

            if df.loc[i, 'RSI_Oversold'] == 1:
                df.loc[i, 'RSI_Oversold_Streak'] = df.loc[i-1, 'RSI_Oversold_Streak'] + 1
            else:
                df.loc[i, 'RSI_Oversold_Streak'] = 0

            # CrossOver 70 / CrossUnder 30
            if rsi_t >= 70 and rsi_t_minus_1 < 70:
                df.loc[i, 'RSI_CrossOver_70'] = 1
            if rsi_t <= 30 and rsi_t_minus_1 > 30:
                df.loc[i, 'RSI_CrossUnder_30'] = 1

            # RSI_Slope
            df.loc[i, 'RSI_Slope'] = rsi_t - rsi_t_minus_1

        return df

    def compute_bollinger_indicators(self, df, df_touches=None):
        """
        Aus Final_Bollinger_Stats.csv werden insbesondere 'Close_bb', 'Upper_Band',
        'Lower_Band' und 'Moving_Avg' genutzt, um:
         - BB_UpperBreak
         - BB_LowerBreak
         - BB_UpperBreak_Streak
         - BB_LowerBreak_Streak
         - BB_Squeeze (z. B. < 2% Bandbreite)
        zu berechnen.

        Falls Bollinger_bands_touches.csv existiert (df_touches),
        können wir das optional mergen.
        """
        required_cols = ['Upper_Band', 'Lower_Band', 'Moving_Avg']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Fehlende Spalten für Bollinger-Berechnungen: {missing_cols}")

        # Referenz-Schlusskurs: 'Close_bb' oder 'Close_sp500'
        close_col = 'Close_bb'
        if close_col not in df.columns:
            if 'Close_sp500' in df.columns:
                close_col = 'Close_sp500'
            else:
                raise ValueError("Keine geeignete Schlusskurs-Spalte (Close_bb/Close_sp500) für Bollinger-Bands gefunden.")

        # Neue Spalten
        df['BB_UpperBreak'] = 0
        df['BB_LowerBreak'] = 0
        df['BB_UpperBreak_Streak'] = 0
        df['BB_LowerBreak_Streak'] = 0
        df['BB_Squeeze'] = 0

        for i in range(len(df)):
            close_val = df.loc[i, close_col]
            upper_val = df.loc[i, 'Upper_Band']
            lower_val = df.loc[i, 'Lower_Band']
            ma_val = df.loc[i, 'Moving_Avg']

            # Break
            if close_val > upper_val:
                df.loc[i, 'BB_UpperBreak'] = 1
            if close_val < lower_val:
                df.loc[i, 'BB_LowerBreak'] = 1

            # Squeeze-Bedingung (z. B. Bandbreite < 2% vom MA)
            if ma_val != 0:
                bandwidth_rel = (upper_val - lower_val) / ma_val
                if bandwidth_rel < 0.02:
                    df.loc[i, 'BB_Squeeze'] = 1

            # Streak-Berechnung erst ab Zeile 1
            if i > 0:
                if df.loc[i, 'BB_UpperBreak'] == 1:
                    df.loc[i, 'BB_UpperBreak_Streak'] = df.loc[i-1, 'BB_UpperBreak_Streak'] + 1
                else:
                    df.loc[i, 'BB_UpperBreak_Streak'] = 0

                if df.loc[i, 'BB_LowerBreak'] == 1:
                    df.loc[i, 'BB_LowerBreak_Streak'] = df.loc[i-1, 'BB_LowerBreak_Streak'] + 1
                else:
                    df.loc[i, 'BB_LowerBreak_Streak'] = 0

        # Optional: Merge mit bollinger_bands_touches
        if df_touches is not None:
            df_merged = pd.merge(df, df_touches, on='Date', how='left')
            return df_merged
        else:
            return df

    # ----------------------------------------------------
    # Hauptmethode: neue Dateien erstellen
    # ----------------------------------------------------
    def create_new_files(self):
        """
        1. Sucht Final_MACD_Stats.csv, Final_RSI_Stats.csv, Final_Bollinger_Stats.csv
           (ggf. bollinger_bands_touches.csv) im Projektverzeichnis.
        2. Lädt DataFrames und berechnet die neuen Indikatorspalten.
        3. Legt 'abgespeckte' DataFrames mit nur 'Date' + neu berechneten Spalten an.
        4. Schreibt diese in die angegebenen Zielordner mit neuem Dateinamen.
        """

        # --- RSI ---
        rsi_file = self.find_csv_file("Final_RSI_Stats")
        if rsi_file is None:
            print("Keine RSI-Quelldatei (Final_RSI_Stats.csv) gefunden.")
        else:
            print(f"RSI-Datei gefunden: {rsi_file}")
            rsi_df = self.load_csv(rsi_file)
            rsi_df = self.compute_rsi_indicators(rsi_df)
            rsi_cols = [
                'Date',
                'RSI_Overbought',
                'RSI_Oversold',
                'RSI_Overbought_Streak',
                'RSI_Oversold_Streak',
                'RSI_CrossOver_70',
                'RSI_CrossUnder_30',
                'RSI_Slope'
            ]
            rsi_output_df = rsi_df[rsi_cols].copy()
            rsi_output_path = os.path.join(self.rsi_output_path, self.rsi_output_file)
            rsi_output_df.to_csv(rsi_output_path, index=False, float_format="%.5f")
            print(f"Neue RSI-Datei erstellt: {rsi_output_path}")

        # --- MACD ---
        macd_file = self.find_csv_file("Final_MACD_Stats")
        if macd_file is None:
            print("Keine MACD-Quelldatei (Final_MACD_Stats.csv) gefunden.")
        else:
            print(f"MACD-Datei gefunden: {macd_file}")
            macd_df = self.load_csv(macd_file)
            macd_df = self.compute_macd_indicators(macd_df)
            macd_cols = [
                'Date',
                'MACD_BullishCrossover',
                'MACD_BearishCrossover',
                'MACD_Bullish_Streak',
                'MACD_Bearish_Streak',
                'MACD_ZeroLine_Crossover',
                'MACD_Slope'
            ]
            macd_output_df = macd_df[macd_cols].copy()
            macd_output_path = os.path.join(self.macd_output_path, self.macd_output_file)
            macd_output_df.to_csv(macd_output_path, index=False, float_format="%.5f")
            print(f"Neue MACD-Datei erstellt: {macd_output_path}")

        # --- Bollinger ---
        bb_file = self.find_csv_file("Final_Bollinger_Stats")
        if bb_file is None:
            print("Keine Bollinger-Quelldatei (Final_Bollinger_Stats.csv) gefunden.")
        else:
            print(f"Bollinger-Datei gefunden: {bb_file}")
            bb_df = self.load_csv(bb_file)

            # Optional: bollinger_bands_touches einbinden
            touches_file = self.find_csv_file("bollinger_bands_touches")
            if touches_file:
                print(f"Bollinger-Touches-Datei gefunden: {touches_file}")
                touches_df = self.load_csv(touches_file)
                bb_df = self.compute_bollinger_indicators(bb_df, touches_df)
            else:
                bb_df = self.compute_bollinger_indicators(bb_df)

            bb_cols = [
                'Date',
                'BB_UpperBreak',
                'BB_LowerBreak',
                'BB_UpperBreak_Streak',
                'BB_LowerBreak_Streak',
                'BB_Squeeze'
            ]
            bb_output_df = bb_df[bb_cols].copy()
            bb_output_path = os.path.join(self.bb_output_path, self.bb_output_file)
            bb_output_df.to_csv(bb_output_path, index=False, float_format="%.5f")
            print(f"Neue Bollinger-Datei erstellt: {bb_output_path}")


# ----------------------------------------------------
# Beispielhafter Aufruf (wenn das Skript direkt ausgeführt wird):
if __name__ == "__main__":
    project_dir = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware"  # Anpassen an dein Projekt
    processor = IndicatorDataProcessor(project_root=project_dir)
    processor.create_new_files()
