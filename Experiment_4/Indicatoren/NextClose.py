import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_regression
import warnings

class StatisticalAnalyzer:
    """
    Klasse zur Berechnung statistischer Maße zwischen den Indikatoren und Close_Direction in CombinedIndicators1.csv.
    Verwendet:
      - Pearson's r für kontinuierliche Indikatoren
      - Point-Biserial Korrelation für binäre Indikatoren
      - Spearman's rho für ordinal skalierte Indikatoren
      - Mutual Information für alle Indikatoren
    """

    def __init__(self, project_root, export_csv=False, export_path='statistical_results.csv'):
        """
        :param project_root: Wurzelverzeichnis des Projekts, unter dem rekursiv nach CombinedIndicators1.csv gesucht wird.
        :param export_csv: Boolean, ob die Ergebnisse in eine CSV-Datei exportiert werden sollen.
        :param export_path: Pfad zur Exportdatei.
        """
        self.project_root = project_root
        self.combined_file_keyword = "CombinedIndicators1"  # kann angepasst werden, falls der Dateiname anders lautet
        self.combined_df = None
        self.start_date = pd.to_datetime("1980-02-08")
        self.export_csv = export_csv
        self.export_path = export_path

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

    def load_combined_csv(self):
        """
        Findet und lädt die CombinedIndicators1.csv als DataFrame.
        Filtert die Daten ab dem 1980-02-08 Datum.
        Entfernt Zeilen mit NaN in 'Close_Direction'.
        """
        combined_file = self.find_csv_file(self.combined_file_keyword)
        if combined_file is None:
            raise FileNotFoundError(
                f"Combined Indicators-Datei mit Keyword '{self.combined_file_keyword}' nicht gefunden."
            )
        print(f"Combined Indicators-Datei gefunden: {combined_file}")

        df = pd.read_csv(combined_file, parse_dates=['Date'])
        df.sort_values(by='Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Filter ab 1980-02-08
        df_filtered = df[df['Date'] >= self.start_date].copy()

        # Entfernen von Zeilen mit NaN in 'Close_Direction'
        if 'Close_Direction' not in df_filtered.columns:
            raise ValueError("Die Spalte 'Close_Direction' fehlt in der geladenen Datei.")
        df_filtered.dropna(subset=['Close_Direction'], inplace=True)

        self.combined_df = df_filtered
        print(f"Daten ab {self.start_date.date()} geladen. Gesamtzeilen nach Filtern: {len(self.combined_df)}")

    def identify_column_types(self):
        """
        Identifiziert automatisch die Typen der Spalten:
         - Binär: genau 2 eindeutige Werte (z.B. 0 und 1)
         - Ordinal: Ganzzahlen mit mehr als 2 eindeutigen Werten
         - Kontinuierlich: numerisch (float) oder integer mit sehr vielen eindeutigen Werten
        """
        binary_cols = []
        ordinal_cols = []
        continuous_cols = []

        for col in self.combined_df.columns:
            # Überspringe Datum und Zielvariable
            if col in ['Date', 'Close_Direction']:
                continue

            # Einzigartige Werte ohne NaNs
            unique_vals = self.combined_df[col].dropna().unique()
            num_unique = len(unique_vals)

            # Prüfen, ob Spalte nur 2 eindeutige Werte hat (z. B. 0, 1 oder 0.0, 1.0)
            if num_unique == 2:
                # Wir nehmen an, dass es sich dabei um binäre Daten handelt
                binary_cols.append(col)
            else:
                # Wenn die Spalte ein Integer-Type ist (z.B. ordinal) oder sehr wenige unique Values
                if pd.api.types.is_integer_dtype(self.combined_df[col]):
                    ordinal_cols.append(col)
                elif pd.api.types.is_float_dtype(self.combined_df[col]):
                    continuous_cols.append(col)

        return binary_cols, ordinal_cols, continuous_cols

    def analyze_correlations(self):
        """
        Berechnet je Spalte die relevanten statistischen Maße:
         - Binär  => Point-Biserial Correlation + Mutual Information
         - Ordinal => Spearman's Rho + Pearson's r + Mutual Information
         - Kontinuierlich => Pearson's r + Spearman's Rho + Mutual Information
        """
        if self.combined_df is None:
            raise ValueError("Daten sind nicht geladen. Führe zuerst load_combined_csv() aus.")

        # Zielvariable
        target = self.combined_df['Close_Direction']

        # Indikator-Spalten (ohne 'Date' und 'Close_Direction')
        indicator_cols = [col for col in self.combined_df.columns if col not in ['Date', 'Close_Direction']]

        # Spaltenkategorien identifizieren
        binary_cols, ordinal_cols, continuous_cols = self.identify_column_types()

        print(f"Erkannte binäre Spalten: {binary_cols}")
        print(f"Erkannte ordinale Spalten: {ordinal_cols}")
        print(f"Erkannte kontinuierliche Spalten: {continuous_cols}")

        # Liste zur Aufnahme aller Berechnungsergebnisse
        results = []

        # Warnungen ignorieren
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            for col in indicator_cols:
                if self.combined_df[col].dropna().empty:
                    continue

                if col in binary_cols:
                    corr_pb, p_value_pb = pointbiserialr(self.combined_df[col], target)
                    results.append({
                        'Indicator': col,
                        'Typ': 'Binär',
                        'Methode': "Point-Biserial Correlation",
                        'Koeffizient': corr_pb,
                        'p-Wert': p_value_pb
                    })

                elif col in ordinal_cols or col in continuous_cols:
                    corr_spear, p_spear = spearmanr(self.combined_df[col], target)
                    corr_pear, p_pear = pearsonr(self.combined_df[col], target)
                    results.append({
                        'Indicator': col,
                        'Typ': 'Ordinal/Kontinuierlich',
                        'Methode': "Spearman's Rho",
                        'Koeffizient': corr_spear,
                        'p-Wert': p_spear
                    })
                    results.append({
                        'Indicator': col,
                        'Typ': 'Ordinal/Kontinuierlich',
                        'Methode': "Pearson's r",
                        'Koeffizient': corr_pear,
                        'p-Wert': p_pear
                    })

        results_df = pd.DataFrame(results)
        results_df.sort_values(by=['Indicator', 'Methode'], inplace=True)

        print("\nStatistische Maße zwischen Indikatoren und Close_Direction:\n")
        print(results_df.to_string(index=False))

        if self.export_csv:
            results_df.to_csv(self.export_path, index=False)
            print(f"\nErgebnisse wurden erfolgreich in '{self.export_path}' exportiert.")

if __name__ == "__main__":
    project_dir = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware"
    analyzer = StatisticalAnalyzer(project_root=project_dir, export_csv=True, export_path='correlation_results.csv')
    analyzer.load_combined_csv()
    analyzer.analyze_correlations()
