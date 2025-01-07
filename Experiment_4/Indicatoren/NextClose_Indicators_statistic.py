import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_regression
import warnings

class StatisticalAnalyzer:
    """
    Klasse zur Berechnung statistischer Maße zwischen den Indikatoren und dem Next_Close in CombinedIndicators1.csv.
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
        Entfernt Zeilen mit NaN in 'Next_Close'.
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

        # Entfernen von Zeilen mit NaN in 'Next_Close'
        # Falls dein CSV-Header abweicht (z. B. "Close+1" statt "Next_Close"), bitte den Spaltennamen hier anpassen
        if 'Next_Close' not in df_filtered.columns:
            # Wenn deine CSV z. B. "Close+1" als Ziel hat, nutze diese Zeile:
            df_filtered.rename(columns={"Close+1": "Next_Close"}, inplace=True)
        df_filtered.dropna(subset=['Next_Close'], inplace=True)

        self.combined_df = df_filtered
        print(f"Daten ab {self.start_date.date()} geladen. Gesamtzeilen nach Filtern: {len(self.combined_df)}")

    def identify_column_types(self):
        """
        Identifiziert automatisch die Typen der Spalten:
         - Binär: genau 2 eindeutige Werte (z.B. 0 und 1)
         - Ordinal: Ganzzahlen mit mehr als 2 eindeutigen Werten
         - Kontinuierlich: numerisch (float) oder integer mit sehr vielen eindeutigen Werten
         - Kategorisch/sonstige Fälle werden nicht weiter unterschieden (falls solche Spalten existieren).
        """
        binary_cols = []
        ordinal_cols = []
        continuous_cols = []

        for col in self.combined_df.columns:
            # Überspringe Datum und Zielvariable
            if col in ['Date', 'Next_Close']:
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
                    # Ist es eine reine Ganzzahlspalte
                    # Wenn es mehr als 2 unique Werte hat, betrachten wir es als ordinal
                    ordinal_cols.append(col)
                elif pd.api.types.is_float_dtype(self.combined_df[col]):
                    # Fließkommazahlen => kontinuierlich
                    continuous_cols.append(col)
                else:
                    # Weitere Typen, z. B. Strings, können hier landen
                    pass

        return binary_cols, ordinal_cols, continuous_cols

    def analyze_correlations(self):
        """
        Berechnet je Spalte die relevanten statistischen Maße:
         - Binär  => Point-Biserial Correlation + Mutual Information
         - Ordinal => Spearman's Rho + (optional) Pearson's r + Mutual Information
         - Kontinuierlich => Pearson's r + Spearman's Rho + Mutual Information
         - Andere => Spearman's Rho + Mutual Information
        Anschließend Ausgabe als DataFrame. Export als CSV optional.
        """
        if self.combined_df is None:
            raise ValueError("Daten sind nicht geladen. Führe zuerst load_combined_csv() aus.")

        # Zielvariable
        target = self.combined_df['Next_Close']

        # Indikator-Spalten (ohne 'Date' und 'Next_Close')
        indicator_cols = [col for col in self.combined_df.columns if col not in ['Date', 'Next_Close']]

        # Spaltenkategorien identifizieren
        binary_cols, ordinal_cols, continuous_cols = self.identify_column_types()

        print(f"Erkannte binäre Spalten: {binary_cols}")
        print(f"Erkannte ordinale Spalten: {ordinal_cols}")
        print(f"Erkannte kontinuierliche Spalten: {continuous_cols}")

        # Liste zur Aufnahme aller Berechnungsergebnisse
        results = []

        # Warnungen (z. B. bei NaNs) ignorieren
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            for col in indicator_cols:
                # Falls Spalte leer ist oder nur NaNs enthält, überspringen
                if self.combined_df[col].dropna().empty:
                    continue

                # Spaltentyp zuordnen
                if col in binary_cols:
                    indicator_type = 'Binär'
                elif col in ordinal_cols:
                    indicator_type = 'Ordinal'
                elif col in continuous_cols:
                    indicator_type = 'Kontinuierlich'
                else:
                    indicator_type = 'Kategorisch'

                try:
                    # === Binär ===
                    if indicator_type == 'Binär':
                        # Point-Biserial-Korrelation
                        corr_pb, p_value_pb = pointbiserialr(self.combined_df[col], target)
                        if np.isnan(corr_pb):
                            corr_pb = 'NaN'
                            p_value_pb = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Point-Biserial Correlation",
                            'Koeffizient': corr_pb,
                            'p-Wert': p_value_pb
                        })

                        # Mutual Information (für binäre Spalte discrete_features=True)
                        mi = mutual_info_regression(
                            self.combined_df[[col]],
                            target,
                            discrete_features=True
                        )[0]
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Mutual Information",
                            'Koeffizient': mi,
                            'p-Wert': 'N/A'
                        })

                    # === Ordinal ===
                    elif indicator_type == 'Ordinal':
                        # Spearman's Rho
                        corr_spear, p_spear = spearmanr(self.combined_df[col], target)
                        if np.isnan(corr_spear):
                            corr_spear = 'NaN'
                            p_spear = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Spearman's Rho",
                            'Koeffizient': corr_spear,
                            'p-Wert': p_spear
                        })

                        # Optional: Pearson's r
                        corr_pear, p_pear = pearsonr(self.combined_df[col], target)
                        if np.isnan(corr_pear):
                            corr_pear = 'NaN'
                            p_pear = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Pearson's r",
                            'Koeffizient': corr_pear,
                            'p-Wert': p_pear
                        })

                        # Mutual Information
                        mi = mutual_info_regression(
                            self.combined_df[[col]],
                            target
                        )[0]
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Mutual Information",
                            'Koeffizient': mi,
                            'p-Wert': 'N/A'
                        })

                    # === Kontinuierlich ===
                    elif indicator_type == 'Kontinuierlich':
                        # Pearson's r
                        corr_pear, p_pear = pearsonr(self.combined_df[col], target)
                        if np.isnan(corr_pear):
                            corr_pear = 'NaN'
                            p_pear = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Pearson's r",
                            'Koeffizient': corr_pear,
                            'p-Wert': p_pear
                        })

                        # Spearman's Rho
                        corr_spear, p_spear = spearmanr(self.combined_df[col], target)
                        if np.isnan(corr_spear):
                            corr_spear = 'NaN'
                            p_spear = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Spearman's Rho",
                            'Koeffizient': corr_spear,
                            'p-Wert': p_spear
                        })

                        # Mutual Information
                        mi = mutual_info_regression(
                            self.combined_df[[col]],
                            target
                        )[0]
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Mutual Information",
                            'Koeffizient': mi,
                            'p-Wert': 'N/A'
                        })

                    # === Kategorisch / Andere ===
                    else:
                        # Spearman's Rho
                        corr_spear, p_spear = spearmanr(self.combined_df[col], target)
                        if np.isnan(corr_spear):
                            corr_spear = 'NaN'
                            p_spear = 'N/A'
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Spearman's Rho",
                            'Koeffizient': corr_spear,
                            'p-Wert': p_spear
                        })

                        # Mutual Information
                        mi = mutual_info_regression(
                            self.combined_df[[col]],
                            target
                        )[0]
                        results.append({
                            'Indicator': col,
                            'Typ': indicator_type,
                            'Methode': "Mutual Information",
                            'Koeffizient': mi,
                            'p-Wert': 'N/A'
                        })

                except Exception as e:
                    print(f"Fehler bei der Berechnung für Spalte '{col}': {e}")

        # Nach Durchlaufen aller Spalten: DataFrame erstellen und sortieren
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=['Indicator', 'Methode'], inplace=True)

        # Ausgabe formatieren
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: f'{x:.6f}' if isinstance(x, float) else x)

        print("\nStatistische Maße zwischen Indikatoren und Next_Close:\n")
        print(results_df.to_string(index=False))

        # Optional: Export als CSV
        if self.export_csv:
            try:
                results_df.to_csv(self.export_path, index=False)
                print(f"\nErgebnisse wurden erfolgreich in '{self.export_path}' exportiert.")
            except Exception as e:
                print(f"Fehler beim Exportieren der Ergebnisse: {e}")


if __name__ == "__main__":
    # Beispielhafter Aufruf (wenn das Skript direkt ausgeführt wird):
    project_dir = r"C:\Users\Anwender\PycharmProjects\Unternehmenssoftware"  # Anpassen an dein Projektverzeichnis
    analyzer = StatisticalAnalyzer(
        project_root=project_dir,
        export_csv=True,
        export_path='statistical_results.csv'
    )
    analyzer.load_combined_csv()
    analyzer.analyze_correlations()
