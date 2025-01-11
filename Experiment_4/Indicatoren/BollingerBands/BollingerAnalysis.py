from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import chi2_contingency, spearmanr
import warnings


# Funktion zum Finden der CSV-Dateien im Projektverzeichnis und allen Unterverzeichnissen
def find_csv_file(file_name):
    project_dir = Path(__file__).resolve().parents[3]  # Geht drei Ebenen hoch – anpassen, wenn nötig.
    for file_path in project_dir.rglob(file_name):
        if file_path.is_file():
            return file_path
    print(f"Datei '{file_name}' wurde im Projektverzeichnis nicht gefunden.")
    return None


# Funktion zum Laden der CSV-Dateien
def load_data(file_name):
    csv_path = find_csv_file(file_name)
    if csv_path:
        return pd.read_csv(csv_path, parse_dates=['Date'])
    else:
        raise FileNotFoundError(f"Datei '{file_name}' nicht gefunden.")


# Daten laden
bollinger_df = load_data("bollinger_bands_2.csv")
historical_df = load_data("SP500_Index_Historical_Data.csv")

# Daten zusammenführen
merged_df = pd.merge(bollinger_df, historical_df, on="Date", how="inner")

# Zeitverschiebung für die Zielvariable (Close-Kurs Differenz)
merged_df['Delta_Close'] = merged_df['Close_y'].diff()


# 1. Korrelation (Pearsons r und Spearman)
def calculate_correlations(df):
    correlations = {'Pearson': {}, 'Spearman': {}}
    for lag in range(1, 6):
        df[f"Lag_{lag}"] = df['Moving_Avg'].shift(lag)
        pearson_corr = df[[f"Lag_{lag}", "Delta_Close"]].corr().iloc[0, 1]
        spearman_corr, _ = spearmanr(df[f"Lag_{lag}"], df['Delta_Close'])
        correlations['Pearson'][f"Lag_{lag}"] = pearson_corr
        correlations['Spearman'][f"Lag_{lag}"] = spearman_corr
    return correlations


correlations = calculate_correlations(merged_df)


# Visualisierung der Korrelationen
def plot_correlations(correlations):
    df_corr = pd.DataFrame(correlations)
    df_corr.plot(kind='bar', figsize=(12, 7), color=['skyblue', 'orange'])
    plt.title("Pearson und Spearman Korrelationen zwischen Bollinger-Bands und Delta Close")
    plt.xlabel("Lag")
    plt.ylabel("Korrelation")
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.legend(title='Korrelationstyp')
    plt.tight_layout()
    plt.show()


plot_correlations(correlations)


# 2. Mutual Information (MI)
def calculate_mutual_information(df):
    mi_values = {}
    for lag in range(1, 6):
        lagged_data = df[[f"Lag_{lag}", "Delta_Close"]].dropna()
        mi = mutual_info_regression(lagged_data[[f"Lag_{lag}"]], lagged_data['Delta_Close'], random_state=0)[0]
        mi_values[f"Lag_{lag}"] = mi
    return mi_values


mutual_info = calculate_mutual_information(merged_df)


# Visualisierung von Mutual Information
def plot_mutual_information(mi_values):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(mi_values.keys()), y=list(mi_values.values()), palette='viridis')
    plt.title("Mutual Information zwischen Bollinger-Bands und Delta Close")
    plt.xlabel("Lag")
    plt.ylabel("Mutual Information")
    plt.tight_layout()
    plt.show()


plot_mutual_information(mutual_info)


# 3. Granger-Kausalität
def perform_granger_causality(df):
    granger_results = {}
    warnings.filterwarnings("ignore", category=FutureWarning)
    for lag in range(1, 6):
        test_result = grangercausalitytests(df[['Delta_Close', f"Lag_{lag}"]].dropna(), maxlag=lag, verbose=False)
        granger_p = test_result[lag][0]['ssr_ftest'][1]
        granger_results[f"Lag_{lag}"] = granger_p
    warnings.filterwarnings("default", category=FutureWarning)
    return granger_results


granger_p_values = perform_granger_causality(merged_df)


# Visualisierung der Granger-Kausalität
def plot_granger_causality(granger_p_values):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(granger_p_values.keys()), y=list(granger_p_values.values()), palette='magma')
    plt.title("Granger-Kausalität zwischen Bollinger-Bands und Delta Close")
    plt.xlabel("Lag")
    plt.ylabel("p-Wert")
    plt.axhline(y=0.05, color='red', linestyle='--', label='Signifikanzniveau (0.05)')
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_granger_causality(granger_p_values)


# 4. Chi²-Test
def perform_chi_squared_test(df):
    # Erstelle eine explizite Kopie, um SettingWithCopyWarning zu vermeiden
    df = df.dropna(subset=['Lower_Band', 'Upper_Band', 'Close_x']).copy()

    # Definiere die Bollinger-Bands Kategorien
    bins = [-np.inf, df['Lower_Band'].min(), df['Upper_Band'].max(), np.inf]
    labels = ['OS', 'NB', 'OB']
    df.loc[:, 'BB_Category'] = pd.cut(df['Close_x'], bins=bins, labels=labels)

    # Definiere die Delta Close Kategorie
    df.loc[:, 'Delta_Close_Category'] = (df['Delta_Close'] > 0).astype(int)

    # Erstelle die Kontingenztabelle
    contingency_table = pd.crosstab(df['BB_Category'], df['Delta_Close_Category'])

    # Führe den Chi²-Test durch
    chi2, p, _, _ = chi2_contingency(contingency_table)

    return contingency_table, chi2, p


contingency_table, chi2_stat, chi2_p_value = perform_chi_squared_test(merged_df)


# Visualisierung der Kontingenztabelle
def plot_contingency_table(contingency_table):
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Kontingenztabelle von Bollinger-Bands-Kategorien und Delta Close")
    plt.xlabel("Delta Close Kategorie")
    plt.ylabel("Bollinger-Bands Kategorie")
    plt.tight_layout()
    plt.show()


plot_contingency_table(contingency_table)


# Ergebnisse in CSV speichern
def save_results(df, correlations, mutual_info, granger_p_values, contingency_table):
    output_dir = Path(__file__).resolve().parent / "analysis_results"
    output_dir.mkdir(exist_ok=True)

    # Gesammelte Daten speichern
    merged_output_path = output_dir / "merged_data_with_indicators.csv"
    df.to_csv(merged_output_path, index=False)

    # Pearson und Spearman Korrelationen speichern
    pearson_corr_df = pd.DataFrame.from_dict(correlations['Pearson'], orient='index', columns=['Pearson Correlation'])
    spearman_corr_df = pd.DataFrame.from_dict(correlations['Spearman'], orient='index',
                                              columns=['Spearman Correlation'])
    correlations_df = pearson_corr_df.join(spearman_corr_df)
    correlations_df.to_csv(output_dir / "correlations.csv")

    # Mutual Information speichern
    mutual_info_df = pd.DataFrame.from_dict(mutual_info, orient='index', columns=['Mutual Information'])
    mutual_info_df.to_csv(output_dir / "mutual_information.csv")

    # Granger-Kausalität speichern
    granger_df = pd.DataFrame.from_dict(granger_p_values, orient='index', columns=['Granger p-Value'])
    granger_df.to_csv(output_dir / "granger_causality.csv")

    # Chi²-Kontingenztabelle speichern
    contingency_table.to_csv(output_dir / "chi_squared_table.csv")


save_results(merged_df, correlations, mutual_info, granger_p_values, contingency_table)

print("Analysen abgeschlossen. Ergebnisse gespeichert im Ordner: analysis_results")
