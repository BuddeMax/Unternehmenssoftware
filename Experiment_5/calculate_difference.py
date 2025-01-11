import pandas as pd
import os


def calculate_percentage_deviation(file_path):
    try:
        # Datei laden
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Die Datei '{file_path}' wurde nicht gefunden.")

        # CSV-Datei einlesen
        df = pd.read_csv(file_path)

        # Sicherstellen, dass die notwendigen Spalten existieren
        required_columns = ['Date', 'Actual_Close', 'Predicted_Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Die Spalte '{col}' fehlt in der CSV-Datei.")

        # Filter für Daten bis 2016
        df['Date'] = pd.to_datetime(df['Date'])
        df_filtered = df[df['Date'] <= '2015-12-18'].copy()

        # Prozentsatz der Abweichung berechnen
        df_filtered['Percentage_Deviation'] = abs(
            (df_filtered['Actual_Close'] - df_filtered['Predicted_Close'])
            / df_filtered['Actual_Close'] * 100
        )

        # Durchschnitt der Abweichung berechnen
        total_deviation = df_filtered['Percentage_Deviation'].sum()
        count = df_filtered['Percentage_Deviation'].count()
        average_deviation = total_deviation / count if count > 0 else 0

        # Ergebnisse zurück in die ursprüngliche CSV schreiben
        df.update(df_filtered)
        df.to_csv(file_path, index=False)

        # Endergebnis anzeigen
        print("Zusammenfassung der Berechnungen bis 2016:")
        print(df_filtered[['Date', 'Actual_Close', 'Predicted_Close', 'Percentage_Deviation']])

        print(f"Gesamte prozentuale Abweichung: {total_deviation:.2f}")
        print(f"Durchschnittliche prozentuale Abweichung: {average_deviation:.2f}")

        print(f"Die Berechnungen wurden erfolgreich abgeschlossen und in die Datei '{file_path}' geschrieben.")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")


# Dynamischer Pfad zur CSV-Datei
file_path = "/Users/maxbudde/Unternehmenssoftware/Experiment_5/lstm_sp500_data/rsi/lstm_sp500_results_5_with_RSI.csv"
calculate_percentage_deviation(file_path)
