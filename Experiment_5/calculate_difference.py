import pandas as pd

# Pfad zur CSV-Datei
file_path = '/Users/maxbudde/Unternehmenssoftware/Experiment_5/lstm_sp500_with_rsi.csv'

# CSV-Datei laden
df = pd.read_csv(file_path)

# Sicherstellen, dass die benötigten Spalten vorhanden sind
if {'Vorhergesagt', 'Tatsächlich'}.issubset(df.columns):
    # Prozentuale Abweichung berechnen
    df['Prozentuale_Abweichung'] = ((df['Tatsächlich'] - df['Vorhergesagt']) / df['Vorhergesagt']) * 100

    # Durchschnittliche prozentuale Abweichung berechnen
    durchschnittliche_abweichung = df['Prozentuale_Abweichung'].mean()

    # Alle Daten anzeigen
    pd.set_option('display.max_rows', None)  # Zeigt alle Zeilen an
    pd.set_option('display.max_columns', None)  # Zeigt alle Spalten an
    pd.set_option('display.expand_frame_repr', False)  # Verhindert Zeilenumbruch für Spalten
    print(df)

    print(f"\nDurchschnittliche prozentuale Abweichung: {durchschnittliche_abweichung:.2f}%")
else:
    print("Die CSV-Datei muss die Spalten 'Vorhergesagt' und 'Tatsächlich' enthalten.")
