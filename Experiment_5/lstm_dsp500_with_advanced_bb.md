/usr/local/bin/python3.12 /Users/maxbudde/Unternehmenssoftware/Experiment_5/lstm_sp500_with_advanced_bb.py 
Daten werden geladen...
Spalten nach dem Merge: ['Date', 'Close_bb', 'Moving_Avg', 'Upper_Band', 'Lower_Band', 'Bandwidth', 'Percent_B', 'Price_Deviation', 'Slope', 'Divergence', 'Close_sp500', 'Volume_normal', 'Open', 'High', 'Low', 'Close', 'Volume_bb', 'BB_Middle', 'BB_Upper', 'BB_Lower']
Daten erfolgreich geladen. Gesamtanzahl der Datenpunkte: 11302
Daten werden vorbereitet...
Daten erfolgreich vorbereitet.
Erstelle das LSTM-Modell...
Modell erfolgreich erstellt mit input_dim=8.
Hyperparameter festgelegt: Epochen=100, Batch-Größe=64, Lernrate=0.001
Erstelle DataLoader für Trainings- und Testdaten...
DataLoader erfolgreich erstellt.
Verwende Gerät: cpu
Verlustfunktion und Optimierer definiert.
Starte das Training des Modells...
Epoche 1/100, Verlust: 0.000749
Epoche 10/100, Verlust: 0.000014
Epoche 20/100, Verlust: 0.000009
Epoche 30/100, Verlust: 0.000008
Epoche 40/100, Verlust: 0.000007
Epoche 50/100, Verlust: 0.000005
Epoche 60/100, Verlust: 0.000004
Epoche 70/100, Verlust: 0.000004
Epoche 80/100, Verlust: 0.000005
Epoche 90/100, Verlust: 0.000004
Epoche 100/100, Verlust: 0.000005
Training abgeschlossen.
Teste das Modell mit den Testdaten...
RMSE auf Testdaten: 178.08
Ergebnisse wurden in ergebnisse.csv gespeichert.
Modell erfolgreich getestet.

Process finished with exit code 0
