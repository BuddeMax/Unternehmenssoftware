/usr/local/bin/python3.12 /Users/maxbudde/Unternehmenssoftware/Experiment_5/lstm_sp500_with_macd.py 
Daten werden geladen...
Daten erfolgreich geladen.
Daten werden vorbereitet...
Daten erfolgreich vorbereitet.
Erstelle das LSTM-Modell...
Modell erfolgreich erstellt.
Hyperparameter festgelegt: Epochen=100, Batch-Größe=64, Lernrate=0.001
Erstelle DataLoader für Trainings- und Testdaten...
DataLoader erfolgreich erstellt.
Verwende Gerät: cpu
Verlustfunktion und Optimierer definiert.
Starte das Training des Modells...
Epoche 1/100, Verlust: 0.005862
Epoche 10/100, Verlust: 0.000014
Epoche 20/100, Verlust: 0.000013
Epoche 30/100, Verlust: 0.000008
Epoche 40/100, Verlust: 0.000009
Epoche 50/100, Verlust: 0.000006
Epoche 60/100, Verlust: 0.000004
Epoche 70/100, Verlust: 0.000003
Epoche 80/100, Verlust: 0.000001
Epoche 90/100, Verlust: 0.000001
Epoche 100/100, Verlust: 0.000001
Training abgeschlossen.
Teste das Modell mit den Testdaten...
RMSE auf Testdaten: 213.66
Ergebnisse wurden in vorhersagen.csv gespeichert.
Modell erfolgreich getestet.

Process finished with exit code 0

Durchschnittliche prozentuale Abweichung: 1.95%
