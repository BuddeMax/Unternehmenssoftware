/usr/local/bin/python3.12 /Users/maxbudde/Unternehmenssoftware/Experiment_5/lstm_sp500_with_bb.py 
✅ Verwendetes Gerät: cpu
📥 Daten werden geladen...
✅ Daten erfolgreich geladen.
🔧 Daten werden vorbereitet...
✅ Daten erfolgreich vorbereitet.
🤖 Erstelle das LSTM-Modell...
✅ Modell erfolgreich erstellt und auf das Gerät verschoben.
⚙️ Hyperparameter festgelegt: Epochen=100, Batch-Größe=64, Lernrate=0.001
📦 Erstelle DataLoader für Trainings- und Testdaten...
✅ DataLoader erfolgreich erstellt.
🔍 Verlustfunktion und Optimierer definiert.
🏋️‍♂️ Starte das Training des Modells...
Epoche 1/100, Verlust: 0.003353
Epoche 10/100, Verlust: 0.000017
Epoche 20/100, Verlust: 0.000009
Epoche 30/100, Verlust: 0.000007
Epoche 40/100, Verlust: 0.000005
Epoche 50/100, Verlust: 0.000003
Epoche 60/100, Verlust: 0.000003
Epoche 70/100, Verlust: 0.000002
Epoche 80/100, Verlust: 0.000001
Epoche 90/100, Verlust: 0.000002
Epoche 100/100, Verlust: 0.000001
🏁 Training abgeschlossen.
📈 Plotten der Trainingsverlustkurve...
✅ Trainingsverlustkurve erfolgreich geplottet.
🔬 Testen des Modells mit den Testdaten...
RMSE auf Testdaten: 98.81
Die Vorhersagen und tatsächlichen Werte wurden in lstm_sp500_bb.csv gespeichert.
✅ Modell erfolgreich getestet und Ergebnisse visualisiert.

Process finished with exit code 0

Durchschnittliche prozentuale Abweichung: 2.89%