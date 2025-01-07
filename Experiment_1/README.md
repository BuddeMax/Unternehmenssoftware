# 📊 SP500 Schlusskurs-Vorhersage mit LSTM und RNN

## Kurzbeschreibung
Dieses Teilprojekt beschäftigt sich mit der Vorhersage von SP500-Schlusskursen unter Verwendung 
von LSTM (Long Short-Term Memory) und RNN (Recurrent Neural Network) Modellen. Es stellt den Anfang 
eines größeren Projekts dar, das sich mit Zeitreihenprognosen zur Finanzdatenanalyse beschäftigt.

---

## 📅 Datenerfassung
Die historischen SP500-Daten stammen aus einer CSV-Datei namens `SP500_Index_Historical_Data.csv`. 
Sie enthält tägliche Aktieninformationen, darunter:

- **Datum**
- **Eröffnungskurs**
- **Höchstkurs**
- **Tiefstkurs**
- **Schlusskurs**
- **Handelsvolumen**

Der Datensatz umfasst den Zeitraum von **1994 bis 2024**, wobei:
- **Trainingsdaten:** 1994–2015
- **Testdaten:** 2016–2024

---

## 📊 Merkmale
Die folgenden Merkmale wurden extrahiert und mithilfe von `MinMaxScaler` skaliert:

- **Eröffnungskurs (Open):** Eröffnungspreis des Index
- **Höchstkurs (High):** Höchstpreis des Index während des Tages
- **Tiefstkurs (Low):** Tiefstpreis des Index während des Tages
- **Schlusskurs (Close):** Schlusskurs des Index
- **Handelsvolumen (Volume):** Anzahl der gehandelten Anteile

**Zielvariable:**
- **Schlusskurs (Close):** Der tägliche Schlusswert des SP500-Index

---

## 🛠️ Modellarchitektur

### LSTM-Modell:
- **Parameter:** 5 (Open, High, Low, Close, Volume)
- **Hidden-Size:** 128
- **Anzahl der Schichten:** 4
- **Dropout:** 0.3
- **Bidirektional:** Ja
- **Verlustfunktion:** SmoothL1Loss
- **Optimierer:** Adam
- **Lernratenplaner:** StepLR

### RNN-Modell:
- **Parameter:** 5
- **Hidden-Size:** 64
- **Anzahl der Schichten:** 4
- **Verlustfunktion:** MSELoss
- **Optimierer:** Adam

---

## 📈 Leistungskriterien
- **Trainingsverlust:** Über die Epochen hinweg überwacht
- **Testverlust (MSE):** Bewertet die Modellleistung auf unbekannten Daten
- **Visualisierung:** Prognostizierte vs. tatsächliche Schlusskurse werden zur Vergleichbarkeit
geplottet

---

## 🚀 Ausgangspunkt
Als Ausgangspunkt wurde ein einfaches **RNN-Modell** implementiert. Das **LSTM-Modell** zeigte eine
deutlich bessere Leistung bei der Erkennung langfristiger Abhängigkeiten und Reduzierung des 
Vorhersagefehlers.

---

## 📊 Ergebnisse
- Die **LSTM-Vorhersagen** zeigen eine bessere Übereinstimmung mit den tatsächlichen Aktienkursen im 
Vergleich zum RNN.
- Die visuelle Gegenüberstellung verdeutlicht, dass **LSTM zeitliche Abhängigkeiten effektiver abbildet**.

**Visualisierungen:**
- `rnn_sp500.png`: Prognosen des RNN-Modells vs. tatsächliche Werte.
- `lstm_sp500.png`: Prognosen des LSTM-Modells vs. tatsächliche Werte.
