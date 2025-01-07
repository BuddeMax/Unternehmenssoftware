# 📊 SP500 Open-Preis-Vorhersage mit LSTM

## 📝 **Kurzbeschreibung**
Dieses Teilprojekt konzentriert sich auf die Vorhersage des Open-Preises des S&P 500 Index mithilfe eines **Long Short-Term Memory (LSTM)**-Modells. 
Ziel ist es, mithilfe von historischen Finanzdaten zukünftige Open-Preise präzise vorherzusagen. Als Ausgangspunkt haben wir das LTSM genommen aus 
dem Experiment 1 und noch einige kleine Anpassungen vorgenommen.

---

## 📅 **Datenerfassung**
- **Datenquelle:** Historische SP500-Daten (CSV-Datei)
- **Enthaltene Informationen:**
  - Datum
  - Eröffnungskurs (**Open**)
  - Höchstkurs (**High**)
  - Tiefstkurs (**Low**)
  - Schlusskurs (**Close**)
  - Handelsvolumen (**Volume**)
- **Zeitraum der Daten:**
  - **Trainingsdaten:** 80% der Daten (ältere Jahre)
  - **Testdaten:** 20% der Daten (neuere Jahre)
- **Vorverarbeitung:**
  - Chronologische Sortierung
  - Skalierung mit **MinMaxScaler** auf den Bereich [0, 1]

---

## 📊 **Merkmale**
- **Eingabe-Features:**
  - Open: Eröffnungspreis
  - High: Höchstpreis
  - Low: Tiefstpreis
  - Close: Schlusskurs
  - Volume: Handelsvolumen
- **Zielvariable:**
  - Open-Preis für den nächsten Tag

---

## 🛠️ **Modellarchitektur**
- **Modelltyp:** LSTM
- **Parameter:**
  - Eingabefunktionen: 5
  - Hidden-Size: 50
  - Anzahl der Schichten: 2
  - Dropout: Nicht verwendet
  - Verlustfunktion: **MSELoss (Mean Squared Error Loss)**
  - Optimierer: **Adam**
  - Lernrate: 0.001
  - Epochen: 100
  - Batch-Größe: 64
- **Ziel:** Vorhersage des Open-Preises basierend auf den vorherigen 60 Tagen

---

## 📈 **Leistungskriterien**
- **Trainingsverlust:** Über die Epochen hinweg überwacht
- **Testverlust (MSE):** Bewertet auf unbekannten Daten
- **Visualisierung:** Tatsächliche vs. vorhergesagte Open-Preise
- **Leistungsmetriken:**
  - **RMSE:** 80.52
  - **MAE:** 47.32
  - **MAPE:** 1.08%

---

## 🚀 **Ausgangspunkt**
Als Ausgangspunkt wurde ein Standard-LSTM-Modell mit zwei LSTM-Schichten und einer vollständig verbundenen Schicht verwendet. Das Modell konnte die komplexen zeitlichen Muster der Finanzdaten effektiv erfassen.

---

## 📊 **Ergebnisse**
- Die LSTM-Vorhersagen zeigen eine enge Übereinstimmung mit den tatsächlichen Open-Preisen.
- Die visuelle Darstellung verdeutlicht die Fähigkeit des Modells, kurz- und langfristige Muster zu erkennen.

**Dateien:**
- 📊 **lstm_sp500.png:** Tatsächliche vs. vorhergesagte Open-Preise
- 📄 **vorhersagen.csv:** Enthält tatsächliche und vorhergesagte Open-Preise mit Datumsangaben

