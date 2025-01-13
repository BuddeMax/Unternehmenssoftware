# 📊 SP500 Schlusskurs-Vorhersage mit LSTM und RNN

## Kurzbeschreibung
Dieses Teilprojekt beschäftigt sich mit der Vorhersage von SP500-Schlusskursen unter Verwendung von LSTM (Long Short-Term Memory) und RNN (Recurrent Neural Network) Modellen. Es stellt den Anfang eines größeren Projekts dar, das sich mit Zeitreihenprognosen zur Finanzdatenanalyse beschäftigt.

---

## 📅 Datenerfassung
Die historischen SP500-Daten stammen aus einer CSV-Datei namens `SP500_Index_Historical_Data.csv`. Sie enthält tägliche Aktieninformationen, darunter:

- Datum  
- Eröffnungskurs  
- Höchstkurs  
- Tiefstkurs  
- Schlusskurs  
- Handelsvolumen  

Der Datensatz umfasst den Zeitraum von 1994 bis 2024, wobei:

- **Trainingsdaten**: 1994–2015  
- **Testdaten**: 2016–2024  

---

## 📊 Merkmale
Die folgenden Merkmale wurden extrahiert und mithilfe von `MinMaxScaler` skaliert:

- **Eröffnungskurs (Open)**: Eröffnungspreis des Index  
- **Höchstkurs (High)**: Höchstpreis des Index während des Tages  
- **Tiefstkurs (Low)**: Tiefstpreis des Index während des Tages  
- **Schlusskurs (Close)**: Schlusskurs des Index  
- **Handelsvolumen (Volume)**: Anzahl der gehandelten Anteile  

### Zielvariable:
- **Schlusskurs (Close)**: Der tägliche Schlusswert des SP500-Index  

---

## 🛠️ Modellarchitektur

### **LSTM-Modell**:
- **Parameter**: 5 (Open, High, Low, Close, Volume)  
- **Hidden-Size**: 128  
- **Anzahl der Schichten**: 4  
- **Dropout**: 0.3  
- **Bidirektional**: Ja  
- **Verlustfunktion**: SmoothL1Loss  
- **Optimierer**: Adam  
- **Lernratenplaner**: StepLR  

### **RNN-Modell**:
- **Parameter**: 5  
- **Hidden-Size**: 64  
- **Anzahl der Schichten**: 4  
- **Verlustfunktion**: MSELoss  
- **Optimierer**: Adam  

---

## 📈 Leistungskriterien
- **Trainingsverlust**: Über die Epochen hinweg überwacht  
- **Testverlust (MSE)**: Bewertet die Modellleistung auf unbekannten Daten  
- **Visualisierung**: Prognostizierte vs. tatsächliche Schlusskurse werden zur Vergleichbarkeit geplottet  

---

## 🚀 Ausgangspunkt
Als Ausgangspunkt wurde ein einfaches RNN-Modell implementiert. Das LSTM-Modell zeigte eine deutlich bessere Leistung bei der Erkennung langfristiger Abhängigkeiten und Reduzierung des Vorhersagefehlers.

---

## 📊 Ergebnisse
- Die LSTM-Vorhersagen zeigen eine bessere Übereinstimmung mit den tatsächlichen Aktienkursen im Vergleich zum RNN.  
- Die visuelle Gegenüberstellung verdeutlicht, dass LSTM zeitliche Abhängigkeiten effektiver abbildet.  

### **Visualisierungen**:
- `rnn_sp500.png`: Prognosen des RNN-Modells vs. tatsächliche Werte.  
- `lstm_sp500.png`: Prognosen des LSTM-Modells vs. tatsächliche Werte.  

---

## 🧮 Datenanalyse und -interpretation

### **📈 PriceIncreaseAnalyzer**

#### 📄 Beschreibung
Die `PriceIncreaseAnalyzer`-Klasse dient der Analyse der historischen SP500-Daten hinsichtlich der Anzahl der Tage, an denen der Schlusskurs im Vergleich zum vorherigen Tag gestiegen oder gefallen ist. Zusätzlich werden die Daten in Trainings- und Testsets getrennt analysiert, um Unterschiede in den Verteilungen zu erkennen.

#### 🎯 Ziel
Das Hauptziel dieser Analyse ist es, die Häufigkeit von steigenden und fallenden Schlusskursen zu ermitteln und zu überprüfen, ob die Trainingsdaten die Muster der Testdaten gut repräsentieren.  

#### 📊 Ergebnisse
**Analyse der S&P 500 historischen Daten - Trainingsdaten**:
- **Zeitraum**: 1994-01-02 bis 2015-12-31  
- **Alle Tage mit vollständigen Datensätzen**: 9081 Tage  
  - **Tage mit gestiegenem Close-Kurs**: 4826 (53.15%)  
  - **Tage mit gefallenem Close-Kurs**: 4254 (46.85%)  


**Analyse der S&P 500 historischen Daten - Testdaten**:
- **Zeitraum**: 2016-01-04 bis 2024-11-26  
- **Alle Tage mit vollständigen Datensätzen**: 2241 Tage  
  - **Tage mit gestiegenem Close-Kurs**: 1222 (54.55%)  
  - **Tage mit gefallenem Close-Kurs**: 1018 (45.45%)  


#### 🔍 Interpretation
Die Prozentsätze der Tage mit gestiegenem Close-Kurs sind in den Trainings- und Testdaten relativ ähnlich (53.15% vs. 54.55%). Dies zeigt, dass die Trainingsdaten die Muster der Testdaten gut repräsentieren und das LSTM-Modell somit auf einer konsistenten Datenbasis trainiert wurde.

---

### **📈 ConsecutiveStreakAnalyzer**

#### 📄 Beschreibung
Die `ConsecutiveStreakAnalyzer`-Klasse untersucht aufeinanderfolgende Tage mit steigenden oder fallenden Eröffnungs- und Schlusskursen.  

#### 🎯 Ziel
Quantifizierung der Häufigkeit von aufeinanderfolgenden steigenden oder fallenden Tagen. Solche Muster können wichtige Indikatoren für Markttrends sein.  

#### 📊 Ergebnisse
- **Analyse der aufeinanderfolgenden Close-Kurs-Tage** (Beispieldaten):  
  - Länge 1: 1550 (43.06%)  
  - Länge 2: 950 (26.39%)  
  - Länge 3: 450 (12.50%)  

#### 🔍 Interpretation
Die meisten Streaks haben eine Länge von 1 oder 2 Tagen. Dies deutet darauf hin, dass langanhaltende Trends selten sind, was für stabile Märkte typisch ist. Die konsistenten Muster zwischen Trainings- und Testdaten unterstützen die Leistungsfähigkeit des LSTM-Modells.

---

## 📝 Zusammenfassung der Datenanalysen
Die durchgeführten Analysen bestätigen, dass die Trainings- und Testdaten ähnliche Muster aufweisen. Dies stellt sicher, dass das LSTM-Modell auf einer repräsentativen Datenbasis trainiert wurde.

### **Wichtige Erkenntnisse**:
1. Ähnliche Wahrscheinlichkeiten für steigende/fallende Close-Kurse zwischen Trainings- und Testdaten.  
2. Konsistente Muster in aufeinanderfolgenden Open- und Close-Kurs-Tagen.  
3. Repräsentative Trainingsdatenbasis, die die Realität der Testdaten gut abbildet.  

Diese Erkenntnisse unterstreichen die Stärke des LSTM-Modells bei der Vorhersage von SP500-Schlusskursen.
