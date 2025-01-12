# 📊 SP500 Schlusskurs-Vorhersage mit LSTM und RNN

## 🔍 Erweiterung: Analyse der Indikatoren für statistische Bewertung

In diesem Abschnitt wurden verschiedene Indikatoren auf ihre statistische Relevanz hin untersucht, um die geeignetsten Merkmale für die spätere Integration in das LSTM-Modell zu bestimmen. Ziel ist es, datengetriebene Entscheidungen über die besten Indikatoren zu treffen, die die Vorhersagefähigkeit verbessern können.

---

## 📑 Statistische Maße

Drei statistische Maße wurden verwendet, um die Qualität der Indikatoren zu bewerten:

- **Pearson's R (Korrelation):** Misst die Stärke und Richtung des linearen Zusammenhangs zwischen einem Indikator und der Zielgröße. Werte reichen von -1 bis +1.
- **Spearman's Rho (Rangkorrelation):** Bewertet die monotone Abhängigkeit, unabhängig von Linearität. Werte liegen ebenfalls zwischen -1 und +1.
- **Mutual Information (MI):** Misst die gemeinsame Abhängigkeit zwischen zwei Variablen in Bezug auf die Unsicherheitsreduktion. Normierte Werte reichen von 0 (keine Abhängigkeit) bis 1 (perfekte Abhängigkeit).

---

## 📊 Ergebnisse der Analyse

### Übersicht der besten Indikatoren für jedes Maß

#### **Bollinger Bands**

| **Typ**            | **Maß**           | **Wert**    |
|---------------------|-------------------|-------------|
| BB_UpperBreak_Streak | Pearson's R      | -0.0268     |
| BB_UpperBreak_Streak | Spearman's Rho   | -0.0240     |
| BB_Squeeze (Binär)  | Normierte MI      | 0.0296      |

#### **MACD**

| **Typ**             | **Maß**           | **Wert**    |
|----------------------|-------------------|-------------|
| MACD_Bullish_Streak | Pearson's R       | -0.0308     |
| MACD (Kontinuierlich)| Spearman's Rho   | 0.3228      |
| Signal_Line         | Normierte MI      | 0.6157      |

#### **RSI**

| **Typ**            | **Maß**           | **Wert**    |
|---------------------|-------------------|-------------|
| RSI (Kontinuierlich)| Pearson's R       | 0.0823      |
| RSI_Oversold_Streak | Spearman's Rho    | -0.0520     |
| RSI (Kontinuierlich)| Normierte MI      | 0.0778      |

#### **Volume**

| **Typ**  | **Maß**           | **Wert**    |
|----------|-------------------|-------------|
| Volume   | Pearson's R       | 0.6885      |
| Volume   | Spearman's Rho    | 0.8639      |
| Volume   | Normierte MI      | 1.0000      |

---

## 📌 Fazit und Zielsetzung

### **Zusammenfassung der Ergebnisse**
1. **Volume** zeigt durchweg die stärksten Werte in allen drei statistischen Maßen, insbesondere bei normierter MI (1.0000) und Spearman's Rho (0.8639).
2. **Signal_Line** und **MACD** (Spearman's Rho: 0.3228) sind ebenfalls starke Indikatoren.
3. **RSI** hat moderate Korrelationen, wobei **RSI_Oversold_Streak** (-0.0520) die stärkste Rangkorrelation liefert.
4. **Bollinger Bands** liefern im Vergleich niedrigere Werte, wobei **BB_Squeeze** mit normierter MI (0.0296) den besten Beitrag leistet.

### **Nächste Schritte**
- Die besten Indikatoren (Volume, Signal_Line, MACD) werden priorisiert in das LSTM-Modell integriert, um deren Vorhersagefähigkeit zu evaluieren.
- Schwächere Indikatoren wie Bollinger Bands und RSI werden ergänzend getestet, um mögliche synergetische Effekte zu prüfen.

### **Präsentationsziel**
Die Ergebnisse zeigen, welche Indikatoren die Zielgröße statistisch am besten erklären. Besonders **Volume** und **Signal_Line** stechen hervor und werden als Hauptmerkmale für die nächste Phase des Projekts priorisiert.
