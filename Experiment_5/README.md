# 📊 SP500 Schlusskurs-Vorhersage mit LSTM und RNN

## 🔍 Erweiterung: Analyse der Indikatoren für statistische Bewertung

In diesem Abschnitt wurden die Indikatoren Bollinger Bands, MACD, RSI und Volume untersucht, um herauszufinden, welche Merkmale am besten geeignet sind, die Schlusskurse des SP500 für das LSTM-Modell vorherzusagen. Die Analyse basiert auf drei statistischen Maßen: Pearson's R, Spearman's Rho und normierte Mutual Information (NMI).

---

## 📑 Statistische Maße

- **Normalized Mutual Information (NMI):** Misst die Informationsmenge, die ein Indikator über die Zielgröße liefert. Werte reichen von 0 (keine Abhängigkeit) bis 1 (perfekte Abhängigkeit). Besonders nützlich, um allgemeine Abhängigkeiten zu bewerten.
- **Pearson's R:** Bewertet die lineare Abhängigkeit zwischen Indikator und Zielgröße. Werte reichen von -1 (perfekte negative Korrelation) bis +1 (perfekte positive Korrelation). Nahe 0 bedeutet keine lineare Beziehung.
- **Spearman's Rho:** Bewertet die monotone Abhängigkeit zwischen Indikator und Zielgröße. Werte wie bei Pearson's R zwischen -1 und +1. Besonders nützlich bei nicht-linearen Beziehungen.

---

## 📊 Ergebnisse der Analyse

### **1. Bollinger Bands**

| **Indikator**                  | **Statistisches Maß**  | **Koeffizient** |
|---------------------------------|------------------------|------------------|
| Bollinger Band Upper Breakstreak | Normalized MI (NMI)    | 0.0              |
| Bollinger Band Upper Breakstreak | Pearson's r           | -0.0268          |
| Bollinger Band Upper Breakstreak | Spearman's Rho        | -0.0240          |

**Bedeutung:** Die Bollinger Bands zeigen keine signifikante Abhängigkeit zu den Schlusskursen. Mit einem NMI von 0.0 und niedrigen Werten für R und Rho scheint dieser Indikator wenig prädiktive Kraft für das LSTM-Modell zu besitzen.

---

### **2. MACD**

| **Indikator**                  | **Statistisches Maß**  | **Koeffizient** |
|---------------------------------|------------------------|------------------|
| MACD                           | Normalized MI (NMI)    | 0.5804           |
| MACD                           | Pearson's r           | 0.2782           |
| MACD                           | Spearman's Rho        | 0.3228           |

**Bedeutung:** Der MACD ist ein starker Indikator für die Schlusskursvorhersage, insbesondere durch seinen hohen NMI-Wert von 0.5804. Auch Pearson's R und Spearman's Rho zeigen moderate Abhängigkeiten, was die Relevanz dieses Indikators für das LSTM-Modell unterstützt.

---

### **3. RSI**

| **Indikator**                  | **Statistisches Maß**  | **Koeffizient** |
|---------------------------------|------------------------|------------------|
| RSI Oversold Streak            | Normalized MI (NMI)    | 0.0039           |
| RSI Oversold Streak            | Pearson's r           | -0.0386          |
| RSI Oversold Streak            | Spearman's Rho        | -0.0520          |

**Bedeutung:** Der RSI Oversold Streak zeigt eine sehr geringe Abhängigkeit zu den Schlusskursen. Mit einem NMI von 0.0039 und negativen Werten für R und Rho scheint dieser Indikator wenig hilfreich für die Modellierung im LSTM zu sein.

---

### **4. Volume**

| **Indikator**                  | **Statistisches Maß**  | **Koeffizient** |
|---------------------------------|------------------------|------------------|
| Volume                         | Normalized MI (NMI)    | 1.0              |
| Volume                         | Pearson's r           | 0.6885           |
| Volume                         | Spearman's Rho        | 0.8639           |

**Bedeutung:** Volume ist der stärkste Indikator in dieser Analyse. Mit einem perfekten NMI von 1.0 sowie hohen Werten bei Pearson's R (0.6885) und Spearman's Rho (0.8639) liefert Volume signifikante Informationen über die Schlusskursentwicklung und ist ein unverzichtbarer Bestandteil für das LSTM-Modell.

---

## 🎯 Bedeutung der Ergebnisse für das LSTM-Modell

1. **Volume** ist der wichtigste Indikator und sollte im LSTM-Modell priorisiert integriert werden. Seine hohe Informationsdichte (NMI: 1.0) und starke lineare sowie monotone Abhängigkeiten machen ihn besonders wertvoll.
2. **MACD** liefert ebenfalls starke prädiktive Hinweise, insbesondere durch den hohen NMI-Wert (0.5804). Er kann als ergänzender Indikator verwendet werden, um zusätzliche Muster zu erkennen.
3. **Bollinger Bands und RSI** zeigen in dieser Analyse keine signifikanten Zusammenhänge und sind für die Vorhersage weniger relevant. Sie könnten jedoch in Kombination mit anderen Indikatoren nützlich sein und sollten bei Bedarf weiter untersucht werden.

### Fazit
Die Ergebnisse dieser Analyse bieten eine klare Grundlage für die Auswahl der wichtigsten Indikatoren. Volume und MACD werden als Hauptmerkmale für das LSTM-Modell priorisiert, da sie statistisch signifikante Zusammenhänge mit den Schlusskursen aufweisen und das Potenzial haben, die Vorhersagegenauigkeit erheblich zu verbessern.
