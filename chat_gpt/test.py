# Notwendige Bibliotheken importieren
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Daten laden
def lade_daten(pfad="/Users/maxbudde/Unternehmenssoftware/sp500_data/SP500_Index_Historical_Data.csv"):
    """
    Lädt die historischen S&P 500 Daten aus einer CSV-Datei.

    Args:
        pfad (str): Der Pfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Das geladene DataFrame mit den Daten.
    """
    daten = pd.read_csv(pfad)
    return daten


# 2. Daten vorbereiten
def bereite_daten_vor(daten, spalten=['Open', 'High', 'Low', 'Close', 'Volume'], sequenz_länge=60):
    """
    Bereitet die Daten für das LSTM-Modell vor. Skalierung, Erstellung von Sequenzen und Aufteilung in Trainings- und Testdaten.

    Args:
        daten (pd.DataFrame): Das geladene DataFrame mit den Daten.
        spalten (list): Die Spalten, die verwendet werden sollen.
        sequenz_länge (int): Die Länge der Sequenzen für das LSTM.

    Returns:
        X_train (torch.Tensor): Trainingsdaten für das Modell.
        y_train (torch.Tensor): Zielwerte für das Training.
        X_test (torch.Tensor): Testdaten für das Modell.
        y_test (torch.Tensor): Zielwerte für den Test.
        scaler (MinMaxScaler): Der verwendete Skalierer zur späteren Denormalisierung.
    """
    # Daten skalieren zwischen 0 und 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    skaliert = scaler.fit_transform(daten[spalten].values)

    X, y = [], []
    for i in range(sequenz_länge, len(skaliert)):
        X.append(skaliert[i - sequenz_länge:i])  # Eingabesequenz
        y.append(skaliert[i, 3])  # Zielwert (Close-Preis)

    X, y = np.array(X), np.array(y)

    # Daten in Training und Test aufteilen (80% Training, 20% Test)
    treningsgröße = int(0.8 * len(X))
    X_train, X_test = X[:treningsgröße], X[treningsgröße:]
    y_train, y_test = y[:treningsgröße], y[treningsgröße:]

    # In Torch-Tensoren umwandeln
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler


# 3. Einfaches LSTM-Modell definieren
class EinfachesLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=50, output_dim=1, num_layers=2):
        """
        Initialisiert das LSTM-Modell.

        Args:
            input_dim (int): Anzahl der Eingabefeatures.
            hidden_dim (int): Anzahl der Neuronen in der versteckten Schicht.
            output_dim (int): Anzahl der Ausgabewerte.
            num_layers (int): Anzahl der LSTM-Schichten.
        """
        super(EinfachesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchlauf des Modells.

        Args:
            x (torch.Tensor): Eingabedaten.

        Returns:
            torch.Tensor: Die Vorhersage des Modells.
        """
        out, _ = self.lstm(x)  # LSTM-Schicht durchlaufen
        out = out[:, -1, :]  # Letzten Zeitschritt nehmen
        out = self.fc(out)  # Ausgabe durch die vollverbundene Schicht
        return out


# 4. Modell trainieren
def trainiere_modell(modell, daten_loader, verlustfunktion, optimierer, gerät, epochen=100):
    """
    Trainiert das LSTM-Modell.

    Args:
        modell (nn.Module): Das zu trainierende Modell.
        daten_loader (DataLoader): DataLoader für die Trainingsdaten.
        verlustfunktion: Die Verlustfunktion (z.B. MSE).
        optimierer: Der Optimierer (z.B. Adam).
        gerät: Das Gerät (CPU oder GPU).
        epochen (int): Anzahl der Trainingsdurchläufe.

    Returns:
        verlust_werte (list): Liste der Verlustwerte pro Epoche.
    """
    verlust_werte = []
    for epoch in range(epochen):
        modell.train()  # Modell in den Trainingsmodus setzen
        gesamt_verlust = 0
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)

            optimierer.zero_grad()  # Gradienten zurücksetzen
            vorhersage = modell(inputs)  # Vorhersage berechnen
            verlust = verlustfunktion(vorhersage, labels)  # Verlust berechnen
            verlust.backward()  # Rückwärtsdurchlauf (Gradienten berechnen)
            optimierer.step()  # Gewichte aktualisieren

            gesamt_verlust += verlust.item() * inputs.size(0)

        durchschnitt_verlust = gesamt_verlust / len(daten_loader.dataset)
        verlust_werte.append(durchschnitt_verlust)

        # Alle 10 Epochen den Verlust anzeigen
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoche {epoch + 1}/{epochen}, Verlust: {durchschnitt_verlust:.6f}')

    return verlust_werte


# 5. Modell testen
def teste_modell(modell, daten_loader, scaler, gerät):
    """
    Testet das Modell und visualisiert die Ergebnisse.

    Args:
        modell (nn.Module): Das trainierte Modell.
        daten_loader (DataLoader): DataLoader für die Testdaten.
        scaler (MinMaxScaler): Der Skalierer zur Denormalisierung der Daten.
        gerät: Das Gerät (CPU oder GPU).

    Returns:
        tuple: Vorhergesagte und tatsächliche Preise.
    """
    modell.eval()  # Modell in den Evaluierungsmodus setzen
    vorhersagen, tatsächliche = [], []

    with torch.no_grad():  # Kein Gradientenberechnung
        for inputs, labels in daten_loader:
            inputs, labels = inputs.to(gerät), labels.to(gerät)
            outputs = modell(inputs)  # Vorhersage berechnen
            vorhersagen.append(outputs.cpu().numpy())
            tatsächliche.append(labels.cpu().numpy())

    vorhersagen = np.concatenate(vorhersagen).flatten()
    tatsächliche = np.concatenate(tatsächliche).flatten()

    # Daten denormalisieren (zurückskalieren)
    vorhersagen_denorm = scaler.inverse_transform(
        np.hstack((np.zeros((vorhersagen.shape[0], 4)), vorhersagen.reshape(-1, 1)))
    )[:, -1]

    tatsächliche_denorm = scaler.inverse_transform(
        np.hstack((np.zeros((tatsächliche.shape[0], 4)), tatsächliche.reshape(-1, 1)))
    )[:, -1]

    # Ergebnisse plotten
    plt.figure(figsize=(12, 6))
    plt.plot(tatsächliche_denorm, label='Tatsächlicher Preis')
    plt.plot(vorhersagen_denorm, label='Vorhergesagter Preis')
    plt.xlabel('Zeit')
    plt.ylabel('S&P 500 Close Preis')
    plt.title('Tatsächlicher vs. Vorhergesagter S&P 500 Close Preis')
    plt.legend()
    plt.show()

    # RMSE berechnen (Root Mean Square Error)
    rmse = np.sqrt(np.mean((vorhersagen_denorm - tatsächliche_denorm) ** 2))
    print(f'RMSE auf Testdaten: {rmse:.2f}')

    return vorhersagen_denorm, tatsächliche_denorm


#+#z^                                                                                                                                                                                                       # 6. Nächsten Preis vorhersagen und prozentuale Änderung berechnen
def vorhersage_nächste_änderung(modell, aktuelle_daten, scaler, sequenz_länge=60, gerät='cpu'):
    """
    Sagt die prozentuale Änderung des nächsten Close-Preises voraus.

    Args:
        modell (nn.Module): Das trainierte Modell.
        aktuelle_daten (pd.DataFrame): Die aktuellen Daten für die Vorhersage.
        scaler (MinMaxScaler): Der Skalierer zur Skalierung der Daten.
        sequenz_länge (int): Die Länge der Sequenz.
        gerät (str): Das Gerät (CPU oder GPU).

    Returns:
        tuple: Richtung ("hoch" oder "runter") und prozentuale Änderung.
    """
    modell.eval()
    with torch.no_grad():
        # Daten skalieren
        skaliert = scaler.transform(aktuelle_daten)
        # Letzte Sequenz nehmen
        letzte_sequenz = skaliert[-sequenz_länge:]
        letzte_sequenz = torch.tensor(letzte_sequenz, dtype=torch.float32).unsqueeze(0).to(gerät)
        # Vorhersage berechnen
        vorhersage_skaliert = modell(letzte_sequenz).cpu().numpy()
        # Denormalisieren (zurückskalieren)
        vorhersage_denorm = scaler.inverse_transform(
            np.hstack((np.zeros((vorhersage_skaliert.shape[0], 4)), vorhersage_skaliert))
        )[:, -1]

        # Tatsächlicher aktueller Preis
        aktueller_preis = aktuelle_daten['Close'].iloc[-1]

        # Prozentuale Änderung berechnen
        prozentuale_änderung = ((vorhersage_denorm[0] - aktueller_preis) / aktueller_preis) * 100

        # Richtung bestimmen
        richtung = "hoch" if prozentuale_änderung > 0 else "runter"

        return richtung, abs(prozentuale_änderung)


# 7. Hauptfunktion
def haupt():
    """
    Führt den gesamten Prozess aus: Laden der Daten, Vorbereiten, Trainieren des Modells,
    Testen und Vorhersagen.
    """
    # Schritt 1: Daten laden
    print("Daten werden geladen...")
    daten = lade_daten()
    print("Daten erfolgreich geladen.")

    # Schritt 2: Daten vorbereiten
    print("Daten werden vorbereitet...")
    sequenz_länge = 60
    X_train, y_train, X_test, y_test, scaler = bereite_daten_vor(daten, sequenz_länge=sequenz_länge)
    print("Daten erfolgreich vorbereitet.")

    # Schritt 3: Modell erstellen
    print("Erstelle das LSTM-Modell...")
    modell = EinfachesLSTM()
    print("Modell erfolgreich erstellt.")

    # Schritt 4: Hyperparameter festlegen
    epochen = 100
    batch_größe = 64
    lernrate = 0.001
    print(f"Hyperparameter festgelegt: Epochen={epochen}, Batch-Größe={batch_größe}, Lernrate={lernrate}")

    # Schritt 5: DataLoader erstellen
    print("Erstelle DataLoader für Trainings- und Testdaten...")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_größe, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_größe, shuffle=False)
    print("DataLoader erfolgreich erstellt.")

    # Schritt 6: Gerät auswählen
    gerät = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {gerät}")
    modell.to(gerät)

    # Schritt 7: Verlustfunktion und Optimierer definieren
    verlustfunktion = nn.MSELoss()
    optimierer = torch.optim.Adam(modell.parameters(), lr=lernrate)
    print("Verlustfunktion und Optimierer definiert.")

    # Schritt 8: Modell trainieren
    print("Starte das Training des Modells...")
    verlust_werte = trainiere_modell(modell, train_loader, verlustfunktion, optimierer, gerät, epochen=epochen)
    print("Training abgeschlossen.")

    # Schritt 9: Verlustkurve plotten
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochen + 1), verlust_werte, label='Training Verlust')
    plt.xlabel('Epoche')
    plt.ylabel('MSE Verlust')
    plt.title('Training Verlust über Epochen')
    plt.legend()
    plt.show()

    # Schritt 10: Modell testen
    print("Teste das Modell mit den Testdaten...")
    teste_modell(modell, test_loader, scaler, gerät)
    print("Modell erfolgreich getestet.")

    # Schritt 11: Nächste Preisänderung vorhersagen
    print("Mache eine Vorhersage für die nächste Preisänderung...")
    aktuelle_daten = daten[['Open', 'High', 'Low', 'Close', 'Volume']].tail(sequenz_länge)
    richtung, prozent_änderung = vorhersage_nächste_änderung(modell, aktuelle_daten, scaler,
                                                             sequenz_länge=sequenz_länge, gerät=gerät)
    print(f'Der Preis wird voraussichtlich {richtung} gehen um {prozent_änderung:.2f}%.')


# 8. Programm starten
if __name__ == "__main__":
    haupt()