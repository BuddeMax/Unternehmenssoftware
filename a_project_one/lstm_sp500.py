import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Prüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")


def load_data(file_path):
    """Daten laden und sortieren."""
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data = data.sort_values(by='Date').reset_index(drop=True)
    return data


def split_data(data, train_start, train_end, test_start, test_end):
    """Daten in Trainings- und Testsets aufteilen."""
    train = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)]
    test = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)]
    return train, test


def scale_data(train, test, features):
    """Daten skalieren."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[features])
    test_scaled = scaler.transform(test[features])
    return train_scaled, test_scaled, scaler


def create_sequences(data, seq_length, feature_index=3):
    """Sequenzen und Labels erstellen."""
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, feature_index])  # 'Close' ist an Index 3
    return np.array(sequences), np.array(labels)


class StockLSTM(nn.Module):
    """Optimiertes LSTM-Modell für Aktienvorhersagen."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 wegen Bidirektionalität

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Nur die letzte Zeitschritt-Ausgabe verwenden
        return out


def plot_results(dates, actual, predicted):
    """Ergebnisse plotten."""
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Tatsächliche Werte', alpha=0.7)
    plt.plot(dates, predicted, label='Vorhersagen', alpha=0.7)
    plt.xlabel('Datum')
    plt.ylabel('SP500 Schlusskurs (normalisiert)')
    plt.title('Tatsächliche vs. Vorhergesagte Werte (normalisiert)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Dynamischen Pfad zur Datei definieren
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Sicherstellen, dass __file__ definiert ist
    file_path = os.path.join(script_dir, "../sp500_data/SP500_Index_Historical_Data.csv")

    # Daten laden und aufteilen
    data = load_data(file_path)
    train_data, test_data = split_data(data, '1994-01-01', '2015-12-31', '2016-01-01', '2024-12-31')

    # Features definieren und skalieren
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    train_scaled, test_scaled, scaler = scale_data(train_data, test_data, features)

    # Sequenzen erstellen
    seq_length = 20
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Tensor-Datensätze erstellen und DataLoader verwenden
    batch_size = 64

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Modell, Verlustfunktion und Optimierer definieren
    input_size = len(features)
    hidden_size = 128
    output_size = 1
    num_layers = 4
    dropout = 0.3

    model = StockLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Testen
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            predictions.extend(preds.flatten())
            actuals.extend(y_batch.numpy().flatten())

    # Verlust auf dem Testset berechnen
    test_loss = criterion(torch.tensor(predictions), torch.tensor(actuals)).item()
    print(f"Test Loss: {test_loss:.6f}")

    # Datum für den Plot auswählen
    test_dates = test_data['Date'].iloc[seq_length:].reset_index(drop=True)

    # Plotten der Ergebnisse
    plot_results(test_dates, actuals, predictions)


if __name__ == "__main__":
    main()