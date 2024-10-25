import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Load historical data from yfinance
def load_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Adj Close'] = data['Adj Close'].fillna(method='ffill')
    return data[['Adj Close']]


# Prepare the dataset for training
def prepare_data(data, sequence_length=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    x, y = [], []
    for i in range(len(scaled_data) - sequence_length - 1):
        x.append(scaled_data[i:(i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])
    return np.array(x), np.array(y), scaler


# LSTM Model
class LSTMModel(nn.Module):
    def _init_(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self)._init_()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 50).to(device)
        c0 = torch.zeros(1, x.size(0), 50).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Training function
def train_model(model, criterion, optimizer, x_train, y_train, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


# Make predictions and evaluate
def predict(model, x_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
    return predictions, y_test


# Main function
def main():
    ticker = 'AAPL'
    data = load_data(ticker)
    x, y, scaler = prepare_data(data.values)

    # Train-test split (5 days ago)
    x_train, x_test = torch.FloatTensor(x[:-5]).unsqueeze(-1).to(device), torch.FloatTensor(x[-5:]).unsqueeze(-1).to(
        device)
    y_train, y_test = torch.FloatTensor(y[:-5]).unsqueeze(-1).to(device), torch.FloatTensor(y[-5:]).unsqueeze(-1).to(
        device)

    # Model, loss function, optimizer
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    model = train_model(model, criterion, optimizer, x_train, y_train)
    predictions, actual = predict(model, x_test, y_test, scaler)

    # Print actual and predicted values
    for i in range(len(predictions)):
        print(f'Day {i + 1}: Actual Price = {actual[i][0]:.2f}, Predicted Price = {predictions[i][0]:.2f}')

    # Calculate accuracy as percentage
    accuracy = 100 - np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f'Prediction Accuracy: {accuracy:.2f}%')

    # Plot the results
    plt.plot(predictions, label='Predicted')
    plt.plot(actual, label='Actual')
    plt.legend()
    plt.show()


if _name_ == "_main_":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()