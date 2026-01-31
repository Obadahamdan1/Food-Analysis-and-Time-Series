import kagglehub
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#Load & Preprocess Data
path = kagglehub.dataset_download("ihelon/coffee-sales")
df = pd.read_csv(os.path.join(path, "index_1.csv"))

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')
data = df['money'].values.reshape(-1, 1)

#Smooth the data
data = pd.Series(data.flatten()).rolling(window=7, center=True, min_periods=1).mean().values.reshape(-1, 1)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

sequence_length = 10

#Custom Dataset
class CoffeeDataset(Dataset):
    def __init__(self, data, seq_length):
        self.x, self.y = [], []
        for i in range(len(data) - seq_length):
            self.x.append(data[i:i + seq_length])
            self.y.append(data[i + seq_length])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_size = int(len(data_normalized) * 0.8)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size - sequence_length:]

train_dataset = CoffeeDataset(train_data, sequence_length)
test_dataset = CoffeeDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

#BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  #*2 for BiLSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_output = lstm_out[:, -1, :] 
        output = self.fc(last_step_output)
        return output

#Training
model = BiLSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

#Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for x_test, y_test in test_loader:
        pred = model(x_test)
        predictions.append(pred.item())
        actuals.append(y_test.item())

predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals_inv = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

mse = mean_squared_error(actuals_inv, predictions_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals_inv, predictions_inv)
r2 = r2_score(actuals_inv, predictions_inv)

print(f"\nBiLSTM Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

#Plot
plt.figure(figsize=(12, 6))
plt.plot(actuals_inv, label='Actual (Smoothed)', color='blue')
plt.plot(predictions_inv, label='Predicted', color='orange')
plt.xlabel("Time Step")
plt.ylabel("Sales Amount")
plt.title("BiLSTM Coffee Sales Forecasting")
plt.legend()
plt.grid(True)
plt.show()