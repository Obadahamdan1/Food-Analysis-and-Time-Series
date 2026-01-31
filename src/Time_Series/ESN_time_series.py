#import libraries 
import kagglehub #for dataset
import os 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler #scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error, r2_score 
import matplotlib.pyplot as plt 
from pyESN import ESN #The Echo State Network model

#Download latest version from kaggle
path = kagglehub.dataset_download("ihelon/coffee-sales")

print("Path to dataset files:", path)

#Function to create lagged features, Lagged features = Past values of your time series used as inputs to predict future values
def create_lagged_features(data, window=14):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

#Load and preprocess data
path = kagglehub.dataset_download("ihelon/coffee-sales")
df = pd.read_csv(os.path.join(path, "index_1.csv"))
df['datetime'] = pd.to_datetime(df['datetime']) #convert to datetime format 
daily_sales = df.groupby(df['datetime'].dt.date)['money'].sum().reset_index() #Aggregates sales per day by summing the money column
daily_sales.columns = ['date', 'total_sales']
daily_sales = daily_sales.sort_values("date")

daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek

# Fill missing dates with 0 sales if needed
#full_range = pd.date_range(start=daily_sales['date'].min(), end=daily_sales['date'].max())
#daily_sales = daily_sales.set_index('date').reindex(full_range).fillna(0).rename_axis('date').reset_index()

#Scale data 
#MinMaxScaler to scale values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(daily_sales[["total_sales"]])

#smoothing
daily_sales['smoothed_sales'] = daily_sales['total_sales'].rolling(window=7).mean().fillna(method='bfill')
scaled_sales = scaler.fit_transform(daily_sales[["smoothed_sales"]])

#Create lagged features
#sliding window of 30 days to predict the next day’s sales
window = 30  
X, y = create_lagged_features(scaled_sales.flatten(), window)
#split data into train/test
train_size = int(len(X) * 0.8)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

#Initialize ESN
esn = ESN(
    n_inputs=window,
    n_outputs=1,
    random_state=42,
    n_reservoir=600, #600 neurons in the reservoir
    sparsity=0.2,    
    noise=0.001,     
    spectral_radius=0.8,  #Controls memory capacity
    input_scaling= 0.7,  #Reduce input influence
)

#Train with warm-up
warmup = 50
esn.fit(train_X[:-warmup], train_y[:-warmup])
pred_train = esn.predict(train_X[-warmup:])
print(f"Warm-up MSE: {mean_squared_error(train_y[-warmup:], pred_train):.4f}")

#Predict test set
pred_esn = esn.predict(test_X)
pred_esn = np.clip(pred_esn, 0, 1)  #Ensure valid range 0 to 1
pred_esn_inv = scaler.inverse_transform(pred_esn.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(test_y.reshape(-1, 1))

#Evaluate
mse = mean_squared_error(y_test_inv, pred_esn_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, pred_esn_inv)
r2 = r2_score(y_test_inv, pred_esn_inv)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MSE: {mse:.4f}")

#Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", linewidth=2)
plt.plot(pred_esn_inv, label="ESN Prediction", linestyle="--")
plt.title("Coffee Sales Forecast (ESN with Lagged Features)")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()