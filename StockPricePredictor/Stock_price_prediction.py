# Importing Packages

import pandas as p
import numpy as n
import yfinance as yf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pickle import NEXT_BUFFER
from math import ceil

# Reading Data

ticker = input('Enter the Ticker :')
exchange = '.ns'
join = ticker + exchange
print(join)
ticker_exchange = join.upper()
data = yf.download(join , period="5d", interval="1m")
print(data)

# Feature Extraction

data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility_5'] = data['Close'].rolling(window=5).std()

data.dropna(inplace=True)
print(data.tail())

# Seperating Traing and Testing Dataset

print(data.dtypes)
independent_variable = data[['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
dependent_variable = data['Close']
x_train , x_test , y_train , y_test = train_test_split(independent_variable , dependent_variable , test_size = 0.25)
print(len(x_train))
print(len(x_test))

# Initializing Models

models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# Choosing Best Model

epochs = 10
batch_size = 64
n_batches = ceil(len(x_train) / batch_size)

best_model = None
best_mae = float('inf')

for name, model in models.items():
    print(f"\nTraining {name}...\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        indices = n.arange(len(x_train))
        n.random.shuffle(indices)
        x_train = x_train.iloc[indices]
        y_train = y_train.iloc[indices]

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            x_batch = x_train.iloc[start:end]
            y_batch = y_train.iloc[start:end]

            model.fit(x_batch, y_batch)

        y_pred = model.predict(x_train)
        mae = mean_absolute_error(y_train, y_pred)
        print(f"{name} MAE after epoch {epoch + 1}: {mae}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            print(f"New best model: {name} with MAE {best_mae}")

print(f"\nBest model selected: {best_model}")

# Prediction for One Day

data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility_5'] = data['Close'].rolling(window=5).std()

data['Next_Day_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)
print(data.tail())

independent_variable = data[['Open', 'High', 'Low', 'Volume', 'Close' , 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
dependent_variable = data['Next_Day_Close']
x_train , x_test , y_train , y_test = train_test_split(independent_variable , dependent_variable , test_size = 0.25)
print(len(x_train))
print(len(x_test))

model = best_model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mae}')

m = model.predict(x_test)
print(m)
recent_data = data[independent_variable.columns].iloc[-1].values.reshape(1, -1)
next_day_close_pred = model.predict(recent_data)
print(f'Predicted Closing Price for One Day: {next_day_close_pred[0]}')

# Prediction For Five Days

data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility_5'] = data['Close'].rolling(window=5).std()

data['Next_Week_Close'] = data['Close'].shift(-5)
data.dropna(inplace=True)
print(data.tail())

independent_variable = data[['Open', 'High', 'Low', 'Volume', 'Close' , 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
dependent_variable = data['Next_Week_Close']
x_train , x_test , y_train , y_test = train_test_split(independent_variable , dependent_variable , test_size = 0.25)
print(len(x_train))
print(len(x_test))

model = best_model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mae}')

m = model.predict(x_test)
print(m)
recent_data = data[independent_variable.columns].iloc[-1].values.reshape(1, -1)
next_week_close_pred = model.predict(recent_data)
print(f'Predicted Closing Price for the Next Week: {next_week_close_pred}')

# Prediction For One Month

data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility_5'] = data['Close'].rolling(window=5).std()

data['Next_Month_Close'] = data['Close'].shift(-25)
data.dropna(inplace=True)
print(data.tail())

independent_variable = data[['Open', 'High', 'Low', 'Volume', 'Close' , 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
dependent_variable = data['Next_Month_Close']
x_train , x_test , y_train , y_test = train_test_split(independent_variable , dependent_variable , test_size = 0.25)
print(len(x_train))
print(len(x_test))

model = best_model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mae}')

m = model.predict(x_test)
print(m)
recent_data = data[independent_variable.columns].iloc[-1].values.reshape(1, -1)
next_month_close_pred = model.predict(recent_data)
print(f'Predicted Closing Price for the Next Month: {next_month_close_pred[0]}')

# Prediction For Six Month

data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volatility_5'] = data['Close'].rolling(window=5).std()

data['Next_Week_Close'] = data['Close'].shift(-110)
data.dropna(inplace=True)
print(data.tail())

independent_variable = data[['Open', 'High', 'Low', 'Volume', 'Close' , 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
dependent_variable = data['Next_Week_Close']
x_train , x_test , y_train , y_test = train_test_split(independent_variable , dependent_variable , test_size = 0.25)
print(len(x_train))
print(len(x_test))

model = best_model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mae}')

recent_data = data[independent_variable.columns].iloc[-1].values.reshape(1, -1)
next_week_close_pred = model.predict(recent_data)
print(f'Predicted Closing Price for the Six Month : {next_week_close_pred[0]}')