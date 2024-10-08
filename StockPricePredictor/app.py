from flask import Flask, request, render_template, jsonify
import pandas as p
import numpy as n
import yfinance as yf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from math import ceil

app = Flask(__name__)

# Initialize models globally so we can reuse them
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# Function for stock prediction
def predict_stock(ticker, period):
    # Download stock data
    data = yf.download(ticker, period="6mo", interval="1d")
    
    if data.empty:
        return "No data available for the provided ticker and period."

    # Feature Extraction
    data['Return'] = (data['Close'] - data['Open']) / data['Open']
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility_5'] = data['Close'].rolling(window=5).std()
    data.dropna(inplace=True)

    independent_variable = data[['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'Volatility_5', 'Return']]
    
    if period == "1d":
        data['Next_Day_Close'] = data['Close'].shift(-1)
        dependent_variable = data['Next_Day_Close']
    elif period == "5d":
        data['Next_Week_Close'] = data['Close'].shift(-5)
        dependent_variable = data['Next_Week_Close']
    elif period == "1mo":
        data['Next_Month_Close'] = data['Close'].shift(-25)
        dependent_variable = data['Next_Month_Close']
    elif period == "6mo":
        data['Next_6Month_Close'] = data['Close'].shift(-110)
        dependent_variable = data['Next_6Month_Close']
    else:
        return "Invalid period selected."

    data.dropna(inplace=True)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=0.25)

    # Choose the best model (we assume this is done previously)
    best_model = models["XGBoost"]
    best_model.fit(x_train, y_train)

    # Prediction
    recent_data = independent_variable.iloc[-1].values.reshape(1, -1)
    prediction = best_model.predict(recent_data)

    return prediction[0]

@app.route('/')
def index():
    return render_template('home.html')  # A simple HTML page to get input from the user

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']  # Get the ticker from the user input
    period = request.form['period']  # Get the period (e.g., 1d, 5d, 1mo, 6mo)
    
    # Append the exchange name if needed
    ticker_exchange = ticker + '.NS'
    
    # Make prediction
    prediction = predict_stock(ticker_exchange, period)
    
    return jsonify({'ticker': ticker_exchange, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
