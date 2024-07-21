import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_crypto_data(symbol, exchange, timeframe='1h', limit=1200):
    logging.info(f"Fetching historical data for {symbol}")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['close']
    return df

def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['price'])
    df['EMA_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Volatility'] = df['price'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data_lstm(df, n_lags):
    features = ['price', 'SMA_20', 'SMA_50', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'Volatility']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(n_lags, len(df)):
        X.append(df[features].iloc[i-n_lags:i].values)
        y.append(df['price'].iloc[i])
    return np.array(X), np.array(y)

def predict_next_move(df, model, n_lags):
    df = add_technical_indicators(df)
    X, _ = prepare_data_lstm(df, n_lags)
    if len(X) == 0:
        raise ValueError("Not enough data to make predictions")
    X = X[-1].reshape(1, X.shape[1], X.shape[2])
    prediction = model.predict(X)
    return 'Buy' if prediction > df['price'].iloc[-1] else 'Sell'

def backtest_strategy_with_compounding_and_trailing(symbol, exchange, model, n_lags):
    crypto_data = fetch_crypto_data(symbol, exchange, limit=1200)
    crypto_data = add_technical_indicators(crypto_data)
    
    initial_balance = 10000
    balance = initial_balance
    position = 0
    balance_history = []
    trailing_stop_loss_pct = 0.05  
    trailing_take_profit_pct = 0.1  
    trailing_stop_loss_price = None
    trailing_take_profit_price = None
    trades_count = 0
    trade_log = []

    for i in tqdm(range(n_lags, len(crypto_data)), desc=f"Backtesting {symbol}"):
        df_slice = crypto_data.iloc[:i]
        try:
            action = predict_next_move(df_slice, model, n_lags)
        except ValueError as e:
            continue
        
        latest_price = df_slice['price'].iloc[-1]
        latest_date = df_slice.index[-1]
        
        if action == 'Buy' and position == 0:
            position = balance / latest_price
            balance = 0
            trailing_stop_loss_price = latest_price
            trailing_take_profit_price = latest_price
            trades_count += 1
            trade_log.append((latest_date, 'Buy', latest_price, balance + position * latest_price))
        elif action == 'Sell' and position > 0:
            balance = position * latest_price
            position = 0
            trailing_stop_loss_price = None
            trailing_take_profit_price = None
            trades_count += 1
            trade_log.append((latest_date, 'Sell', latest_price, balance))
        elif position > 0:
            if latest_price > trailing_take_profit_price:
                trailing_take_profit_price = latest_price
            if latest_price < trailing_stop_loss_price:
                trailing_stop_loss_price = latest_price
            
            if latest_price <= trailing_stop_loss_price * (1 - trailing_stop_loss_pct):
                balance = position * latest_price
                position = 0
                trailing_stop_loss_price = None
                trailing_take_profit_price = None
                trades_count += 1
                trade_log.append((latest_date, 'Sell', latest_price, balance))
            elif latest_price >= trailing_take_profit_price * (1 + trailing_take_profit_pct):
                balance = position * latest_price
                position = 0
                trailing_stop_loss_price = None
                trailing_take_profit_price = None
                trades_count += 1
                trade_log.append((latest_date, 'Sell', latest_price, balance))
        
        total_balance = balance + (position * latest_price)
        balance_history.append(total_balance)
    
    profit_percentage = ((total_balance - initial_balance) / initial_balance) * 100
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(balance_history)), balance_history, label='Balance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.title(f'Backtesting Results for {symbol}')
    plt.legend()
    plt.show()
    
    trade_log_df = pd.DataFrame(trade_log, columns=['Date', 'Action', 'Price', 'Balance'])
    print(trade_log_df)
    
    return total_balance, profit_percentage

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    kraken_futures = config['kraken_futures']
    
    exchange = ccxt.krakenfutures({
        'apiKey': kraken_futures['apiKey'],
        'secret': kraken_futures['secret'],
        'enableRateLimit': kraken_futures['enableRateLimit']
    })
    exchange.set_sandbox_mode(True)
    
    n_lags = 60
    symbol = 'BTC/USD:USD'
    model_path = f'{symbol.replace("/", "_")}_model.h5'
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        final_balance, profit_percentage = backtest_strategy_with_compounding_and_trailing(symbol, exchange, model, n_lags)
        
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Profit Percentage: {profit_percentage:.2f}%")
    else:
        print(f"Model for {symbol} not found at {model_path}")

if __name__ == "__main__":
    main()
