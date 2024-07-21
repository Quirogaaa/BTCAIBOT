import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import logging
import warnings
import urllib3
import json
import os
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting main function")
    
    logging.info("Loading configuration from config.json")
    with open('config.json', 'r') as f:
        config = json.load(f)

    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

    logging.info("Setting up Kraken Futures API")
    kraken_futures = config['kraken_futures']

    exchange = ccxt.krakenfutures({
        'apiKey': kraken_futures['apiKey'],
        'secret': kraken_futures['secret'],
        'enableRateLimit': kraken_futures['enableRateLimit']
    })
    exchange.set_sandbox_mode(True)

    def fetch_crypto_data(symbol, timeframe='1h', limit=2000):
        logging.info(f"Fetching historical data for {symbol} from Kraken")
        filename = f'data_{symbol.replace("/", "_")}.csv'
        all_data = pd.DataFrame()

        if os.path.exists(filename):
            logging.info(f"Loading existing data from {filename}")
            all_data = pd.read_csv(filename, parse_dates=['timestamp'])

        if not all_data.empty:
            latest_timestamp = all_data['timestamp'].max()
            since = int(latest_timestamp.timestamp() * 1000)
            exchange.load_markets()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if not ohlcv:
                logging.info(f"No new data fetched for {symbol}.")
                return all_data

            new_data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            all_data = pd.concat([all_data, new_data]).drop_duplicates().reset_index(drop=True)
        else:
            exchange.load_markets()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                logging.error(f"No data fetched for {symbol}.")
                return pd.DataFrame()

            all_data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], unit='ms')

        all_data.to_csv(filename, index=False)
        return all_data

    def add_technical_indicators(df):
        logging.info("Adding technical indicators")
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = compute_rsi(df['close'])
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Volatility'] = df['close'].rolling(window=20).std()
        df = df.dropna()
        logging.info("Technical indicators added")
        return df

    def compute_rsi(series, period=14):
        logging.info("Computing RSI")
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_data_lstm(df, n_lags):
        logging.info("Preparing data for LSTM")
        features = ['close', 'SMA_20', 'SMA_50', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'Volatility']
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        X, y = [], []
        for i in range(n_lags, len(df)):
            X.append(df[features].iloc[i-n_lags:i].values)
            y.append(df['close'].iloc[i])
        return np.array(X), np.array(y)

    def create_lstm_model(input_shape=(60, 9), units=50, dropout_rate=0.2):
        logging.info("Creating LSTM model")
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_and_save_model(symbol, X, y, df):
        logging.info(f"Training and saving model for {symbol}")
        model_path = f'{symbol.replace("/", "_")}_model.h5'
        if os.path.exists(model_path):
            logging.info(f"Loading existing model from {model_path}")
            model = load_model(model_path)
            model.compile(optimizer='adam', loss='mean_squared_error') 
        else:
            model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        epochs = 50
        batch_size = 32

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            model.fit(X, y, epochs=1, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        model.save(model_path)
        logging.info(f"Model saved for {symbol}")

        df_full_data_with_indicators = df.copy()
        df_full_data_with_indicators.to_csv(f'data_{symbol.replace("/", "_")}.csv', index=False)
        logging.info(f"Data with indicators saved")
        
        return model

    def predict_next_move(df, model, n_lags):
        df = add_technical_indicators(df)
        if df.shape[0] < n_lags:
            raise ValueError("Not enough data to make predictions")
        X, _ = prepare_data_lstm(df, n_lags)
        if len(X) == 0:
            raise ValueError("No data available to make predictions")
        X = X[-1].reshape(1, X.shape[1], X.shape[2])
        prediction = model.predict(X)
        return 'Buy' if prediction > df['close'].iloc[-1] else 'Sell'

    win_prob = 0.55  
    win_loss_ratio = 1.5 

    def kelly_criterion(win_prob, win_loss_ratio):
        return win_prob - (1 - win_prob) / win_loss_ratio

    kelly_fraction = kelly_criterion(win_prob, win_loss_ratio)

    def run_bot_with_trailing_stop(symbol, model, exchange, interval=600, max_trades_per_day=3):
        logging.info(f"Running bot with trailing stop for {symbol}")
        trailing_stop_loss_pct = 0.05  
        trailing_take_profit_pct = 0.1  
        trailing_stop_loss_price = None
        trailing_take_profit_price = None
        trades_count = 0
        start_time = time.time()
        balance = 10000 
        initial_balance = balance

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 86400:
                trades_count = 0
                start_time = current_time

            if trades_count >= max_trades_per_day:
                logging.info("Max trades per day reached. Waiting for next day.")
                time.sleep(3600) 
                continue

            logging.info(f"Fetching latest data for {symbol}...")
            crypto_data = fetch_crypto_data(symbol, limit=2000)
            if crypto_data.empty or len(crypto_data) < 400:
                logging.error("No sufficient data loaded. Skipping this iteration.")
                time.sleep(interval)
                continue

            crypto_data = add_technical_indicators(crypto_data)
            if len(crypto_data) < 60:
                logging.error("Not enough data after adding indicators. Skipping this iteration.")
                time.sleep(interval)
                continue

            logging.info("Analyzing data to determine the next move...")
            try:
                next_move = predict_next_move(crypto_data, model, 60)
            except ValueError as e:
                logging.error(e)
                time.sleep(interval)
                continue

            logging.info(f'Next move for {symbol}: {next_move}')

            latest_price = crypto_data['price'].iloc[-1]
            logging.info(f"Latest price for {symbol}: {latest_price}")

            latest_rsi = crypto_data.iloc[-1]['RSI']
            logging.info(f"Current RSI: {latest_rsi}")
            try:
                position_size = balance * kelly_fraction / latest_price 
                if next_move == 'Buy' and (latest_rsi < 70 and latest_rsi > 30):
                    if trailing_stop_loss_price is None or latest_price < trailing_stop_loss_price * (1 - trailing_stop_loss_pct):
                        order = exchange.create_market_buy_order(symbol, position_size)
                        logging.info(f"Trade executed: {order}")
                        trailing_stop_loss_price = latest_price
                        trailing_take_profit_price = latest_price
                        trades_count += 1
                    else:
                        logging.warning(f"Skipping buy due to stop-loss limit: {latest_price} < {trailing_stop_loss_price * (1 - trailing_stop_loss_pct)}")
                elif next_move == 'Sell' and (latest_rsi < 70 and latest_rsi > 30):
                    if trailing_stop_loss_price is not None and (latest_price > trailing_take_profit_price * (1 + trailing_take_profit_pct) or latest_price < trailing_stop_loss_price * (1 - trailing_stop_loss_pct)):
                        order = exchange.create_market_sell_order(symbol, position_size)
                        logging.info(f"Trade executed: {order}")
                        trailing_stop_loss_price = None
                        trailing_take_profit_price = None
                        trades_count += 1
                    else:
                        if latest_price > trailing_take_profit_price:
                            trailing_take_profit_price = latest_price
                        if latest_price < trailing_stop_loss_price:
                            trailing_stop_loss_price = latest_price
                        logging.warning(f"Skipping sell due to take-profit or stop-loss limit: {latest_price} > {trailing_take_profit_price * (1 + trailing_take_profit_pct)} or {latest_price} < {trailing_stop_loss_price * (1 - trailing_stop_loss_pct)}")
                else:
                    logging.warning(f"Trade not executed due to safety limits: {next_move} {symbol}")
            except Exception as e:
                logging.error(f"Error executing trade: {e}")

            logging.info(f"Sleeping for {interval} seconds before next iteration...")
            time.sleep(interval)

    logging.info("Preparing data")
    crypto_symbols = ['BTC/USD:USD']

    for symbol in crypto_symbols:
        model_path = f'{symbol.replace("/", "_")}_model.h5'
        try:
            logging.info(f"Loading model for {symbol} from {model_path}")
            model = load_model(model_path)
            run_bot_with_trailing_stop(symbol, model, exchange, interval=600, max_trades_per_day=3) 
        except Exception as e:
            logging.error(f"Error loading model for {symbol} from {model_path}: {e}")

if __name__ == "__main__":
    main()
