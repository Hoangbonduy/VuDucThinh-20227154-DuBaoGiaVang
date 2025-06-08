import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date', 'Lần cuối': 'Price',
        'Mở': 'Open', 'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Price']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace(',', '')
    df['Vol'] = df['Vol'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    
    # Tính các chỉ số kỹ thuật mở rộng
    df['ma_5'] = df['Price'].rolling(window=5).mean()
    df['ma_10'] = df['Price'].rolling(window=10).mean()
    df['ma_20'] = df['Price'].rolling(window=20).mean()
    df['std_10'] = df['Price'].rolling(window=10).std()
    df['return'] = df['Price'].pct_change()
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 3])  # Chỉ số Price
    return np.array(X), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)
    
    features = ['Open', 'High', 'Low', 'Price', 'Vol', 'ma_5', 'ma_10', 'ma_20', 'std_10', 'return']
    data = df[features].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    SEQ_LEN = 30
    X, y = create_sequences(data_scaled, SEQ_LEN)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Xây dựng mô hình LSTM
    model = Sequential([
        LSTM(80, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
        Dropout(0.2),
        LSTM(40),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred = model.predict(X_test).flatten()
    
    def recover_price(pred):
        temp = np.zeros((len(pred), len(features)))
        temp[:, 3] = pred
        return scaler.inverse_transform(temp)[:, 3]
    
    y_test_real = recover_price(y_test)
    y_pred_real = recover_price(y_pred)
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    
    mean_price = y_test_real.mean()
    mae_pct = (mae / mean_price) * 100
    rmse_pct = (rmse / mean_price) * 100
    r2_pct = r2 * 100
    
    dates = df['Date'].values[-len(y_test_real):]
    
    # Trả về đúng thứ tự cần thiết, có MAPE, không có mse
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates, y_test_real, y_pred_real
