import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date',
        'Lần cuối': 'Price',
        'Mở': 'Open',
        'Cao': 'High',
        'Thấp': 'Low',
        'KL': 'Vol'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace('K', '').str.replace('M', '')
    df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce') * 1000
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)

    N_LAG = 14
    for lag in range(1, N_LAG + 1):
        df[f'lag_{lag}'] = df['Price'].shift(lag)
    df['diff'] = df['Price'] - df['Price'].shift(1)
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['SMA_7'] = df['Price'].rolling(7).mean()
    df['EMA_7'] = df['Price'].ewm(span=7, adjust=False).mean()
    df['Volatility_7'] = df['Price'].rolling(7).std()
    df['Momentum_5'] = df['Price'] - df['Price'].shift(5)

    df = df.dropna().reset_index(drop=True)

    features = ['Open', 'High', 'Low', 'Vol'] + [f'lag_{i}' for i in range(1, N_LAG + 1)] + ['diff', 'Day', 'Month', 'Year', 'SMA_7', 'EMA_7', 'Volatility_7', 'Momentum_5']

    split_idx = int(0.8 * len(df))
    X_train, X_test = df[features].iloc[:split_idx], df[features].iloc[split_idx:]
    y_train, y_test = df['Price'].iloc[:split_idx], df['Price'].iloc[split_idx:]
    dates_test = df['Date'].iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mean_price_test = y_test.mean()
    mae_pct = (mae / mean_price_test) * 100
    rmse_pct = (rmse / mean_price_test) * 100
    r2_pct = r2 * 100

    # Trả về đúng thứ tự đã thống nhất
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates_test.values, y_test.values, y_pred
