import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date', 'Lần cuối': 'Price', 'Mở': 'Open',
        'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace('K', '').str.replace('M', '')
    df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce') * 1000
    return df

def add_features(df):
    df['pct_change'] = df['Price'].pct_change()
    for i in range(1, 5):
        df[f'lag_{i}'] = df['Price'].shift(i)
    df['diff'] = df['Price'] - df['lag_1']
    df['ma_5'] = df['Price'].rolling(window=5).mean()
    df['ma_10'] = df['Price'].rolling(window=10).mean()
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)

    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()

    df_train = add_features(df_train).dropna()
    df_test = add_features(df_test).dropna()

    features = [
        'Open', 'High', 'Low', 'Vol', 'pct_change',
        'lag_1', 'lag_2', 'lag_3', 'lag_4',
        'diff', 'ma_5', 'ma_10', 'Day', 'Month', 'Year'
    ]

    X_train = df_train[features]
    y_train = df_train['Price']
    X_test = df_test[features]
    y_test = df_test['Price']
    dates = df_test['Date']

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mean_price_test = np.mean(y_test)
    mae_pct = (mae / mean_price_test) * 100
    rmse_pct = (rmse / mean_price_test) * 100
    r2_pct = r2 * 100
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Trả về tất cả metrics cần thiết, không có mse
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates.values, y_test.values, y_pred

