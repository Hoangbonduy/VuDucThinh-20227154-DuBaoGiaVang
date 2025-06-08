import pandas as pd
import numpy as np
from xgboost import XGBRegressor
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

    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['pct_change'] = df['Price'].pct_change()

    for i in range(1, 5):
        df[f'lag_{i}'] = df['Price'].shift(i)

    df['diff'] = df['Price'] - df['lag_1']
    df['ma_5'] = df['Price'].rolling(window=5).mean()
    df['ma_10'] = df['Price'].rolling(window=10).mean()

    df.dropna(inplace=True)

    df['ma_20'] = df['Price'].rolling(window=20).mean()
    df['std_10'] = df['Price'].rolling(window=10).std()
    df['weekday'] = df['Date'].dt.weekday
    df['quarter'] = df['Date'].dt.quarter

    for i in range(5, 11):
        df[f'lag_{i}'] = df['Price'].shift(i)

    df.dropna(inplace=True)
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)

    features = ['Open', 'High', 'Low', 'Vol', 'pct_change',
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10',
                'diff', 'ma_5', 'ma_10', 'ma_20', 'std_10', 'Day', 'Month', 'Year', 'weekday', 'quarter']

    split_date = df['Date'].quantile(0.8)
    train_idx = df['Date'] <= split_date
    test_idx = df['Date'] > split_date

    X_train, X_test = df.loc[train_idx, features], df.loc[test_idx, features]
    y_train, y_test = df.loc[train_idx, 'Price'], df.loc[test_idx, 'Price']
    dates_test = df.loc[test_idx, 'Date']

    model = XGBRegressor(
        n_estimators=180,
        learning_rate=0.015,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.2,
        reg_lambda=2.5,
        min_child_weight=4,
        random_state=42,
        tree_method='hist'
    )
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

    # Return đúng thứ tự chuẩn cho app.py
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates_test.values, y_test.values, y_pred
