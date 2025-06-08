import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

def load_and_preprocess(df):
    warnings.filterwarnings("ignore")

    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date', 'Lần cuối': 'Price', 'Mở': 'Open',
        'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace(',', '')
    df['Vol'] = df['Vol'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    df = df.sort_values('Date').reset_index(drop=True)

    # CHỈ GIỮ 2 FEATURE ĐƠN GIẢN
    df['SMA_7'] = df['Price'].rolling(7).mean()
    df['Momentum_3'] = df['Price'] - df['Price'].shift(3)
    df = df.dropna().reset_index(drop=True)
    df.set_index('Date', inplace=True)

    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)

    y = df['Price']
    # CHỈ GIỮ 2 FEATURE ĐƠN GIẢN
    exog = df[['SMA_7', 'Momentum_3']]

    split = int(0.8 * len(df))
    y_train, y_test = y[:split], y[split:]
    X_train, X_test = exog[:split], exog[split:]

    # CHỈ DÙNG 1 BỘ THAM SỐ CƠ BẢN, BỎ SEASONAL
    order = (1, 1, 0)
    seasonal_order = (0, 0, 0, 0)

    best_model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast = best_model.predict(start=len(y_train), end=len(y)-1, exog=X_test)

    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    r2 = r2_score(y_test, forecast)
    mape = mean_absolute_percentage_error(y_test, forecast)
    mean_price = y_test.mean()
    mae_pct = (mae / mean_price) * 100
    rmse_pct = (rmse / mean_price) * 100
    r2_pct = r2 * 100

    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, y_test.index.values, y_test.values, forecast.values
