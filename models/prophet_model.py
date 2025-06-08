import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={'Ngày': 'Date', 'Lần cuối': 'Price', 'Mở': 'Open',
                       'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace(',', '')
    df['Vol'] = df['Vol'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    df = df.sort_values('Date').reset_index(drop=True)

    df['SMA_7'] = df['Price'].rolling(7).mean()
    df['Momentum_3'] = df['Price'] - df['Price'].shift(3)
    df['log_Vol'] = np.log1p(df['Vol'])
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df = df.dropna().reset_index(drop=True)
    
    prophet_df = df[['Date', 'Price', 'SMA_7', 'Momentum_3', 'log_Vol', 'is_month_end']].rename(
        columns={'Date': 'ds', 'Price': 'y'}
    )
    return prophet_df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    prophet_df = load_and_preprocess(df)
    split = int(0.8 * len(prophet_df))
    train_df = prophet_df.iloc[:split]
    test_df = prophet_df.iloc[split:]

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.add_regressor('SMA_7')
    model.add_regressor('Momentum_3')
    model.add_regressor('log_Vol')
    model.add_regressor('is_month_end')

    model.fit(train_df)

    future = test_df[['ds', 'SMA_7', 'Momentum_3', 'log_Vol', 'is_month_end']]
    forecast = model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mean_price = y_true.mean()
    mae_pct = (mae / mean_price) * 100
    rmse_pct = (rmse / mean_price) * 100
    r2_pct = r2 * 100

    dates = test_df['ds'].values

    # Trả về metrics giống các model khác
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates, y_true, y_pred
