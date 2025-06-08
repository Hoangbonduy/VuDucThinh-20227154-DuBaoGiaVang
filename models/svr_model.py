import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Đặt seed để bảo đảm reproducibility (nếu muốn)
np.random.seed(42)

def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date', 'Lần cuối': 'Price',
        'Mở': 'Open', 'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Vol'] = df['Vol'].astype(str).str.replace(',', '')
    df['Vol'] = df['Vol'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))

    N_LAG = 5
    for i in range(1, N_LAG + 1):
        df[f'lag_{i}'] = df['Price'].shift(i)
    df['SMA_5'] = df['Price'].rolling(5).mean()
    df['Momentum_5'] = df['Price'] - df['Price'].shift(5)
    df['diff_1'] = df['Price'] - df['lag_1']
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday

    df = df.dropna().reset_index(drop=True)
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(df):
    df = load_and_preprocess(df)

    features = [f'lag_{i}' for i in range(1, 6)] + ['SMA_5', 'Momentum_5', 'diff_1', 'Day', 'Month', 'Weekday']

    split = int(0.8 * len(df))
    df_train, df_test = df.iloc[:split], df.iloc[split:]
    dates_test = df_test['Date']

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(df_train[features])
    X_test = scaler_X.transform(df_test[features])

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(df_train[['Price']]).ravel()
    y_test = scaler_y.transform(df_test[['Price']]).ravel()

    param_grid = {
        'C': [2, 3],
        'epsilon': [0.05],
        'gamma': [0.01, 0.02]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=tscv, verbose=0)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # Huấn luyện mô hình với y_train (không thêm noise!)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)

    mean_price = y_test_real.mean()
    mae_pct = (mae / mean_price) * 100
    rmse_pct = (rmse / mean_price) * 100
    r2_pct = r2 * 100

    # Return đúng thứ tự
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates_test.values, y_test_real, y_pred_real
