import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # Sử dụng Ridge Regression để regularization
from sklearn.ensemble import RandomForestRegressor  # Sử dụng Random Forest để giảm overfitting
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel

# Đọc và xử lý dữ liệu
def load_and_preprocess(df):
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Ngày': 'Date', 'Lần cuối': 'Price',
        'Mở': 'Open', 'Cao': 'High', 'Thấp': 'Low', 'KL': 'Vol'
    }, inplace=True)
    
    # Chuyển đổi cột 'Date' và xử lý các giá trị NaN
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')

    # Xử lý giá trị thiếu NaN bằng cách thay thế với giá trị trung bình
    for col in ['Open', 'High', 'Low', 'Price']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        df[col].fillna(df[col].mean(), inplace=True)  # Thay thế NaN bằng giá trị trung bình cột
    
    # Xử lý cột 'Vol' (Khối lượng giao dịch)
    df['Vol'] = df['Vol'].astype(str).str.replace(',', '')
    df['Vol'] = df['Vol'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    df['Vol'].fillna(df['Vol'].mean(), inplace=True)  # Thay thế NaN bằng giá trị trung bình cột

    return df

# Tạo các chuỗi dữ liệu với cửa sổ trượt
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i].flatten())
        y.append(data[i, 3])  # Giá 'Price' tại chỉ số 3
    return np.array(X), np.array(y)

# Tính MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Huấn luyện mô hình và dự đoán
def train_and_predict(df):
    df = load_and_preprocess(df)
    features = ['Open', 'High', 'Low', 'Price', 'Vol']
    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, 30)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Chọn mô hình Random Forest hoặc Ridge để giảm overfitting
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # Cross-validation để đánh giá mô hình
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-validation MAE: {-cv_scores.mean()}")  # In kết quả MAE từ cross-validation

    # Feature Selection - Chọn đặc trưng quan trọng nhất
    selector = SelectFromModel(model, threshold="mean", max_features=5)  # Chọn 5 đặc trưng quan trọng nhất
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Huấn luyện lại mô hình với các đặc trưng đã chọn
    model.fit(X_train_selected, y_train)

    # Dự đoán
    y_pred = model.predict(X_test_selected)

    # Khôi phục giá trị dự đoán từ chuẩn hóa
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

    # Return bổ sung mape, đúng chuẩn app.py
    return mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates, y_test_real, y_pred_real
