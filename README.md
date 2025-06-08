# Dự báo giá vàng ngày tiếp theo với nhiều mô hình

Ứng dụng Streamlit sử dụng 7 mô hình Machine Learning và Deep Learning để dự báo giá vàng ngày tiếp theo dựa trên dữ liệu lịch sử.

---

## Các mô hình sử dụng

- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LSTM (Long Short-Term Memory)
- SARIMAX (Seasonal ARIMA)

---

## Cài đặt

# Gold Price Forecast App

## Cách chạy ứng dụng

### 1. Tạo môi trường ảo
```bash

# Windows (Python 3.10+ đã cài và thêm vào PATH):
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate



```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng
```bash
streamlit run app.py
```

> ⚠️ Yêu cầu: Python 3.10 trở lên
> Link: https://dubaogiavang.streamlit.app
