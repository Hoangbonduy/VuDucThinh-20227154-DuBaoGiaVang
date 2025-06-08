import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time

from models import (
    linear_model,
    svr_model,
    random_forest_model,
    gradient_boosting_model,
    xgboost_model,
    lstm_model,
    sarimax_model,
    prophet_model,
)

st.set_page_config(page_title="Dự báo giá vàng ngày tiếp theo với 8 mô hình", layout="wide")
st.title("📈 Dự báo giá vàng ngày tiếp theo với 8 mô hình")

st.markdown("""
Chọn file CSV dữ liệu vàng lịch sử để dự báo giá vàng ngày tiếp theo.

**Lưu ý:** File có thể lớn, vui lòng chờ trong quá trình xử lý.
""")

uploaded_file = st.file_uploader("🔄 Tải lên file CSV dữ liệu vàng", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Đang đọc và xử lý dữ liệu... Vui lòng chờ"):
        df = pd.read_csv(uploaded_file)
        time.sleep(0.5)
    st.success(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột")

    model_names = [
        "Linear Regression",
        "SVR",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LSTM",
        "SARIMAX",
        "Prophet",
    ]
    model_choice = st.selectbox("⚙️ Chọn mô hình dự báo", model_names)

    if st.button("🚀 Bắt đầu dự báo"):
        with st.spinner("Đang huấn luyện và dự báo..."):
            start_time = time.time()

            model_map = {
                "Linear Regression": linear_model.train_and_predict,
                "SVR": svr_model.train_and_predict,
                "Random Forest": random_forest_model.train_and_predict,
                "Gradient Boosting": gradient_boosting_model.train_and_predict,
                "XGBoost": xgboost_model.train_and_predict,
                "LSTM": lstm_model.train_and_predict,
                "SARIMAX": sarimax_model.train_and_predict,
                "Prophet": prophet_model.train_and_predict,
            }

            train_predict_func = model_map[model_choice]
            # Output: mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates, y_test, y_pred
            mae, mae_pct, rmse, rmse_pct, r2, r2_pct, mape, dates, y_test, y_pred = train_predict_func(df)
            elapsed = time.time() - start_time

        st.success(f"Dự báo hoàn thành trong {elapsed:.2f} giây!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE (%)", f"{mae_pct:.2f}%")
        col2.metric("RMSE (%)", f"{rmse_pct:.2f}%")
        col3.metric("MAPE (%)", f"{mape:.2f}%")
        col4.metric("R² (%)", f"{r2_pct:.2f}%")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, y_test, label="Giá thực tế", color="blue")
        ax.plot(dates, y_pred, label="Giá dự báo", color="orange")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá vàng")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Hiển thị duy nhất kết quả dự báo ngày tiếp theo (ngày cuối)
        next_day = pd.to_datetime(dates[-1])
        real_next = y_test[-1]
        pred_next = y_pred[-1]
        st.subheader("🎯 Dự báo giá vàng ngày tiếp theo:")
        st.markdown(f"""
- **Ngày:** {next_day.date()}
- **Giá thực tế:** {real_next:,.2f}
- **Giá dự báo:** :orange[{pred_next:,.2f}]
        """)

else:
    st.info("📄 Vui lòng tải lên file CSV dữ liệu vàng để bắt đầu dự báo.")

st.markdown("---")
st.markdown("💡 **Lưu ý:** Dữ liệu đầu vào cần đủ lớn (ít nhất 60 ngày) để mô hình dự báo hiệu quả.")
