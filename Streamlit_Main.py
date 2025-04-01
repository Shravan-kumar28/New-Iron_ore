#Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine, text
import pymysql



# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load saved scaler and model
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model = load_model("GRU_model1.h5", compile=False)


# Streamlit UI
st.set_page_config(page_title="Iron Ore Price Prediction", page_icon="📊", layout="wide")
st.title("🔮 Iron Ore Price Prediction with GRU Model")
st.markdown("---")

# Sidebar - Database Connection
st.sidebar.header("🔌 Database Connection")
db_host = st.sidebar.text_input("Enter Database Host", value="localhost")
db_user = st.sidebar.text_input("Enter Database Username")
db_password = st.sidebar.text_input("Enter Database Password", type="password")
db_name = st.sidebar.text_input("Enter Database Name")

# Sidebar button for database connection
if st.sidebar.button("🔗 Connect to Database"):
    try:
        # Securely encode database credentials to avoid issues with special characters
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        with engine.begin():
            st.sidebar.success("✅ Successfully connected to the database!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")

st.sidebar.markdown("---")

# Sidebar - File Uploader
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        conn = engine.connect()
        
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.subheader("📜 Raw Data Preview")
        st.dataframe(data.head(10))
        
        # Store data in the database
        data.to_sql('iron_ore_prices', con=engine, if_exists='replace', index=False, chunksize=1000)
        st.success("📊 Data successfully uploaded to database!")
        
        # Fetch data from database
        sql = 'SELECT * FROM iron_ore_prices'
        df = pd.read_sql_query(text(sql), con=engine.connect())
        df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y')
        df = df.sort_values("Date").set_index("Date")
        
        # Rolling Mean Trend Analysis
        st.subheader("📈 365-Days Rolling Mean Analysis")
        rolling_mean_series = df["Price"].rolling(window=365).mean()
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        ax_trend.plot(df.index, df["Price"], label="Iron Ore Price", color='steelblue')
        ax_trend.plot(df.index, rolling_mean_series, label="365-days Rolling Mean", color="red")
        ax_trend.set_xlabel("Date")
        ax_trend.set_ylabel("Price")
        ax_trend.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_trend)
        
        # Train-test split
        train_data = df[df.index < '2022-01-01']
        test_data = df[df.index >= '2022-01-01']
        
        # Scale data
        scaled_test_data = scaler.transform(test_data[['Price']].values)
        
        # Create dataset function
        def create_dataset(dataset, time_step=365):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(y)
        
        # Define time_step (365 for one year of data)
        time_step = 365
        X_test, y_test = create_dataset(scaled_test_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Predictions
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1, 1)  # Reshape to (samples, 1)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        test_dates = test_data.index[time_step:time_step + len(predictions)]
        
        # Store predictions in database
        results_df = pd.DataFrame({"Date": test_dates, "Actual Price": y_test_actual.flatten(), "Predicted Price": predictions.flatten()})
        
        # Connect to the database and execute a SQL command to drop the table if it exists
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS GRU_forecast_predictions"))
            connection.commit()
         
        # Save the DataFrame to a SQL table named 'GRU_forecast_predictions'    
        results_df.to_sql('GRU_forecast_predictions', con=engine, if_exists='replace', index=False, chunksize=1000)
        
        # Display results
        st.subheader("📊 Prediction Results")
        st.dataframe(results_df.head(10))
        
        # Plot predictions
        st.subheader("📉 Prediction Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_dates, y_test_actual, label="Actual Price", color="green")
        ax.plot(test_dates, predictions, label="Predicted Price", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Forecast future prices
        forecast_days = st.sidebar.slider("🔮 Forecast Length (Days)", 1, 365, 30)
        forecast_input = scaled_test_data[-time_step:].reshape(1, time_step, 1)
        forecast_predictions = []
        
        # Loop through the number of forecast days to generate predictions iteratively
        for _ in range(forecast_days):
            pred = model.predict(forecast_input)[0, 0]
            forecast_predictions.append(pred)
            forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)
        
        # Convert predictions back to actual values using the inverse scaler transformation
        forecast_predictions_actual = scaler.inverse_transform(np.array(forecast_predictions).reshape(-1, 1)).flatten()
        forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_days+1, freq='D')[1:]
        
        
        # Compute standard deviation of predictions to estimate confidence intervals
        forecast_std = np.std(forecast_predictions_actual) * 0.1  # 10% of standard deviation
        # Compute lower and upper bounds for uncertainty estimation
        lower_bound = forecast_predictions_actual - forecast_std
        upper_bound = forecast_predictions_actual + forecast_std

        # Create DataFrame with Confidence Intervals
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Price": forecast_predictions_actual,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound
        })
        
        # Connect to the database and execute a SQL command to drop the table if it exists
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS GRU_forecast_future"))
            connection.commit()
        
        # Save the DataFrame to a SQL table named 'GRU_forecast_future'
        forecast_df.to_sql('GRU_forecast_future', con=engine, if_exists='replace', index=False, chunksize=1000)
        
        # Display Forecast Results
        st.subheader("📊 Future Forecast")
        st.dataframe(forecast_df.head(10))
        
        # Plot Forecast
        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
        ax_forecast.plot(forecast_dates, forecast_predictions_actual, label="Forecasted Price", color="blue", linestyle='dashed')
        ax_forecast.set_xlabel("Date")
        ax_forecast.set_ylabel("Price")
        ax_forecast.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_forecast)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
