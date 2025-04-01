# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:43:12 2025

@author: shrav
"""

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random
import joblib
from sqlalchemy import create_engine, text
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# Load dataset from CSV file
data = pd.read_csv(r"E:/360DigiTmg/ML Project/Python/Data_set_Iron/Data_set_Iron.csv")

# Database connection setup
user = 'root'
password = 'password'
db = 'Iron_ore'

# Creating SQL engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

# Storing dataset in MySQL table
data.to_sql('iron_ore_prices', con=engine, if_exists='replace', index=False, chunksize=1000)

# Fetching data from database
sql = 'select * from iron_ore_prices'
df = pd.read_sql_query(text(sql), con=engine.connect())

# Displaying first and last records
df.head()
df.tail()

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display dataset information
df.info()

# Display dataset shape
df.shape

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y')

# Removing percentage sign and converting 'Change %' column to float
df['Change %'] = df['Change %'].str.replace("%", "").astype(float)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dropping irrelevant column 'Vol.'
df.drop(columns=["Vol."], inplace=True)

# Selecting numeric columns for analysis
Numeric_data = df.select_dtypes(include=["number"])

# First Moment Business Decision: Central Tendency (Mean, Median, Mode)
mean_values = Numeric_data.mean()
median_values = Numeric_data.median()
mode_values = Numeric_data.mode().iloc[0]  # Mode can have multiple values, so selecting first occurrence

CentralTendency_df = pd.DataFrame({
    "Mean": mean_values,
    "Median": median_values,
    "Mode": mode_values
})

#Displaying the results
print(CentralTendency_df)

# Second Moment Business Decision: Dispersion (Variance, Std Dev, Range)
variance_values = Numeric_data.var()
std_dev_values = Numeric_data.std()
range_values = Numeric_data.max() - Numeric_data.min()

Dispersion_df = pd.DataFrame({
    "Variance": variance_values,
    "Standard Deviation": std_dev_values,
    "Range": range_values
})

#Displaying the results
print(Dispersion_df)

# Third Moment Business Decision: Skewness
skewness_values = Numeric_data.skew()
print("Skewness of each numeric column:")
print(skewness_values) #Displaying the results

# Fourth Moment Business Decision: Kurtosis
kurtosis_values = Numeric_data.kurt()
print("Kurtosis of each numerical column:")
print(kurtosis_values) #Displaying the results





""" Business Insights:-
1.Mode(126.01) is higher than both mean and median,that means 126.01 was a frequently occurring price.
2.Change %,Most days had little or no price change, confirming a stable market.
3.Mean is slightly positive, the overall trend in prices might be slightly upward over time.
4.price(Variance):- High price volatility increase financial risk.
5.right-skewed distribution,it means there are some extreme high prices, but most values are concentrated on the lower end.
6.frequent small declines in price,Buyers should purchase during dips rather than peaks.
7.kurtosis of 1.06 suggests that iron ore prices were relatively stable over time.
"""

# Calculate rolling variance with a 30-day window
rolling_variance = df["Price"].rolling(window=365).var()

# Plotting rolling variance
plt.figure(figsize=(10,6))
plt.plot(df['Date'], rolling_variance, label="365-Days Rolling Variance", color="purple")
plt.title("Rolling Variance of Iron Ore Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Variance")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# Rolling Mean (Trend Analysis) for the target variable 'Price'
rolling_mean_series = df["Price"].rolling(window=365).mean()

plt.figure(figsize=(10,6))
plt.plot(df["Date"], df["Price"], label="Iron Ore Price")
plt.plot(df["Date"], rolling_mean_series, label="365-days Rolling Mean", color="red")
plt.title("365-Days Rolling Mean of Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Univariate Analysis
# Histogram
plt.figure(figsize=(10,6))
df.hist(figsize=(10,6), bins=30, edgecolor="black")
plt.title("Histograms of Numeric Columns", fontsize=14)
plt.show()

# Box-Plot (Detect Outliers)
df.plot(kind='box', subplots=True, sharey=False, figsize=(18,10))
plt.title("Outliers Detection")
plt.subplots_adjust(wspace=0.75)
plt.show()

# Q-Q Plot (Check Normality of Price Distribution)
plt.figure(figsize=(8,6))
stats.probplot(df["Price"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Price")
plt.show()

# Bivariate Analysis (Correlation Analysis)
correlation_matrix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Multivariate Analysis (Pair Plot)
g=sns.pairplot(df, diag_kind="kde")
g.fig.suptitle("Pair Plot of Features", y=1.02, fontsize=14)
plt.show()

#Mean,median, and mode for Price, Open, High, and Low are almost identical.
#High correlation between these features
#Multicollinearity can distort model predictions and increase variance.
#Extra columns increase memory usage and processing time.
# Dropping highly correlated redundant columns
df.drop(columns=['Open', 'High', 'Low'], inplace=True)


"""
Dickey-Fuller Test (ADF Test)
--------------------------------
Null Hypothesis (H₀): The series is non-stationary (has a unit root).
Alternative Hypothesis (H₁): The series is stationary (does NOT have a unit root).
Decision Rule:
- If p-value < 0.05 → Reject H₀ → The series is STATIONARY.
- If p-value ≥ 0.05 → Fail to reject H₀ → The series is NON-STATIONARY.
"""

adf_test = adfuller(df["Price"])
print(f"ADF Statistic: {adf_test[0]}")
print(f"P-Value: {adf_test[1]}")

if adf_test[1] < 0.05:
    print("The time series is STATIONARY.")
else:
    print("The time series is NON-STATIONARY.")
    
    
# Sorting data by Date and setting it as index
df = df.sort_values("Date")  

# Perform Seasonal Decomposition of Time Series
# 'additive' model assumes the data is a sum of trend, seasonality, and residual components
# 'period=365' assumes a yearly seasonal pattern (useful for daily data spanning multiple years)
decomposition = seasonal_decompose(df["Price"], model="additive", period=365) 

# Set figure size for better visualization
plt.figure(figsize=(12, 8))

# Plot Original Data (Raw Time Series)
plt.subplot(4, 1, 1)  # Create the first subplot (out of 4 rows)
plt.plot(df["Price"], label="Original Data")  # Plot the original price data
plt.legend()  # Add legend to the plot

# Plot the Trend Component
plt.subplot(4, 1, 2)  # Second subplot
plt.plot(decomposition.trend, label="Trend", color="red")  # Trend component in red
plt.legend()

# Plot the Seasonality Component
plt.subplot(4, 1, 3)  # Third subplot
plt.plot(decomposition.seasonal, label="Seasonality", color="green")  # Seasonal pattern in green
plt.legend()

# Plot the Residual (Noise) Component
plt.subplot(4, 1, 4)  # Fourth subplot
plt.plot(decomposition.resid, label="Residual (Noise)", color="gray")  # Residual component in gray
plt.legend()
# Adjust layout to prevent overlapping of plots
plt.tight_layout()
# Display the complete decomposition plots
plt.show()

#High Autocorrelation values past values strongly influence future values.
# Computing autocorrelation for different lags
lags = [1, 7, 30, 90, 180, 365]  # Daily, weekly, monthly, quarterly, semi-annual, annual
autocorr_values = [df["Price"].autocorr(lag=lag) for lag in lags]

# Created a DataFrame to display results
autocorr_df = pd.DataFrame({"Lag": lags, "Autocorrelation": autocorr_values})

print(autocorr_df)

# Applying log transformation to Price
df['Price']=np.log(df['Price'])

# Q-Q Plot (Price After log Transformation)
plt.figure(figsize=(8,6))
stats.probplot(df["Price"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Price")
plt.show()

df=df.set_index("Date")

train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

train
test
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")


########################################################
                        #ARIMA
########################################################

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import itertools
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# Define the p, d, q parameters for ARIMA model
p = d = q=range(0,3)

# Generate all possible combinations of p, d, q
pdq_combinations = list(itertools.product(p, d, q))

# Grid search for best ARIMA model
best_model = None
best_aic = float('inf')
best_order = None

for order in pdq_combinations:
    try:
        # Fit an ARIMA model using the current order (p, d, q)
        model = ARIMA(train['Price'], order=order)
        model_fit = model.fit()
        # Check if the current model has a lower AIC (Akaike Information Criterion)
        # A lower AIC indicates a better-fitting model
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_model = model_fit
            best_order = order
    except:
        continue

# Display the best ARIMA model parameters
print(f"Best ARIMA Model Order: {best_order}")
print(f"Best AIC: {best_aic}")

# Forecasting using the best ARIMA model
forecast_steps = len(test)
forecast = best_model.forecast(steps=forecast_steps)

# Evaluate model performance
arima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast))
arima_mape = mean_absolute_percentage_error(test['Price'], forecast)
print(f"ARIMA RMSE: {arima_rmse}")
print(f"ARIMA MAPE: {arima_mape}")

arima_r2 = r2_score(test['Price'], forecast)  
print(f"R-squared (R²) Score arima: {arima_r2:.4f}")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Price'], label='Training Data', color='blue')
plt.plot(test.index, test['Price'], label='Actual Prices', color='green')
plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='dashed', color='red')
plt.xlabel('Date')
plt.ylabel('Iron Price')
plt.title('ARIMA Model Forecast vs Actual')
plt.legend()
plt.show()


########################################################
                        #SARIMA
########################################################

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR logs

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppresses additional TensorFlow logs

# Splitting the dataset into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Define the SARIMA model parameters
# (p, d, q) are the non-seasonal parameters, (P, D, Q, m) are the seasonal parameters.

# Define SARIMA model with optimized parameters
sarima_model = SARIMAX(train["Price"], 
                       order=(1, 1, 1),  # Reduce q
                       seasonal_order=(1, 1, 1, 7),  # Use weekly seasonality
                       enforce_stationarity=False, enforce_invertibility=False)

# Fit with faster method and low memory
sarima_model_fit = sarima_model.fit(method="lbfgs", low_memory=True , disp=False)

# Forecast
forecast_steps = len(test)
forecast = sarima_model_fit.forecast(steps=forecast_steps)


# Calculate the Mean Squared Error (MSE) between the forecasted and actual values
# Calculate the Mean Absolute Percentage Error (MAPE) for the SARIMA model
Sarima_rmse = np.sqrt(mean_squared_error(test["Price"], forecast))
Sarima_mape=mean_absolute_percentage_error(test["Price"], forecast)
print(f"Mean Squared Error Sarima: {Sarima_rmse}")
print(f"Mean Absolute Percentage Error (MAPE) Sarima: {Sarima_mape}")

# Calculate the R-squared (R²) score to measure model performance
Sarima_r2 = r2_score(test['Price'], forecast)  # y_test comes first!
print(f"R-squared (R²) Score Sarima: {Sarima_r2:.4f}")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["Price"], label="Train Data", color="blue")
plt.plot(test.index, test["Price"], label="Test Data", color="green")
plt.plot(test.index, forecast, label="Forecasted Data", color="red")
plt.title("SARIMA Model Forecasting")
plt.xlabel("Date")
plt.ylabel("Log(Price)")
plt.legend()
plt.xticks(rotation=45)
plt.show()

######################################
                #LSTM
######################################


# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Split into train and test sets
train_data = df[df.index < '2022-01-01']
test_data = df[df.index >= '2022-01-01']

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['Price'].values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data['Price'].values.reshape(-1, 1))

# Create dataset for time series forecasting
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time_step (365 for one year of data)
time_step = 365
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Reshaping for the LSTM model
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshaping to [samples, time_steps, features]
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model()

# Fit the LSTM model
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions
lstm_predictions = lstm_model.predict(X_test_lstm)

# Inverse transform the predictions and actual values
lstm_predictions = np.exp(lstm_predictions)  # Reverting log transformation
y_test_actual = np.exp(y_test)  # Reverting log transformation for actual data

# Calculate Root Mean Squared Error for the LSTM model
# Calculate the Mean Absolute Percentage Error (MAPE) for the LSTM model
LSTM_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
LSTM_mape=mean_absolute_percentage_error(y_test_actual, lstm_predictions)
print(f"Root Mean Squared Error (LSTM): {LSTM_rmse}")
print(f"Mean Absolute Percentage Error (MAPE) LSTM: {LSTM_mape}")

# Calculate the R-squared (R²) score to measure model performance
LSTM_r2 = r2_score(y_test_actual, lstm_predictions)  
print(f"R-squared (R²) Score LSTM: {LSTM_r2:.4f}")

# Ensure the correct alignment of test dates with the length of predictions
test_dates = test_data.index[time_step:]  # Starting from the time_step index to match predictions length

# Ensure both test dates and predictions have the same length (cutting the excess dates or predictions)
test_dates = test_dates[:len(lstm_predictions)]  # Align the dates to the number of predictions

# Plotting the results
plt.figure(figsize=(12, 6))

# Plotting the actual and predicted values
plt.plot(test_dates, y_test_actual, label="Actual Price", color="green")
plt.plot(test_dates, lstm_predictions, label="Predicted Price (LSTM)", color="red")

plt.legend()
plt.title("LSTM Model Forecasting 2023-2024 Predictions)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()


###############################################
                #XGBoost
###############################################

import xgboost as xgb

# Create dataset function for time series forecasting
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time_step (365 for one year of data)
time_step = 365

# Prepare the dataset for training and testing
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Reshaping the dataset for XGBoost [samples, time_steps]
X_train_xgb = X_train.reshape(X_train.shape[0], X_train.shape[1])  # Flatten for XGBoost
X_test_xgb = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Define the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Train the model
model.fit(X_train_xgb, y_train)

# Make predictions
XG_predictions = model.predict(X_test_xgb)

# Inverse transform predictions and actual values to get back to original scale
XG_predictions = scaler.inverse_transform(XG_predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Root Mean Squared Error
# Calculate the Mean Absolute Percentage Error (MAPE)
XGBoost_rmse = np.sqrt(mean_squared_error(y_test_actual, XG_predictions))
XGBoost_mape=mean_absolute_percentage_error(y_test_actual, XG_predictions)
print(f"Root Mean Squared Error (XGBoost): {XGBoost_rmse}")
print(f"Mean Absolute Percentage Error (MAPE) XGBoost: {XGBoost_mape}")

# Calculate the R-squared (R²) score to measure model performance
XGBoost_r2 = r2_score(y_test_actual, XG_predictions)  
print(f"R-squared (R²) Score XGBoost: {XGBoost_r2:.4f}")

# Ensure correct alignment of the test dates and predictions
# Aligning the test_dates with the length of predictions
test_dates = test_data.index[time_step:time_step + len(XG_predictions)]  # Aligning test dates with predictions

# Plotting the results
plt.figure(figsize=(12, 6))

# Plotting the actual and predicted values
plt.plot(test_dates, y_test_actual, label="Actual Price", color="green")
plt.plot(test_dates, XG_predictions, label="Predicted Price (XGBoost)", color="red")

plt.legend()
plt.title("XGBoost Model Forecasting (2023-2024 Predictions)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()


#########################################
              #GRU
#########################################

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine, text
import random
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score

# Load dataset from CSV file
data = pd.read_csv(r"E:/360DigiTmg/ML Project/Python/Data_set_Iron/Data_set_Iron.csv")

# Database connection setup
user = 'root'
password = 'password'
db = 'Iron_ore'

# Creating SQL engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

# Storing dataset in MySQL table
data.to_sql('iron_ore_prices', con=engine, if_exists='replace', index=False, chunksize=1000)

# Fetching data from database
sql = 'select * from iron_ore_prices'
df = pd.read_sql_query(text(sql), con=engine.connect())

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(data["Date"], format='%m/%d/%Y')

# Sorting data by Date and setting it as index
df = df.sort_values("Date")  

df=df.set_index("Date")

# Set random seed for reproducibility
SEED = 42  # Fixed seed to ensure consistent predictions
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Split into train and test sets
train_data = df[df.index < '2022-01-01']
test_data = df[df.index >= '2022-01-01']

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['Price'].values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data['Price'].values.reshape(-1, 1))

# Save the scaler to a pickle file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create dataset for time series forecasting
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time_step (365 for one year of data)
time_step = 365
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Reshaping the dataset for GRU [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the GRU model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize a Sequential model for building the GRU-based neural network
GRU_model = Sequential()

# Add GRU layer
GRU_model.add(GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
GRU_model.add(Dropout(0.2))  # Add Dropout to prevent overfitting

# Add output layer
GRU_model.add(Dense(units=1))

# Compile the model
GRU_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = GRU_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
GRU_mse = GRU_model.evaluate(X_test, y_test)

# Make predictions
GRU_predictions = GRU_model.predict(X_test)

# Save the trained GRU model to a file in HDF5 format for later use
GRU_model.save("GRU_model1.h5")

# Calculate Mean Squared Error for GRU
GRU_mse = mean_squared_error(y_test, GRU_predictions)

# Calculate RMSE
GRU_rmse = np.sqrt(GRU_mse)

# Calculate MAPE
GRU_mape = mean_absolute_percentage_error(y_test, GRU_predictions)

# Print results
print(f"Mean Squared Error (MSE) GRU: {GRU_mse:.6f}")
print(f"Root Mean Squared Error (RMSE) GRU: {GRU_rmse:.6f}")
print(f"Mean Absolute Percentage Error (MAPE) GRU: {GRU_mape:.6f}")

#R square Accuracy
GRU_r2 = r2_score(y_test, GRU_predictions)  
print(f"R-squared (R²) Score GRU: {GRU_r2:.4f}")

# Inverse transform predictions and actual values
GRU_predictions = scaler.inverse_transform(GRU_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Ensure correct alignment of the test dates and predictions
test_dates = test_data.index[time_step:time_step + len(GRU_predictions)]  # Aligning test dates with predictions

# Plotting the results
plt.figure(figsize=(12, 6))

# Plotting the actual and predicted values
plt.plot(test_dates, y_test_actual, label="Actual Price", color="green")
plt.plot(test_dates, GRU_predictions, label="Predicted Price", color="red")

plt.title("GRU Model Forecasting (2023-2024)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# Create DataFrame with test dates, actual prices, and predicted prices
results_df_p = pd.DataFrame({
    "Date": test_dates,
    "Actual Price": y_test_actual.flatten(),
    "Predicted Price": GRU_predictions.flatten()
})

# Reset index for better readability
results_df_p.reset_index(drop=True, inplace=True)

# Display first few rows
print(results_df_p.head())

# Connect to the database and execute a SQL command to drop the table if it exists
with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS GRU_forecast"))
    connection.commit()

# Save the DataFrame to a SQL table named 'GRU_forecast'
results_df_p.to_sql('GRU_forecast', con=engine, if_exists='replace', index=False, chunksize=1000)



###################################################
                   #CNN-LSTM
###################################################

from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Ensure that your test set spans the desired range (from 2022 to 2024)
train_data = df[df.index < '2022-01-01']
test_data = df[df.index >= '2022-01-01']  # This will make sure test data starts from 2022

# Scaling the data (only 'Price' for prediction)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[['Price']].values)  
scaled_test_data = scaler.transform(test_data[['Price']].values)

# Create dataset for time series forecasting (univariate)
def create_univariate_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])  # Only 'Price' as input
        y.append(dataset[i + time_step, 0])  # Predicting the next 'Price'
    return np.array(X), np.array(y)

# Define time_step (365 for one year of data)
time_step = 365
X_train, y_train = create_univariate_dataset(scaled_train_data, time_step)
X_test, y_test = create_univariate_dataset(scaled_test_data, time_step)

# Reshaping for CNN-LSTM model (univariate)
X_train_cnn_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # [samples, time_steps, 1 feature]
X_test_cnn_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN-LSTM model (same as before)
def build_cnn_lstm_model():
    model = Sequential()

    # CNN Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn_lstm.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM Layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Output Layer
    model.add(Dense(units=1))  # Predicting the next 'Price'
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

cnn_lstm_model = build_cnn_lstm_model()

# Fit the CNN-LSTM model
cnn_lstm_model.fit(X_train_cnn_lstm, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions
cnn_lstm_predictions = cnn_lstm_model.predict(X_test_cnn_lstm)

# Inverse transform the predictions and actual values
cnn_lstm_predictions = scaler.inverse_transform(cnn_lstm_predictions)  # Reverting scaling
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reverting scaling for actual data

# Calculate Mean Squared Error for the CNN-LSTM model
# Calculate the Mean Absolute Percentage Error (MAPE)
CNN_LSTM_rmse = np.sqrt(mean_squared_error(y_test_actual, cnn_lstm_predictions))
CNN_LSTM_mape = mean_absolute_percentage_error(y_test_actual, cnn_lstm_predictions)
print(f"Mean Squared Error (CNN-LSTM): {CNN_LSTM_rmse}")
print(f"Mean Absolute Percentage Error (MAPE) CNN_LSTM: {CNN_LSTM_mape}")

# R Square Accuracy
CNN_LSTM_r2 = r2_score(y_test_actual, cnn_lstm_predictions)  
print(f"R-squared (R²) Score CNN_LSTM: {CNN_LSTM_r2:.4f}")

# Ensure the correct alignment of test dates with the length of predictions
test_dates = test_data.index[time_step:]  # Starting from the time_step index to match predictions length
test_dates = test_dates[:len(cnn_lstm_predictions)]  # Align dates with the number of predictions

# Plotting the results
plt.figure(figsize=(12, 6))

# Plotting the actual and predicted values
plt.plot(test_dates, y_test_actual, label="Actual Price", color="green")
plt.plot(test_dates, cnn_lstm_predictions, label="Predicted Price (CNN-LSTM)", color="red")

plt.legend()
plt.title("CNN-LSTM Model Forecasting (2023-2024 Predictions)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()


# Store the RMSE and MAPE evaluation metrics for different forecasting models in a dictionary
results = {
    "ARIMA": {"RMSE": arima_rmse, "MAPE": arima_mape},
    "SARIMA": {"RMSE": Sarima_rmse, "MAPE": Sarima_mape},
    "LSTM": {"RMSE": LSTM_rmse, "MAPE": LSTM_mape},
    "XGBoost": {"RMSE": XGBoost_rmse, "MAPE": XGBoost_mape},
    "GRU": {"RMSE": GRU_rmse, "MAPE": GRU_mape},
    "CNN-LSTM":{"RMSE":CNN_LSTM_rmse,"MAPE":CNN_LSTM_mape}
}

# Convert dictionary to DataFrame
results_new = pd.DataFrame(results).T

# Display the dictionary
print(results)

# Create a DataFrame with only R² values
r2_results_df = pd.DataFrame.from_dict({
    "ARIMA": arima_r2,
    "SARIMA": Sarima_r2,
    "LSTM": LSTM_r2,
    "XGBoost": XGBoost_r2,
    "GRU": GRU_r2,
    "CNN-LSTM": CNN_LSTM_r2
}, orient='index', columns=["R² Score"])

# Display the DataFrame
print(r2_results_df)


