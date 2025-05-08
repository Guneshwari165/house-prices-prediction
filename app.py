import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title
st.title("California House Price Prediction")

# Load dataset from CSV
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.write(df)

# Preprocessing
df = df.dropna()  # Remove rows with missing values
df = df.drop("ocean_proximity", axis=1)  # Drop non-numeric column for simplicity

# Split features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Output
st.subheader("Model Performance")
st.write(f"Root Mean Squared Error: {rmse:,.2f}")

# Predict on user input
st.subheader("Try it Yourself: Predict House Value")
user_input = {}
for col in X.columns:
    val = st.number_input(f"Input value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.write(f"**Predicted Median House Value:** ${prediction:,.2f}")
