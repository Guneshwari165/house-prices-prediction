import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load data
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df

# Preprocess and train models
@st.cache_data
def train_models(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipelines
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]

    # Stacking Regressor
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression()
    )

    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("stacking", stacking_model)
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model_pipeline, rmse, X.columns.tolist()

# Main UI
def main():
    st.title("🏡 Smart House Price Forecaster")
    st.markdown("Using advanced regression models (Random Forest, XGBoost, Stacking)")

    df = load_data()
    model, rmse, feature_names = train_models(df)

    st.write("### Model RMSE on test data:", round(rmse, 3))

    st.sidebar.header("Enter House Features")

    input_data = {}
    for feature in feature_names:
        val = st.sidebar.slider(
            label=f"{feature}",
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean())
        )
        input_data[feature] = val

    input_df = pd.DataFrame([input_data])

    if st.button("Predict House Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Estimated House Price: ${prediction * 100000:.2f}")

    if st.checkbox("Show Sample Data"):
        st.write(df.head())

if __name__ == "__main__":
    main()