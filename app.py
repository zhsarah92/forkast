import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from prophet import Prophet
from data_processing import DataProcessor
import os

class ForecastModel:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path # Model path is provided by the user
        self.label_encoders = None

    def prepare_data(self, bills_csv: str, venues_csv: str):
        data_processor = DataProcessor(bills_csv, venues_csv)
        data_processor.clean_data()
        data_processor.process_data()
        data, self.label_encoders = data_processor.get_hourly_orders()
        self.regressors = [col for col in data.columns if col not in ["ds", "y"]]
        return data

    def load_model(self):
        if self.model_path is None or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model, self.regressors, self.label_encoders = joblib.load(self.model_path)
        return self.model

    def forecast_and_plot(self, concept: str, city: str, country: str, future_periods: int = 48):
        if self.model is None:
            self.load_model()

        future = self.model.make_future_dataframe(periods=future_periods, freq="H")

        future["concept"] = self.label_encoders["concept"].transform([concept] * len(future))
        future["city"] = self.label_encoders["city"].transform([city] * len(future))
        future["country"] = self.label_encoders["country"].transform([country] * len(future))

        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

        future_forecast = forecast.tail(future_periods)

        plt.figure(figsize=(10, 5))
        plt.plot(future_forecast["ds"], future_forecast["yhat"], label="Forecast", color="blue")
        plt.fill_between(
            future_forecast["ds"], future_forecast["yhat_lower"], future_forecast["yhat_upper"], color="blue", alpha=0.2
        )
        plt.xlabel("Date")
        plt.ylabel("Predicted Orders")
        plt.title(f"Future Order Forecast for {concept} in {city}, {country}")
        plt.legend()
        plt.grid()

        plot_path = f"./{Path(self.model_path).stem}_forecast_{concept}_{city}_{country}.png"
        plt.savefig(plot_path)

        return plot_path


def main():
    # Page layout and styling
    st.set_page_config(page_title="Forkast", page_icon="📊", layout="centered")
    st.title("FORKAST 🍽️")

    # Sidebar with title and description
    st.sidebar.header("About")
    st.sidebar.write("For the CXC Datathon, we developed an innovative order forecasting application designed to address the TouchBistro Challenge. Our app leverages a machine learning model trained with historical order data to generate accurate forecasts for restaurants and venues. By inputting parameters such as the concept, city, and country, users can quickly obtain future order predictions. Our solution showcases how data-driven forecasting can be applied to real-world challenges in the foodservice industry.")
    st.sidebar.markdown("### [See our project files](https://discord.com/channels/@me/1337846893199364287)")

    # Add a model download link in the sidebar 
    model_download_url = "https://drive.google.com/uc?export=download&id=18srWcM5xms9Fd4HFWKItkAYQypQS0AYe"
    st.markdown(f"[Download our model](<{model_download_url}>)")

    # User uploads their model
    model_file = st.file_uploader("Upload your model (.pkl file)", type=["pkl"])

    forecast_model = None

    if model_file:
        model_path = f"./uploaded_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        forecast_model = ForecastModel(model_path=model_path)

    with st.expander("What does our model do?"):
        st.write(""" 
            This model forecasts restaurant order volumes for TouchBistro by analyzing past billing data. 
            It helps predict how many orders a restaurant will receive in future hours or days, allowing restaurants to make better business decisions.
            """)

    with st.expander("How does this help TouchBistro?"):
        st.write(""" 
            Forkast allows TouchBistro to offer their customers a way to predict future order demand.
            This would allow restaurants to manage year-long inventories, create dynamic pricing plans, and also gives TouchBistro insights on their customers all over the world.
            """)

    # User input fields with dropdown menus 
    concept_options = ["FINE_DINING", "CASUAL_DINING", "FAST_CASUAL", "CAFÉ"]
    concept = st.selectbox("Concept", concept_options, placeholder="Choose a concept", index=None)
    city_options = ["Toronto", "Vancouver", "Montreal", "Calgary"]
    city = st.selectbox("City", city_options, placeholder="Choose a city", index=None)
    country_options = ["CA", "US", "UK", "AU"]
    country = st.selectbox("Country", country_options, placeholder="Choose a country", index=None)

    future_periods = st.slider("Number of future periods", min_value=0, max_value=300)

    # Buttons and interactive elements
    st.markdown("### Forecast Options")
    st.write("Click the button below to generate a forecast for the specified concept, city, and country.")

    # Generate forecast button
    if st.button("Generate Forecast", key="forecast_button"):
        if forecast_model:
            st.spinner("Generating forecast...")
            plot_path = forecast_model.forecast_and_plot(concept, city, country, future_periods)

            # Display forecast plot with cool interactive features
            st.image(plot_path, caption="Forecast Plot", use_container_width=True)
            st.success(f"Forecast plot generated and saved as {plot_path}")
        else:
            st.error("Please upload a valid model file first.")

if __name__ == "__main__":
    main()