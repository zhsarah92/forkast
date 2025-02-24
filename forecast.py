import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from datetime import datetime
from prophet import Prophet

from data_processing import DataProcessor


def parse_args():
    """Argument parser for the forecast model"""
    parser = argparse.ArgumentParser(description="Train or predict using the forecast model.")

    # Add arguments for both train and predict options
    parser.add_argument("--train", action="store_true", help="Flag to train the model.")
    parser.add_argument("--predict", action="store_true", help="Flag to make predictions.")

    # Arguments for training
    parser.add_argument("--bill_file", type=str, help="Path to the bill data CSV file.")
    parser.add_argument("--venue_file", type=str, help="Path to the venue data CSV file.")

    # Arguments for prediction
    parser.add_argument("--concept", type=str, help="Concept for prediction (e.g., 'FINE_DINING').")
    parser.add_argument("--city", type=str, help="City for prediction.")
    parser.add_argument("--country", type=str, help="Country for prediction.")
    parser.add_argument("--future_periods", type=int, help="Number of future periods for forecasting.")
    parser.add_argument("--model", type=str, help="Path to the trained model file for prediction.")

    return parser.parse_args()


class ForecastModel:
    def __init__(self, model_path: str = "model.pkl"):
        """Create a new ForecastModel instance.

        Parameters
        ----------
        model_path : str, optional
            Path to save the trained model, by default "model.pkl"
        """
        self.model = None
        self.model_path = model_path
        self.label_encoders = None

    def prepare_data(self, bills_csv: str, venues_csv: str):
        """Prepare data for training the model.
        Also saves the label encoders and refressor names for use in inference

        Parameters
        ----------
        bills_csv : str
            path to the bills csv file containing hourly order data for training
        venues_csv : str
            path to the venues csv file containing venue information for training

        Returns
        -------
        dataframe
            Dataframe containing hourly order data for multiple venues.
        """
        data_processor = DataProcessor(bills_csv, venues_csv)
        data_processor.clean_data()
        data_processor.process_data()
        data, self.label_encoders = data_processor.get_hourly_orders()
        self.regressors = [col for col in data.columns if col not in ["ds", "y"]]

        return data

    def train_model(self, data: pd.DataFrame):
        """Train a Prophet model on the given data.

        Parameters
        ----------
        data : dataframe
            Dataframe containing hourly order data for multiple venues.

        Returns
        -------
        model : Prophet
            Trained Prophet model.
        """
        self.model = Prophet()

        # Add categorical features as regressors
        for col in self.regressors:
            self.model.add_regressor(col)

        # Constrain predictions to be non-negative
        self.model.fit(data)

        # Save model per venue
        print(f"Saving model to {self.model_path}")
        joblib.dump((self.model, self.regressors, self.label_encoders), self.model_path)

        return self.model

    def load_model(self):
        """Load a trained prophet model from disk.

        Returns
        -------
        model : Prophet
            Loaded Prophet model.

        Raises
        ------
        FileNotFoundError
            If the model file is not found.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model, self.regressors, self.label_encoders = joblib.load(self.model_path)
        return self.model

    def forecast_and_plot(
        self, concept: str, city: str, country: str, future_periods: int = 48, plot_path: Optional[str] = None
    ):
        """Make future predictions for a given venue and plot the results.

        Parameters
        ----------
        concept : str
            name of the concept (like 'FINE_DINING', 'BAR', etc.)
        city : str
            name of the city (like 'Toronto', 'New York', etc.)
        country : str
            name of the country (like 'CA', 'US', etc.)
        future_periods : int, optional
            number of future hours to predict for, by default 48
        plot_path : str, optional
            path to save the plot, by default None

        Returns
        -------
        dataframe
            Dataframe containing future predictions.

        Raises
        ------
        ValueError
            if the concept, city, or country is not recognized.
        FileNotFoundError
            if the model is not loaded and model file is not found.
        """
        if self.model is None:
            self.load_model()

        future = self.model.make_future_dataframe(periods=future_periods, freq="H")

        # check if the concept, city, and country are one of expected in label encoders
        if concept not in self.label_encoders["concept"].classes_:
            raise ValueError(
                f"Concept '{concept}' not recognized. Try one of {self.label_encoders['concept'].classes_}"
            )
        if city not in self.label_encoders["city"].classes_:
            raise ValueError(f"City '{city}' not recognized. Try one of {self.label_encoders['city'].classes_}")
        if country not in self.label_encoders["country"].classes_:
            raise ValueError(
                f"Country '{country}' not recognized. Try one of {self.label_encoders['country'].classes_}"
            )

        # Add required regressors with given values
        future["concept"] = self.label_encoders["concept"].transform([concept] * len(future))
        future["city"] = self.label_encoders["city"].transform([city] * len(future))
        future["country"] = self.label_encoders["country"].transform([country] * len(future))

        forecast = self.model.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)  # Ensure predictions are >= 0
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)  # Ensure predictions are >= 0

        # Keep only the last `future_periods` rows for plotting
        future_forecast = forecast.tail(future_periods)

        # # Plot only future predictions
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

        # model.plot(forecast)
        if plot_path is None:
            plot_path = f"./{Path(self.model_path).stem}_forecast_{concept}_{city}_{country}.png"
        plt.savefig(plot_path)

        return forecast


def main():
    args = parse_args()

    if args.train:
        if not args.bill_file or not args.venue_file:
            print("Error: Both --bill_file and --venue_file must be specified for training.")
            return

        # Generate unique identifier for the model
        unique_identifier = datetime.now().strftime("%Y%m%d%H%M%S")

        # Main pipeline for training
        forecast_model = ForecastModel(model_path=f"./{unique_identifier}_model.pkl")
        data = forecast_model.prepare_data(args.bill_file, args.venue_file)
        forecast_model.train_model(data)
        print(f"Model trained and saved as {unique_identifier}_model.pkl")

    elif args.predict:
        if not args.concept or not args.city or not args.country or not args.future_periods or not args.model:
            print(
                "Error: All prediction arguments (--concept, --city, --country, --future_periods, --model) must be specified."
            )
            return

        # Load the model for prediction
        forecast_model = ForecastModel(model_path=args.model)

        # Perform forecasting
        forecast_model.forecast_and_plot(
            concept=args.concept, city=args.city, country=args.country, future_periods=args.future_periods
        )
        print("Prediction completed.")


if __name__ == "__main__":
    main()