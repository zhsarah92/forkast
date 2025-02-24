import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self, bills_csv: str, venues_csv: str):
        """Create a new DataProcessor instance.

        Parameters
        ----------
        bills_csv : str
            path to the bills csv file containing hourly order data for training
        venues_csv : str
            path to the venues csv file containing venue information for training
        """
        self.bills_csv = bills_csv
        self.venues_csv = venues_csv

        print("Loading data...")
        df_bills = pd.read_csv(bills_csv, parse_dates=["bill_paid_at_local"])
        df_venues = pd.read_csv(venues_csv)
        print("Data loaded.")
        self.data = df_bills.merge(df_venues, on="venue_xref_id", how="left")

    def clean_data(self):
        """Clean the data by removing missing values and duplicates."""
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()

    def process_data(self):
        """Process the data by extracting time-based features."""
        # Extract time-based features
        self.data["hour"] = self.data["bill_paid_at_local"].dt.hour
        self.data["day_of_week"] = self.data["bill_paid_at_local"].dt.dayofweek
        self.data["date"] = self.data["bill_paid_at_local"].dt.date

    def get_hourly_orders(self):
        """Get hourly order data for training the model and label encoders for categorical variables.

        Returns
        -------
        dataframe, dict
            Dataframe containing hourly order data for multiple venues and dict of label encoders for categorical variables.
        """
        # Count number of orders per hour per restaurant
        hourly_orders = (
            self.data.groupby([self.data["bill_paid_at_local"].dt.floor("H"), "concept", "city", "country"])
            .size()
            .reset_index()
        )
        hourly_orders.columns = [
            "ds",
            "concept",
            "city",
            "country",
            "y",
        ]

        # Encode categorical variables as numerical labels
        label_encoders = {}
        for col in ["concept", "city", "country"]:
            le = LabelEncoder()
            hourly_orders[col] = le.fit_transform(hourly_orders[col])
            label_encoders[col] = le

        return hourly_orders, label_encoders