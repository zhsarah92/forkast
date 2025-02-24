# Forecast Model Training and Prediction

This repository contains a script for training and making predictions with a forecast model. The script allows you to train a new model or use an existing model for making predictions based on various parameters.

## Requirements

Ensure you have the following installed:
- Python 3.11
- Required dependencies (can be installed via pip)

To install dependencies, you can run:

```
pip install -r requirements.txt
```

## Usage
### Training the Model

To train a new model, use the `--train` flag and specify the paths to the CSV files for bills and venues data using `--bill_file` and `--venue_file`. This will train a new model and save it as a .pkl file with a unique timestamp in the filename.

```
python forecast.py --train --bill_file ./data/bills.csv --venue_file ./data/venues.csv
```

- `--bill_file` (str): Path to the bill data CSV file.
- `--venue_file` (str): Path to the venue data CSV file.

This will save a model with a name like `20250222123045_model.pkl`.


### Making Predictions

To use a trained model to make predictions, use the --predict flag. You need to specify the following parameters:

    --concept: Concept for prediction (e.g., 'FINE_DINING').
    --city: City for prediction.
    --country: Country for prediction.
    --future_periods: The number of future periods to forecast.
    --model: Path to the trained model .pkl file.

Example:

```
python forecast.py --predict --concept FINE_DINING --city Toronto --country CA --future_periods 120 --model ./20250222123045_model.pkl
```