import pandas as pd
import glob


def load_data():
    """Load the Kaggle supply-chain dataset."""

    files = glob.glob("data/*.csv")

    if not files:
        raise FileNotFoundError("No CSV dataset found inside data/")

    df = pd.read_csv(files[0])

    return df


def prepare_features(df):
    """Prepare features for machine learning."""

    data = df.copy()

    # ---------------------------------------------
    # Date feature engineering
    # ---------------------------------------------

    data["Date"] = pd.to_datetime(data["Date"])

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day_of_Week"] = data["Date"].dt.dayofweek

    # Shipment ID is only an identifier
    data = data.drop(columns=["Shipment_ID"])

    # Original date is no longer required
    data = data.drop(columns=["Date"])

    # ---------------------------------------------
    # Separate target
    # ---------------------------------------------

    X = data.drop(columns=["Disruption_Occurred"])
    y = data["Disruption_Occurred"]

    # ---------------------------------------------
    # Convert categorical variables
    # ---------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False
    )

    return X, y


if __name__ == "__main__":

    df = load_data()

    print("=" * 60)
    print("PREPROCESSING CHECK")
    print("=" * 60)

    print("\nOriginal shape:")
    print(df.shape)

    X, y = prepare_features(df)

    print("\nFeature matrix shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nPreprocessing completed successfully.")