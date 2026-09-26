import pandas as pd
import numpy as np
import glob
import os


# ============================================================
# LOAD DATA
# ============================================================

files = glob.glob("data/*.csv")

if not files:
    raise FileNotFoundError(
        "No CSV file found inside data/"
    )

file = files[0]

df = pd.read_csv(file)

print("=" * 75)
print("SUPPLY CHAIN TARGET PATTERN ANALYSIS")
print("=" * 75)

print("\nDataset:")
print(file)

print("\nShape:")
print(df.shape)


# ============================================================
# BASIC TARGET INFORMATION
# ============================================================

target = "Disruption_Occurred"

print("\n" + "=" * 75)
print("TARGET DISTRIBUTION")
print("=" * 75)

print(
    df[target].value_counts()
)

print("\nPercentage:")

print(
    (
        df[target]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# FUNCTION: DISRUPTION RATE
# ============================================================

def disruption_rate(column):

    result = (
        df.groupby(column)[target]
        .agg(
            Count="count",
            Disruptions="sum",
            Disruption_Rate="mean"
        )
        .sort_values(
            "Disruption_Rate",
            ascending=False
        )
    )

    result["Disruption_Rate"] *= 100

    result["Disruption_Rate"] = (
        result["Disruption_Rate"]
        .round(2)
    )

    print(
        f"\n--- {column} ---"
    )

    print(result)

    return result


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

categorical_columns = [
    "Weather_Condition",
    "Transport_Mode",
    "Product_Category",
    "Origin_Port",
    "Destination_Port"
]

categorical_results = {}

for column in categorical_columns:

    categorical_results[column] = (
        disruption_rate(column)
    )


# ============================================================
# NUMERICAL FEATURE BANDS
# ============================================================

print("\n" + "=" * 75)
print("NUMERICAL FEATURE ANALYSIS")
print("=" * 75)


numerical_columns = [
    "Distance_km",
    "Weight_MT",
    "Fuel_Price_Index",
    "Geopolitical_Risk_Score",
    "Carrier_Reliability_Score",
    "Lead_Time_Days"
]


for column in numerical_columns:

    print(
        f"\n--- {column} ---"
    )

    try:

        df[f"{column}_Band"] = pd.qcut(
            df[column],
            q=5,
            duplicates="drop"
        )

        result = (
            df.groupby(
                f"{column}_Band",
                observed=True
            )[target]
            .agg(
                Count="count",
                Disruptions="sum",
                Disruption_Rate="mean"
            )
        )

        result["Disruption_Rate"] *= 100

        result["Disruption_Rate"] = (
            result["Disruption_Rate"]
            .round(2)
        )

        print(result)

    except Exception as e:

        print(
            "Could not analyze:",
            e
        )


# ============================================================
# WEATHER + GEOPOLITICAL RISK
# ============================================================

print("\n" + "=" * 75)
print("WEATHER + GEOPOLITICAL RISK")
print("=" * 75)

df["Geo_Risk_Band"] = pd.cut(
    df["Geopolitical_Risk_Score"],
    bins=[
        -np.inf,
        2,
        4,
        6,
        8,
        np.inf
    ],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)


weather_geo = (
    df.groupby(
        [
            "Weather_Condition",
            "Geo_Risk_Band"
        ],
        observed=True
    )[target]
    .agg(
        Count="count",
        Disruptions="sum",
        Disruption_Rate="mean"
    )
)

weather_geo["Disruption_Rate"] *= 100

weather_geo["Disruption_Rate"] = (
    weather_geo["Disruption_Rate"]
    .round(2)
)

print(weather_geo)


# ============================================================
# WEATHER + LEAD TIME
# ============================================================

print("\n" + "=" * 75)
print("WEATHER + LEAD TIME")
print("=" * 75)

df["Lead_Time_Band"] = pd.cut(
    df["Lead_Time_Days"],
    bins=[
        -np.inf,
        5,
        10,
        20,
        40,
        np.inf
    ],
    labels=[
        "Very Short",
        "Short",
        "Medium",
        "Long",
        "Very Long"
    ]
)


weather_lead = (
    df.groupby(
        [
            "Weather_Condition",
            "Lead_Time_Band"
        ],
        observed=True
    )[target]
    .agg(
        Count="count",
        Disruptions="sum",
        Disruption_Rate="mean"
    )
)

weather_lead["Disruption_Rate"] *= 100

weather_lead["Disruption_Rate"] = (
    weather_lead["Disruption_Rate"]
    .round(2)
)

print(weather_lead)


# ============================================================
# WEATHER + CARRIER RELIABILITY
# ============================================================

print("\n" + "=" * 75)
print("WEATHER + CARRIER RELIABILITY")
print("=" * 75)

df["Carrier_Reliability_Band"] = pd.cut(
    df["Carrier_Reliability_Score"],
    bins=[
        -np.inf,
        0.6,
        0.7,
        0.8,
        0.9,
        np.inf
    ],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)


weather_carrier = (
    df.groupby(
        [
            "Weather_Condition",
            "Carrier_Reliability_Band"
        ],
        observed=True
    )[target]
    .agg(
        Count="count",
        Disruptions="sum",
        Disruption_Rate="mean"
    )
)

weather_carrier["Disruption_Rate"] *= 100

weather_carrier["Disruption_Rate"] = (
    weather_carrier["Disruption_Rate"]
    .round(2)
)

print(weather_carrier)


# ============================================================
# TOP RISK COMBINATIONS
# ============================================================

print("\n" + "=" * 75)
print("TOP RISK COMBINATIONS")
print("=" * 75)


combination = (
    df.groupby(
        [
            "Weather_Condition",
            "Transport_Mode",
            "Product_Category"
        ]
    )[target]
    .agg(
        Count="count",
        Disruptions="sum",
        Disruption_Rate="mean"
    )
)

combination["Disruption_Rate"] *= 100

combination["Disruption_Rate"] = (
    combination["Disruption_Rate"]
    .round(2)
)


# Only combinations with at least 20 records
combination = combination[
    combination["Count"] >= 20
]

print(
    combination
    .sort_values(
        "Disruption_Rate",
        ascending=False
    )
    .head(20)
)


# ============================================================
# LOWEST RISK COMBINATIONS
# ============================================================

print("\n" + "=" * 75)
print("LOWEST RISK COMBINATIONS")
print("=" * 75)

print(
    combination
    .sort_values(
        "Disruption_Rate",
        ascending=True
    )
    .head(20)
)


# ============================================================
# CORRELATION
# ============================================================

print("\n" + "=" * 75)
print("NUMERICAL CORRELATIONS")
print("=" * 75)

numeric = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = (
    numeric.corr()[target]
    .sort_values(
        ascending=False
    )
)

print(correlation)


# ============================================================
# CHECK EXTREME TARGET PATTERNS
# ============================================================

print("\n" + "=" * 75)
print("EXTREME WEATHER PATTERN CHECK")
print("=" * 75)

for weather in df[
    "Weather_Condition"
].unique():

    subset = df[
        df["Weather_Condition"] == weather
    ]

    rate = subset[target].mean() * 100

    print(
        f"{weather:15s} "
        f"records={len(subset):4d} "
        f"disruption={rate:6.2f}%"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

df.to_csv(
    "outputs/target_analysis_data.csv",
    index=False
)


print("\n" + "=" * 75)
print("TARGET ANALYSIS COMPLETED")
print("=" * 75)

print(
    "\nSaved:"
)

print(
    "outputs/target_analysis_data.csv"
)