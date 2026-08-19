"""Validate the dataset expected by the course notebooks."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "life_satisfaction_2010_2024.csv"
REQUIRED_COLUMNS = {
    "Country",
    "Code",
    "Year",
    "Life Satisfaction",
    "GDP per capita",
    "Employment Rate (%)",
}
NUMERIC_COLUMNS = [
    "Year",
    "Life Satisfaction",
    "GDP per capita",
    "Employment Rate (%)",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values")

    for column in NUMERIC_COLUMNS:
        pd.to_numeric(df[column], errors="raise")

    df_2024 = df.loc[df["Year"] == 2024]
    if df_2024.empty:
        raise ValueError("No observations found for 2024")
    if df_2024["Country"].duplicated().any():
        raise ValueError("Duplicate country observations found for 2024")

    required_countries = {"Japan", "South Korea", "Italy"}
    missing_countries = required_countries.difference(df_2024["Country"])
    if missing_countries:
        raise ValueError(f"Missing required 2024 countries: {sorted(missing_countries)}")

    print(f"OK: {len(df):,} rows, {df['Year'].min()}–{df['Year'].max()}")
    print(f"OK: {len(df_2024)} unique country observations for 2024")
    print("OK: required columns are complete and numeric fields are valid")


if __name__ == "__main__":
    main()
