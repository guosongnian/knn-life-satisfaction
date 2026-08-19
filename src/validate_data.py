"""验证项目 notebook 所需的数据集。"""

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
        raise ValueError(f"缺少字段：{sorted(missing_columns)}")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("必需字段中存在缺失值")

    for column in NUMERIC_COLUMNS:
        pd.to_numeric(df[column], errors="raise")

    df_2024 = df.loc[df["Year"] == 2024]
    if df_2024.empty:
        raise ValueError("没有找到 2024 年的数据")
    if df_2024["Country"].duplicated().any():
        raise ValueError("2024 年数据中存在重复国家")

    required_countries = {"Japan", "South Korea", "Italy"}
    missing_countries = required_countries.difference(df_2024["Country"])
    if missing_countries:
        raise ValueError(f"2024 年数据缺少必需国家：{sorted(missing_countries)}")

    print(f"通过：共 {len(df):,} 行，年份范围 {df['Year'].min()}—{df['Year'].max()}")
    print(f"通过：2024 年包含 {len(df_2024)} 个不重复国家")
    print("通过：必需字段完整，数值字段格式有效")


if __name__ == "__main__":
    main()
