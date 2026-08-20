"""从零实现 k-NN 回归，预测国家生活满意度。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["GDP per capita", "Employment Rate (%)"]
TARGET = "Life Satisfaction"
REQUIRED_COLUMNS = ["Country", "Code", "Year", TARGET, *FEATURES]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """校验数据并返回 2024 年训练集和日本测试样本。"""
    raw = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    if raw[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("必需字段中存在缺失值")

    data = raw.loc[raw["Year"] == 2024].copy()
    if data["Country"].duplicated().any():
        raise ValueError("2024 年数据中存在重复国家")
    data = data.set_index("Country")
    if "Japan" not in data.index:
        raise ValueError("2024 年数据中缺少 Japan")
    return data.drop(index="Japan"), data.loc[["Japan"]]


def fit_standardizer(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """仅使用当前训练数据估计标准化参数。"""
    return frame.mean(), frame.std()


def standardize(
    frame: pd.DataFrame | pd.Series,
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame | pd.Series:
    """使用训练集参数变换数据。"""
    if (std == 0).any():
        raise ValueError("至少一个特征的标准差为 0")
    return (frame - mean) / std


def knn_predict(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    query_scaled: pd.Series,
    k: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回预测值、近邻位置和全部距离。"""
    if not 1 <= k <= len(x_train_scaled):
        raise ValueError("k 必须位于 1 与训练样本数之间")
    distances = np.linalg.norm(
        x_train_scaled.to_numpy(dtype=float)
        - query_scaled.to_numpy(dtype=float),
        axis=1,
    )
    neighbor_positions = np.argsort(distances)[:k]
    prediction = float(y_train.iloc[neighbor_positions].mean())
    return prediction, neighbor_positions, distances


def leave_one_out_errors(
    x: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """在每个验证折内重新标准化，计算所有合法 k 的绝对误差。"""
    k_values = list(range(1, len(x)))
    errors: dict[str, list[float]] = {}
    for validation_country in x.index:
        x_train = x.drop(index=validation_country)
        y_train = y.drop(index=validation_country)
        mean, std = fit_standardizer(x_train)
        x_train_scaled = standardize(x_train, mean, std)
        query_scaled = standardize(x.loc[validation_country], mean, std)
        errors[validation_country] = [
            abs(
                knn_predict(x_train_scaled, y_train, query_scaled, k)[0]
                - float(y.loc[validation_country])
            )
            for k in k_values
        ]
    return pd.DataFrame(errors, index=k_values).T


def linear_loo_mae(x: pd.DataFrame, y: pd.Series) -> float:
    """计算无第三方模型依赖的线性回归留一验证 MAE。"""
    errors = []
    for validation_country in x.index:
        x_train = x.drop(index=validation_country)
        y_train = y.drop(index=validation_country)
        mean, std = fit_standardizer(x_train)
        x_train_scaled = standardize(x_train, mean, std)
        query_scaled = standardize(x.loc[validation_country], mean, std)
        design = np.column_stack(
            [np.ones(len(x_train_scaled)), x_train_scaled.to_numpy(dtype=float)]
        )
        coefficients = np.linalg.lstsq(
            design, y_train.to_numpy(dtype=float), rcond=None
        )[0]
        prediction = float(
            np.r_[1.0, query_scaled.to_numpy(dtype=float)] @ coefficients
        )
        errors.append(abs(prediction - float(y.loc[validation_country])))
    return float(np.mean(errors))


def run(root: Path) -> dict[str, object]:
    """运行模型选择、基线比较和日本样本最终预测。"""
    train, test = load_data(root / "data" / "life_satisfaction_2011_2024.csv")
    x = train[FEATURES].astype(float)
    y = train[TARGET].astype(float)
    errors = leave_one_out_errors(x, y)
    mean_errors = errors.mean(axis=0)
    best_k = int(mean_errors.idxmin())

    mean, std = fit_standardizer(x)
    x_scaled = standardize(x, mean, std)
    japan_scaled = standardize(test.loc["Japan", FEATURES].astype(float), mean, std)
    prediction, positions, distances = knn_predict(
        x_scaled, y, japan_scaled, best_k
    )
    actual = float(test.loc["Japan", TARGET])
    result = {
        "selected_k": best_k,
        "leave_one_out_mae": float(mean_errors.loc[best_k]),
        "linear_baseline_mae": linear_loo_mae(x, y),
        "mean_baseline_mae": float(
            np.mean([abs(y.drop(index=i).mean() - y.loc[i]) for i in y.index])
        ),
        "japan_prediction": prediction,
        "japan_actual": actual,
        "japan_absolute_error": abs(prediction - actual),
        "neighbors": [str(x.index[position]) for position in positions],
        "neighbor_distances": [float(distances[position]) for position in positions],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
