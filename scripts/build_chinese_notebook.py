"""生成合并后的中文求职展示 notebook。"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "life_satisfaction_2010_2024.csv"
OUTPUT_PATH = ROOT / "notebooks" / "knn_life_satisfaction_zh.ipynb"
FEATURES = ["GDP per capita", "Employment Rate (%)"]
TARGET = "Life Satisfaction"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def stream_output(text: str) -> dict:
    return {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}


def dataframe_output(frame: pd.DataFrame, count: int) -> dict:
    return {
        "data": {
            "text/html": frame.to_html(),
            "text/plain": repr(frame),
        },
        "execution_count": count,
        "metadata": {},
        "output_type": "execute_result",
    }


def figure_output(fig: plt.Figure) -> dict:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {
        "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text/plain": "<Figure>",
        },
        "metadata": {},
        "output_type": "display_data",
    }


def code(source: str, count: int, outputs: list[dict] | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def fit_standardizer(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame.mean(), frame.std()


def standardize(frame: pd.DataFrame | pd.Series, mean: pd.Series, std: pd.Series):
    return (frame - mean) / std


def knn_predict(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    x_query_scaled: pd.Series,
    k: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(
        x_train_scaled.to_numpy(dtype=float) - x_query_scaled.to_numpy(dtype=float),
        axis=1,
    )
    neighbor_positions = np.argsort(distances)[:k]
    prediction = float(y_train.iloc[neighbor_positions].mean())
    return prediction, neighbor_positions, distances


def loo_knn_errors(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    k_values = list(range(1, len(x)))
    error_rows: dict[str, list[float]] = {}
    for validation_country in x.index:
        x_train = x.drop(index=validation_country)
        y_train = y.drop(index=validation_country)
        x_validation = x.loc[validation_country]
        mean, std = fit_standardizer(x_train)
        x_train_scaled = standardize(x_train, mean, std)
        x_validation_scaled = standardize(x_validation, mean, std)
        errors = []
        for k in k_values:
            prediction, _, _ = knn_predict(
                x_train_scaled, y_train, x_validation_scaled, k
            )
            errors.append(abs(prediction - float(y.loc[validation_country])))
        error_rows[validation_country] = errors
    return pd.DataFrame(error_rows, index=k_values).T


def baseline_loo_mae(x: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    mean_errors = []
    linear_errors = []
    for validation_country in x.index:
        x_train = x.drop(index=validation_country)
        y_train = y.drop(index=validation_country)
        x_validation = x.loc[validation_country]

        mean_errors.append(abs(float(y_train.mean()) - float(y.loc[validation_country])))

        mean, std = fit_standardizer(x_train)
        x_train_scaled = standardize(x_train, mean, std)
        x_validation_scaled = standardize(x_validation, mean, std)
        design = np.column_stack(
            [np.ones(len(x_train_scaled)), x_train_scaled.to_numpy(dtype=float)]
        )
        coefficients = np.linalg.lstsq(
            design, y_train.to_numpy(dtype=float), rcond=None
        )[0]
        linear_prediction = float(
            np.r_[1.0, x_validation_scaled.to_numpy(dtype=float)] @ coefficients
        )
        linear_errors.append(abs(linear_prediction - float(y.loc[validation_country])))
    return float(np.mean(mean_errors)), float(np.mean(linear_errors))


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    data_2024 = raw.loc[raw["Year"] == 2024].set_index("Country")
    train = data_2024.drop(index="Japan")
    test = data_2024.loc[["Japan"]]
    x = train[FEATURES].astype(float)
    y = train[TARGET].astype(float)

    # 为 notebook 预先计算可复现输出。
    south_korea = "South Korea"
    one_feature_rows = []
    for feature in FEATURES:
        candidates = x.drop(index=south_korea)
        distances = (candidates[feature] - x.loc[south_korea, feature]).abs().sort_values()
        neighbor = distances.index[0]
        prediction = float(y.loc[neighbor])
        one_feature_rows.append(
            {
                "特征": feature,
                "最近邻国家": neighbor,
                "距离": float(distances.iloc[0]),
                "预测值": prediction,
                "实际值": float(y.loc[south_korea]),
                "绝对误差": abs(prediction - float(y.loc[south_korea])),
            }
        )
    one_feature_result = pd.DataFrame(one_feature_rows).set_index("特征").round(4)

    raw_distance = float(np.linalg.norm(x.loc["South Korea"] - x.loc["Italy"]))
    x_mean, x_std = fit_standardizer(x)
    x_scaled = standardize(x, x_mean, x_std)
    scaled_distance = float(
        np.linalg.norm(x_scaled.loc["South Korea"] - x_scaled.loc["Italy"])
    )

    error_table = loo_knn_errors(x, y)
    mean_errors = error_table.mean(axis=0)
    best_k = int(mean_errors.idxmin())
    knn_loo_mae = float(mean_errors.loc[best_k])
    mean_baseline_mae, linear_loo_mae = baseline_loo_mae(x, y)
    comparison = pd.DataFrame(
        {
            "模型": ["训练集均值基线", "线性回归基线", f"k-NN（k={best_k}）"],
            "留一交叉验证 MAE": [mean_baseline_mae, linear_loo_mae, knn_loo_mae],
        }
    ).set_index("模型").sort_values("留一交叉验证 MAE").round(4)

    final_mean, final_std = fit_standardizer(x)
    x_final_scaled = standardize(x, final_mean, final_std)
    japan_scaled = standardize(test.loc["Japan", FEATURES].astype(float), final_mean, final_std)
    japan_prediction, neighbor_positions, distances = knn_predict(
        x_final_scaled, y, japan_scaled, best_k
    )
    japan_actual = float(test.loc["Japan", TARGET])
    neighbors = pd.DataFrame(
        {
            "距离": distances[neighbor_positions],
            "生活满意度": y.iloc[neighbor_positions].to_numpy(),
        },
        index=y.index[neighbor_positions],
    ).round(4)
    neighbors.index.name = "国家"

    # 图 1：原始数据关系。坐标轴保留数据字段原名，避免混淆源数据定义。
    fig_eda, axes = plt.subplots(1, 2, figsize=(10, 4))
    train.plot.scatter(x=FEATURES[0], y=TARGET, ax=axes[0], alpha=0.75)
    train.plot.scatter(x=FEATURES[1], y=TARGET, ax=axes[1], alpha=0.75)
    fig_eda.tight_layout()

    # 图 2：k 与留一验证误差。
    fig_cv, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_errors.index, mean_errors.values, marker="o", markersize=3)
    ax.axvline(best_k, color="tab:red", linestyle="--", label=f"best k = {best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean absolute error")
    ax.legend()
    ax.grid(alpha=0.25)
    fig_cv.tight_layout()

    cells: list[dict] = []
    cells.append(markdown("""# 从零实现 k 近邻回归：国家生活满意度预测

本项目将三份课程 notebook 合并为一条完整、去重的分析流程。目标是使用**人均 GDP**和**就业率**预测国家的生活满意度，并通过留一交叉验证选择近邻数量 `k`。

## 项目亮点

- 仅使用 NumPy 和 pandas，从零实现 k-NN 回归。
- 展示单特征与多特征距离的区别。
- 在每个验证折中只使用训练数据计算标准化参数，避免数据泄漏。
- 使用留一交叉验证选择 `k`，并与均值及线性回归基线比较。
- 将日本完整保留为最终测试样本。

> 教学结构参考 `tomonari-masada/course2026-sml` 中的 k 近邻课程 notebook；本文件重新组织了结构、代码、注释与分析。数据来源和再发布许可仍需进一步确认。"""))

    cells.append(markdown("""## 1. 环境与数据准备

数据包含 2011—2024 年的国家生活满意度、人均 GDP 和就业率。本分析只使用 2024 年数据；日本作为最终测试样本，其余国家用于训练和交叉验证。"""))

    cells.append(code("""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 50)
pd.set_option("display.precision", 4)

# 同时兼容从仓库根目录和 notebooks/ 目录启动 Jupyter。
data_candidates = [
    Path("data/life_satisfaction_2010_2024.csv"),
    Path("../data/life_satisfaction_2010_2024.csv"),
]
data_path = next((path for path in data_candidates if path.exists()), None)
if data_path is None:
    raise FileNotFoundError("未找到 data/life_satisfaction_2010_2024.csv")

raw = pd.read_csv(data_path)
data_2024 = raw.loc[raw["Year"] == 2024].set_index("Country")

# 日本只用于最终测试，调参阶段不读取其目标值。
train = data_2024.drop(index="Japan")
test = data_2024.loc[["Japan"]]

FEATURES = ["GDP per capita", "Employment Rate (%)"]
TARGET = "Life Satisfaction"
X = train[FEATURES].astype(float)
y = train[TARGET].astype(float)

print(f"原始数据：{len(raw)} 行，年份范围 {raw['Year'].min()}—{raw['Year'].max()}")
print(f"2024 年训练国家：{len(train)}；测试国家：{test.index[0]}")""", 1, [stream_output(
        f"原始数据：{len(raw)} 行，年份范围 {raw['Year'].min()}—{raw['Year'].max()}\n"
        f"2024 年训练国家：{len(train)}；测试国家：Japan\n"
    )]))

    cells.append(markdown("""## 2. 探索性分析

下图分别观察人均 GDP、就业率与生活满意度之间的关系。人均 GDP 呈现更明显的正相关趋势；就业率与生活满意度的关系相对较弱。"""))
    cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
train.plot.scatter(x=FEATURES[0], y=TARGET, ax=axes[0], alpha=0.75)
train.plot.scatter(x=FEATURES[1], y=TARGET, ax=axes[1], alpha=0.75)
fig.tight_layout()
plt.show()""", 2, [figure_output(fig_eda)]))

    cells.append(markdown("""## 3. 单特征 k-NN：理解最近邻

首先只使用一个特征，并设置 `k=1`。预测韩国时，从其他国家中寻找该特征最接近的国家，再用该国的生活满意度作为预测值。

这个步骤对应原第一课的核心思想：**距离近的样本可能具有相似目标值**。"""))
    cells.append(code("""validation_country = "South Korea"
rows = []

for feature in FEATURES:
    candidates = X.drop(index=validation_country)
    distances = (
        candidates[feature] - X.loc[validation_country, feature]
    ).abs().sort_values()
    neighbor = distances.index[0]
    prediction = float(y.loc[neighbor])
    actual = float(y.loc[validation_country])
    rows.append({
        "特征": feature,
        "最近邻国家": neighbor,
        "距离": float(distances.iloc[0]),
        "预测值": prediction,
        "实际值": actual,
        "绝对误差": abs(prediction - actual),
    })

one_feature_result = pd.DataFrame(rows).set_index("特征").round(4)
one_feature_result""", 3, [dataframe_output(one_feature_result, 3)]))

    cells.append(markdown("""## 4. 多特征距离与标准化

直接计算二维欧氏距离会产生尺度问题：人均 GDP 的数值通常是数万，而就业率约为 0—100，因此未经处理时 GDP 几乎完全支配距离。

解决方法是在每个训练折中分别计算均值和标准差，再将特征变换为均值约为 0、标准差约为 1 的尺度。"""))
    cells.append(code("""raw_distance = np.linalg.norm(X.loc["South Korea"] - X.loc["Italy"])

X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std
scaled_distance = np.linalg.norm(
    X_scaled.loc["South Korea"] - X_scaled.loc["Italy"]
)

print(f"标准化前的韩国—意大利距离：{raw_distance:.2f}")
print(f"标准化后的韩国—意大利距离：{scaled_distance:.4f}")""", 4, [stream_output(
        f"标准化前的韩国—意大利距离：{raw_distance:.2f}\n"
        f"标准化后的韩国—意大利距离：{scaled_distance:.4f}\n"
    )]))

    cells.append(markdown("""## 5. 从零实现 k-NN 回归

下面的函数接收已经标准化的训练特征、目标值和一个查询样本：

1. 计算查询样本到所有训练样本的欧氏距离；
2. 找到距离最小的 `k` 个位置；
3. 返回这些近邻目标值的平均数。"""))
    cells.append(code("""def fit_standardizer(frame):
    # 只使用当前训练集估计标准化参数。
    return frame.mean(), frame.std()


def standardize(frame, mean, std):
    # 使用训练集参数变换训练、验证或测试数据。
    return (frame - mean) / std


def knn_predict(X_train_scaled, y_train, x_query_scaled, k):
    # 返回预测值、近邻位置以及所有训练样本的距离。
    distances = np.linalg.norm(
        X_train_scaled.to_numpy(dtype=float)
        - x_query_scaled.to_numpy(dtype=float),
        axis=1,
    )
    neighbor_positions = np.argsort(distances)[:k]
    prediction = float(y_train.iloc[neighbor_positions].mean())
    return prediction, neighbor_positions, distances""", 5))

    cells.append(markdown("""## 6. 留一交叉验证选择 k

数据量较小，因此依次把每个国家作为验证样本，其余国家作为训练集。每次都重新计算标准化参数，避免验证数据参与训练。

对每个 `k` 汇总所有国家的绝对误差，平均误差最低的 `k` 被选为最终超参数。"""))
    cells.append(code("""k_values = list(range(1, len(X)))
error_rows = {}

for validation_country in X.index:
    X_train = X.drop(index=validation_country)
    y_train = y.drop(index=validation_country)
    x_validation = X.loc[validation_country]

    # 关键：每个验证折只使用该折训练集计算 mean/std。
    mean, std = fit_standardizer(X_train)
    X_train_scaled = standardize(X_train, mean, std)
    x_validation_scaled = standardize(x_validation, mean, std)

    errors = []
    for k in k_values:
        prediction, _, _ = knn_predict(
            X_train_scaled, y_train, x_validation_scaled, k
        )
        errors.append(abs(prediction - float(y.loc[validation_country])))
    error_rows[validation_country] = errors

error_table = pd.DataFrame(error_rows, index=k_values).T
mean_errors = error_table.mean(axis=0)
best_k = int(mean_errors.idxmin())

print(f"最优 k：{best_k}")
print(f"对应的留一交叉验证 MAE：{mean_errors.loc[best_k]:.4f}")""", 6, [stream_output(
        f"最优 k：{best_k}\n对应的留一交叉验证 MAE：{knn_loo_mae:.4f}\n"
    )]))
    cells.append(code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mean_errors.index, mean_errors.values, marker="o", markersize=3)
ax.axvline(best_k, color="tab:red", linestyle="--", label=f"best k = {best_k}")
ax.set_xlabel("k")
ax.set_ylabel("Mean absolute error")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()""", 7, [figure_output(fig_cv)]))

    cells.append(markdown("""## 7. 与简单基线模型比较

模型必须优于简单规则才有实际意义。这里比较：

- **训练集均值基线**：所有样本都预测为当前训练集的平均生活满意度；
- **线性回归基线**：用 NumPy 最小二乘法拟合两个标准化特征；
- **k-NN**：使用交叉验证选出的 `k`。

线性回归的平均验证误差略低于 k-NN，说明当前数据总体关系可能更接近线性趋势。诚实展示这一结果比只报告 k-NN 的最好成绩更有分析价值。"""))
    cells.append(code("""mean_baseline_errors = []
linear_errors = []

for validation_country in X.index:
    X_train = X.drop(index=validation_country)
    y_train = y.drop(index=validation_country)
    x_validation = X.loc[validation_country]
    actual = float(y.loc[validation_country])

    mean_baseline_errors.append(abs(float(y_train.mean()) - actual))

    mean, std = fit_standardizer(X_train)
    X_train_scaled = standardize(X_train, mean, std)
    x_validation_scaled = standardize(x_validation, mean, std)
    design = np.column_stack([
        np.ones(len(X_train_scaled)),
        X_train_scaled.to_numpy(dtype=float),
    ])
    coefficients = np.linalg.lstsq(
        design, y_train.to_numpy(dtype=float), rcond=None
    )[0]
    linear_prediction = float(
        np.r_[1.0, x_validation_scaled.to_numpy(dtype=float)] @ coefficients
    )
    linear_errors.append(abs(linear_prediction - actual))

comparison = pd.DataFrame({
    "模型": ["训练集均值基线", "线性回归基线", f"k-NN（k={best_k}）"],
    "留一交叉验证 MAE": [
        np.mean(mean_baseline_errors),
        np.mean(linear_errors),
        mean_errors.loc[best_k],
    ],
}).set_index("模型").sort_values("留一交叉验证 MAE").round(4)
comparison""", 8, [dataframe_output(comparison, 8)]))

    cells.append(markdown("""## 8. 最终预测：日本

只有完成超参数选择后，才首次使用日本的特征进行最终预测。标准化参数由全部 41 个训练国家计算，日本使用同一组参数进行变换。"""))
    cells.append(code("""final_mean, final_std = fit_standardizer(X)
X_final_scaled = standardize(X, final_mean, final_std)
japan_scaled = standardize(
    test.loc["Japan", FEATURES].astype(float), final_mean, final_std
)

japan_prediction, neighbor_positions, distances = knn_predict(
    X_final_scaled, y, japan_scaled, best_k
)
japan_actual = float(test.loc["Japan", TARGET])

print(f"使用的 k：{best_k}")
print(f"日本生活满意度预测值：{japan_prediction:.3f}")
print(f"日本生活满意度实际值：{japan_actual:.3f}")
print(f"绝对误差：{abs(japan_prediction - japan_actual):.3f}")""", 9, [stream_output(
        f"使用的 k：{best_k}\n"
        f"日本生活满意度预测值：{japan_prediction:.3f}\n"
        f"日本生活满意度实际值：{japan_actual:.3f}\n"
        f"绝对误差：{abs(japan_prediction - japan_actual):.3f}\n"
    )]))
    cells.append(code("""neighbors = pd.DataFrame({
    "距离": distances[neighbor_positions],
    "生活满意度": y.iloc[neighbor_positions].to_numpy(),
}, index=y.index[neighbor_positions]).round(4)
neighbors.index.name = "国家"
neighbors""", 10, [dataframe_output(neighbors, 10)]))

    cells.append(markdown(f"""## 9. 结论与局限

### 主要结果

- 单特征最近邻容易受特征选择影响；GDP 的单特征表现优于就业率。
- 多特征距离必须先处理尺度差异。
- 留一交叉验证选择的最优参数为 **`k={best_k}`**，平均 MAE 为 **{knn_loo_mae:.4f}**。
- 日本的预测生活满意度为 **{japan_prediction:.3f}**，实际值为 **{japan_actual:.3f}**，绝对误差为 **{abs(japan_prediction - japan_actual):.3f}**。
- 线性回归的留一验证 MAE（**{linear_loo_mae:.4f}**）略优于 k-NN，说明模型选择不能只依赖单个测试样本。

### 局限

1. 2024 年只有 42 个国家，样本量很小。
2. 最终测试集只有日本一个样本，无法稳定估计泛化误差。
3. GDP 和就业率不能完整解释生活满意度，教育、健康、社会支持等变量可能同样重要。
4. 当前分析展示相关性预测，不代表因果关系。
5. 数据来源、更新流程和许可证需要进一步补充后才能安全再发布。

### 可继续改进

- 使用多年度面板数据并设置多个测试国家；
- 加入更多社会经济特征；
- 比较正则化线性模型、树模型等方法；
- 对预测不确定性和特征敏感性进行分析。"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
