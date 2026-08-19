# 从零实现 k 近邻回归：国家生活满意度预测

这是一个完整的数据科学分析项目。项目使用 NumPy 和 pandas 从零实现 **k 近邻（k-NN）回归**，根据国家的人均 GDP 和就业率预测生活满意度。

原有三份课程 notebook 已合并为一份完整中文分析，删除了重复的数据读取、变量准备和交叉验证代码，并补充了中文注释、基线比较、结果解释与项目局限。

## 项目亮点

- 从零实现距离计算、近邻选择和回归预测，不依赖 scikit-learn 的 k-NN 实现。
- 从单特征最近邻逐步扩展到双特征模型。
- 在每个验证折中只使用训练数据计算标准化参数，避免数据泄漏。
- 使用留一交叉验证选择最佳近邻数量 `k`。
- 与训练集均值和线性回归基线进行比较。
- 将日本完整保留为最终测试样本。
- notebook 已保存图表和关键运行结果，可直接在 GitHub 中预览。

## 项目结构

```text
.
├── data/
│   └── life_satisfaction_2010_2024.csv
├── notebooks/
│   └── knn_life_satisfaction_zh.ipynb
├── src/
│   └── validate_data.py
├── .gitignore
├── NOTICE.md
├── README.md
└── requirements.txt
```

## 主要结果

| 模型 | 留一交叉验证 MAE |
|---|---:|
| 线性回归基线 | 0.3498 |
| k-NN（`k=14`） | 0.3659 |
| 训练集均值基线 | 0.4890 |

k-NN 对日本的最终预测结果：

| 指标 | 数值 |
|---|---:|
| 最优 `k` | 14 |
| 预测生活满意度 | 6.656 |
| 实际生活满意度 | 6.147 |
| 绝对误差 | 0.509 |

线性回归在整体留一验证中的平均误差略低于 k-NN，但 k-NN 对日本这一单独测试样本的误差更小。这说明不能只根据一个测试样本判断模型优劣。

## 运行方法

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
python -m pip install -r requirements.txt
```

验证数据集：

```bash
python src/validate_data.py
```

启动 Jupyter：

```bash
jupyter lab
```

然后打开：

```text
notebooks/knn_life_satisfaction_zh.ipynb
```

notebook 同时支持从仓库根目录或 `notebooks/` 目录启动 Jupyter。

## 数据字段

数据文件包含 555 行记录，年份覆盖 2011—2024 年。本项目筛选出 2024 年的 42 个国家，其中 41 个用于训练和验证，日本用于最终测试。

| 字段 | 含义 |
|---|---|
| `Country` | 国家名称 |
| `Code` | 三位国家代码 |
| `Year` | 年份 |
| `Life Satisfaction` | 生活满意度，预测目标 |
| `GDP per capita` | 人均 GDP，预测特征 |
| `Employment Rate (%)` | 就业率，预测特征 |

数据字段名保留原始英文名称，代码和分析说明均使用中文。

## 分析方法

1. 检查数据完整性并筛选 2024 年数据。
2. 将日本从建模数据中完全分离。
3. 使用韩国演示单特征最近邻预测。
4. 说明多特征距离中的尺度问题并进行标准化。
5. 从零实现 k-NN 回归。
6. 使用留一交叉验证选择最优 `k`。
7. 与均值和线性回归基线比较。
8. 使用最终模型预测日本生活满意度。
9. 分析结果、模型局限和改进方向。

## 项目局限

- 2024 年仅包含 42 个国家，样本量较小。
- 最终测试集只有日本一个样本，不能稳定估计泛化误差。
- GDP 和就业率不能完整解释生活满意度。
- 预测关系不代表因果关系。
- 数据来源、更新流程和再发布许可证仍需进一步确认。

## 来源与说明

项目教学结构参考了 [`tomonari-masada/course2026-sml`](https://github.com/tomonari-masada/course2026-sml) 中的 k 近邻课程 notebook。本仓库对三份 notebook 进行了合并、重构和中文化。

上游课程材料及数据没有检索到明确的再发布许可证。在获得权利人许可或确认许可证之前，本仓库仅用于个人学习与技术交流，不应作为商业产品再分发。详情见 [NOTICE.md](NOTICE.md)。
