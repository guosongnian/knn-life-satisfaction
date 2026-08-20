# 从零实现 k 近邻回归：国家生活满意度预测

这是一个完整的数据科学分析项目。项目使用 NumPy 和 pandas 从零实现 **k 近邻（k-NN）回归**，根据国家的人均 GDP 和就业率预测生活满意度。

**[查看完整分析 notebook](notebooks/knn_life_satisfaction.ipynb)**

## 项目亮点

- 从零实现距离计算、近邻选择和回归预测，不依赖 scikit-learn 的 k-NN 实现。
- 从单特征最近邻逐步扩展到双特征模型。
- 在每个验证折中只使用训练数据计算标准化参数，避免数据泄漏。
- 使用留一交叉验证选择最佳近邻数量 `k`。
- 与训练集均值和线性回归基线进行比较。
- 将日本完整保留为最终测试样本。
- notebook 已保存图表和关键运行结果，可直接在 GitHub 中预览。
- 提供可直接运行的 [`knn_life_satisfaction.py`](knn_life_satisfaction.py)，Notebook 与脚本共用相同方法逻辑。

## 项目结构

```text
.
├── data/
│   └── life_satisfaction_2011_2024.csv
├── notebooks/
│   └── knn_life_satisfaction.ipynb
├── tests/
│   └── test_smoke.py
├── .github/workflows/ci.yml
├── knn_life_satisfaction.py
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

线性回归在整体留一验证中的平均误差低于 k-NN，因此如果目标是从候选模型中选择整体表现更好的模型，当前结果支持线性回归。日本部分保留 k-NN 预测，是为了完整展示 k-NN 工作流程；不能利用日本单个样本的结果反向选择模型。

## 运行方法

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
python -m pip install -r requirements.txt
```

启动 Jupyter：

```bash
jupyter lab
```

然后打开：

```text
notebooks/knn_life_satisfaction.ipynb
```

也可以直接运行完整 Python 版本：

```bash
python knn_life_satisfaction.py
```

notebook 同时支持从仓库根目录或 `notebooks/` 目录启动 Jupyter。

数据结构可以通过不依赖 Jupyter 的快速测试验证：

```bash
python -m unittest discover -s tests -v
```

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

1. 在 notebook 中检查字段、缺失值、数值类型和重复国家，再筛选 2024 年数据。
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
