"""校验仓库内数据文件的基本结构。"""

import csv
import unittest
from pathlib import Path

import pandas as pd

from knn_life_satisfaction import fit_standardizer, knn_predict, standardize


class KnnDataSmokeTest(unittest.TestCase):
    def test_dataset_schema(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "life_satisfaction_2011_2024.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        required = {
            "Country",
            "Code",
            "Year",
            "Life Satisfaction",
            "GDP per capita",
            "Employment Rate (%)",
        }
        self.assertTrue(required.issubset(reader.fieldnames or []))
        self.assertGreater(len(rows), 500)

    def test_knn_prediction(self) -> None:
        x = pd.DataFrame({"a": [1.0, 2.0, 5.0], "b": [1.0, 2.0, 5.0]})
        y = pd.Series([1.0, 2.0, 5.0])
        mean, std = fit_standardizer(x)
        scaled = standardize(x, mean, std)
        query = standardize(pd.Series({"a": 1.1, "b": 1.1}), mean, std)
        prediction, positions, _ = knn_predict(scaled, y, query, k=1)
        self.assertEqual(prediction, 1.0)
        self.assertEqual(positions.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
