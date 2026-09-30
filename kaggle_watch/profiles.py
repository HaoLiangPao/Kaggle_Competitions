"""Curated competition facts verified on Kaggle; update dates and sources when revised."""

PROFILES = {
    "titanic": {
        "verified_on": "2026-09-27",
        "title": "Titanic 生存预测",
        "task": "预测乘客是否生还（二分类）",
        "metric": "准确率",
        "difficulty": 1,
        "fit": 5,
        "train_test_bytes": 61194 + 28629,
        "reason": "官方入门赛、数据很小，适合从最简单的预测与验证开始。",
        "source": "https://www.kaggle.com/competitions/titanic/overview/evaluation",
    },
    "house-prices-advanced-regression-techniques": {
        "verified_on": "2026-09-27",
        "title": "House Prices 房价预测",
        "task": "预测房屋售价（回归）",
        "metric": "对数价格的 RMSE",
        "difficulty": 3,
        "fit": 4,
        "train_test_bytes": 460676 + 451405,
        "reason": "可延续现有回归代码，但 79 个特征会增加数据处理难度。",
        "source": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/description",
    },
    "playground-series-s6e9": {
        "verified_on": "2026-09-27",
        "title": "Playground S6E9 电动车购买意愿",
        "task": "预测购买电动车的概率（二分类）",
        "metric": "ROC AUC",
        "difficulty": 2,
        "fit": 4,
        "train_test_bytes": 44707646 + 18298347,
        "reason": "题目适合练习表格分类，但截止临近，建议作为下一次月赛的预演。",
        "source": "https://www.kaggle.com/competitions/playground-series-s6e9/overview",
    },
}
