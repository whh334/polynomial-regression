# 1. 导入 `seaborn` 包
import seaborn as sns
sns.set_theme(style="darkgrid")

# 2. 加载 tips 数据集
tips = sns.load_dataset("tips")
tips.head()

# 3. 绘制统计图
sns.jointplot(
    x="total_bill",
    y="tip",
    data=tips,
    kind="reg",
    truncate=False,
    xlim=(0, 60),
    ylim=(0, 12),
    color="m",
    height=7,
)
