import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 生成示例数据
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, (100, 1))

# 多项式回归
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# 预测
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.6, label='数据点')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='多项式拟合')
plt.xlabel('X')
plt.ylabel('y')
plt.title('多项式回归 (degree=2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
