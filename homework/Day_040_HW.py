
#%% [markdown]
# ## [作業重點]
# 使用 Sklearn 中的 Lasso, Ridge 模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，
# 也請了解模型中各項參數的意義。

# 機器學習的模型非常多種，但要訓練的資料多半有固定的格式，確保你了解訓練資料的格式為何，
# 這樣在應用新模型時，就能夠最快的上手開始訓練！

#%% [markdown]
# ## 練習時間
# 試著使用 sklearn datasets 的其他資料集 (boston, ...)，
# 來訓練自己的線性迴歸模型，並加上適當的正則話來觀察訓練情形。

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# wine = datasets.load_wine()
boston = datasets.load_boston()
# breast_cancer = datasets.load_breast_cancer()
db = boston

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.1, random_state=4)

#%% [markdown]
# ## Linear Regression

#%%
# 建立一個線性回歸模型
regr = linear_model.LinearRegression()
# 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)
# 訓練好的參數
regr.coef_

#%%
# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

#%% [markdown]
# ## LASSO Regression

#%%
# 建立一個線性回歸模型
lasso = linear_model.Lasso(alpha=1.0)
# 將訓練資料丟進去模型訓練
lasso.fit(x_train, y_train)
# 訓練好的參數
lasso.coef_

#%%
y_pred = lasso.predict(x_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

#%% [markdown]
# ## Ridge Regression

#%%
# 建立一個線性回歸模型
ridge = linear_model.Ridge(alpha=1.0)
# 將訓練資料丟進去模型訓練
ridge.fit(x_train, y_train)
# 訓練好的參數
ridge.coef_

#%%
y_pred = ridge.predict(x_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

