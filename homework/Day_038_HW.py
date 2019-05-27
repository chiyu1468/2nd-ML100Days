

#%% [markdown]
# ## [作業重點]
# 使用 Sklearn 中的線性迴歸模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，也請了解模型中各項參數的意義
#%% [markdown]
# ## 作業
# 試著使用 sklearn datasets 的其他資料集 (wine, boston, ...)，來訓練自己的線性迴歸模型。
# > https://scikit-learn.org/stable/datasets/index.html

#%% [markdown]
# ### HINT: 注意 label 的型態，確定資料集的目標是分類還是回歸，在使用正確的模型訓練！

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

#%%
# 先把資料的欄位抓出來看看
print(f'features number : {len(db.feature_names)} \nname : {db.feature_names} \n')
print(f'target type : {db.target.dtype}')

#%% [markdown]
# > 藉由上面的觀察，可以發現這個資料的 target 為一連續實數
# > 故試著用 LinearRegression 來預測這個資料集

#%%
# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.1, random_state=4)

# 建立一個線性回歸模型
regr = linear_model.LinearRegression()

# 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)

#%%
# 可以看回歸模型的參數值
# print('Coefficients: ', regr.coef_)

str1 = 'predict value = '
for i in range(len(regr.coef_)):
    str1 += f'({regr.coef_[i]:.04f})*'
    str1 += f'{db.feature_names[i]} +'

print("Linear Regression Formula :")
print(str1[:-2])

# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))