
#%%
import warnings
warnings.filterwarnings('ignore')


#%% [markdown]
# ## [範例重點]
# 了解隨機森林的建模方法及其中超參數的意義

#%%
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#%%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=20, max_depth=4)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

#%%
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
print("Features: ", iris.feature_names)
print("Feature importance: ", clf.feature_importances_)

#%% [markdown]
# ## 作業
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較

#%%
# wine = datasets.load_wine()
# boston = datasets.load_boston()
breast_cancer = datasets.load_breast_cancer()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(
    breast_cancer.data, breast_cancer.target, test_size=0.25, random_state=4)

#%%
# 建立模型
clf1 = RandomForestClassifier(n_estimators=30, max_depth=6, bootstrap=True)

# 訓練模型
clf1.fit(x_train, y_train)

# 預測測試集
y_pred = clf1.predict(x_test)

#%%
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
print("Features: ", breast_cancer.feature_names)
print("Feature importance: ", clf1.feature_importances_)





