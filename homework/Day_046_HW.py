

#%%
from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

#%%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

#%%
acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)
print("Features: ", iris.feature_names)
print("Feature importance: ", clf.feature_importances_)

#%% [markdown]
# ### 作業
# 目前已經學過許多的模型，相信大家對整體流程應該比較掌握了，這次作業請改用**手寫辨識資料集**，步驟流程都是一樣的，請試著自己撰寫程式碼來完成所有步驟

#%%
digits = datasets.load_digits()
# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=4)
# 建立模型
clf = GradientBoostingClassifier()
# 訓練模型
clf.fit(x_train, y_train)
# 預測測試集
y_pred = clf.predict(x_test)

#%%
acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)

