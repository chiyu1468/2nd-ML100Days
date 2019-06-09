
#%% [markdown]
# ## [範例重點]
# 了解 sklearn 中，GridSearchCV 的使用方法與原理

#%%
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier

#%%
# 讀取波士頓房價資料集
boston = datasets.load_boston()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=42)

# 建立模型
clf = GradientBoostingRegressor(random_state=7)

#%%
# 先看看使用預設參數得到的結果，約為 8.379 的 MSE
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))

#%%
# 設定要訓練的超參數組合
n_estimators = [100, 200, 300]
max_depth = [1, 3, 5]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(clf, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)

# 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型


#%%
# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#%%
grid_result.best_params_

#%%
# 使用最佳參數重新建立模型
clf_bestparam = GradientBoostingRegressor(max_depth=grid_result.best_params_['max_depth'],
                                           n_estimators=grid_result.best_params_['n_estimators'])

# 訓練模型
clf_bestparam.fit(x_train, y_train)

# 預測測試集
y_pred = clf_bestparam.predict(x_test)

#%%
# 調整參數後約可降至 8.30 的 MSE
print(metrics.mean_squared_error(y_test, y_pred))

#%% [markdown]
# ### 作業
# 請使用不同的資料集，並使用 hyper-parameter search 的方式，看能不能找出最佳的超參數組合

#%%
breast_cancer = datasets.load_breast_cancer()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(
    breast_cancer.data, breast_cancer.target, test_size=0.25, random_state=4)

#%%
# 先看看使用預設參數得到的結果
clf1 = RandomForestClassifier(n_estimators=30, max_depth=6, bootstrap=True)
clf1.fit(x_train, y_train)
y_pred = clf1.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

#%%
# 設定要訓練的超參數組合
n_estimators = [20, 30, 40, 50]
max_depth = [3, 4, 5, 6]
bootstrap=[True, False]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(clf1, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)

#%%
# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#%%
# 使用最佳參數重新建立模型
clf_hat = RandomForestClassifier(n_estimators=grid_result.best_params_['n_estimators'], 
                                    max_depth=grid_result.best_params_['max_depth'], 
                                    bootstrap=grid_result.best_params_['bootstrap'])
clf_hat.fit(x_train, y_train)
y_pred = clf_hat.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

