#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass

#%%
import os
import numpy as np
import pandas as pd


#%%
# 設定 data_path
dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

#%% [markdown]
# 檢視資料中各個欄位類型的數量

#%%
app_train.dtypes.value_counts()

#%% [markdown]
# 檢視資料中類別型欄位各自類別的數量
# 欄位名稱              該 column 中包含的種類

#%%
app_train.select_dtypes(include=["int64"]).apply(pd.Series.nunique, axis = 0)

#%% [markdown]
# #### Label encoding
# 有仔細閱讀[參考資料](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)的人可以發現，Label encoding 的表示方式會讓同一個欄位底下的類別之間有大小關係 (0<1<2<...)，所以在這裡我們只對有類別數量小於等於 2 的類別型欄位示範使用 Label encoding，但不表示這樣處理是最好的，一切取決於欄位本身的意義適合哪一種表示方法

#%%
from sklearn.preprocessing import LabelEncoder


#%%
# Create a label encoder object
# 完善這個範例 
# (1.可以處理 YES NO 與 空值 的狀況。 2. 記錄原label與數字對應關係)
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # get the category array (ndarray)
        cat_arr = app_train[col].unique()
        # 對於有空值的欄位執行 fit 會造成 error
        # 使用 mask 把該 row 遮掉
        mask_train = ~app_train[col].isnull()
        mask_test = ~app_test[col].isnull()
        # If 2 or fewer unique categories
        if len(list(cat_arr)) <= 3:
            # Train on the training data
            le.fit(app_train[col][mask_train])

            # Transform both training and testing data
            # app_train[col][mask_train] = le.transform(app_train[col][mask_train]) this has warning
            app_train.loc[mask_train,col] = le.transform(app_train[col][mask_train])
            app_test.loc[mask_test,col] = le.transform(app_test[col][mask_test])

            cmap = {}
            for i in range(np.shape(le.classes_)[0]):
                cmap[le.classes_[i]] = i
            print("for item {} : category mapping is {} ".format(col, cmap))

            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

#%% [markdown]
# #### One Hot encoding
# pandas 中的 one hot encoding 非常方便，一行程式碼就搞定

#%%
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())

#%% [markdown]
# 可以觀察到原來的類別型欄位都轉為 0/1 了
#%% [markdown]
# ## 作業
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化

#%%
app_train = pd.read_csv(f_app_train)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


#%%
"""
Your Code Here
"""
one_hot_sub_train = pd.get_dummies(sub_train)
print(one_hot_sub_train.shape)
one_hot_sub_train.head()

#%%
# 如果不要用 get_dummies 這麼 dummy 的方法，可以用下面這樣
days = []
for day in sub_train.values:
    if day[0] not in days:
        days.append(day[0])
days

#%%
for day in days:
    sub_train[day] = sub_train['WEEKDAY_APPR_PROCESS_START'].apply(lambda x: 1 if x == day else 0)
sub_train.head()


