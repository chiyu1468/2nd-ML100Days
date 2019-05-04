#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'homework'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 作業 : (Kaggle)鐵達尼生存預測
#%% [markdown]
# # [作業目標]
# - 試著完成三種不同特徵類型的三種資料操作, 觀察結果
# - 思考一下, 這三種特徵類型, 哪一種應該最複雜/最難處理
#%% [markdown]
# # [作業重點]
# - 完成剩餘的八種 類型 x 操作組合 (In[6]~In[13], Out[6]~Out[13])
# - 思考何種特徵類型, 應該最複雜

#%%
# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape


#%%
# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
# df.head()


#%%
# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df


#%%
#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
# print(f'{len(int_features)} Integer Features : {int_features}\n')
# print(f'{len(float_features)} Float Features : {float_features}\n')
# print(f'{len(object_features)} Object Features : {object_features}')
print("{} Integer Features : {}".format(len(int_features), int_features))
print("{} Integer Features : {}".format(len(float_features), float_features))
print("{} Integer Features : {}".format(len(object_features), object_features))

#%% [markdown]
# # 作業1 
# * 試著執行作業程式，觀察三種類型 (int / float / object) 的欄位分別進行( 平均 mean / 最大值 Max / 相異值 nunique )  
# 中的九次操作會有那些問題? 並試著解釋那些發生Error的程式區塊的原因?  
# <br />
# # 作業2
# * 思考一下，試著舉出今天五種類型以外的一種或多種資料類型，你舉出的新類型是否可以歸在三大類中的某些大類?  
# 所以三大類特徵中，哪一大類處理起來應該最複雜?

#%%
# 請依序列出 三種特徵類型 (int / float / object) x 三種方法 (平均 mean / 最大值 Max / 相異值 nunique) 的其餘操作
mylist = ["int_features", "float_features", "object_features"]
for one_feat in mylist:
    print("this is ", one_feat)
    print(df[eval(one_feat)].mean())
    print(df[eval(one_feat)].max())
    print(df[eval(one_feat)].nunique())
    print("===========")

#%% [markdown]
# # 作業1 作答
#   由上述操作可看出三種特徵類型都可以執行平均、最大值、相異值，
#   但是 object type 的平均值是算不出來的，而求得最大值的意義不明確。<br />
#   所以雖然不會造成 Error 但是得到的結果沒有參考價值
#
# # 作業2 作答
#   特徵工程主要作用在於將資料點轉化成為可解釋的量化指標，
#   通常 object type 的量化特徵不如數字形態的資料明顯，
#   也需要額外的 domain knowledge 從旁協助，故處理上相對不直觀。<br />
#   <br />
#   但數字形態的資料不一定好處理，例如:<br />
#   音頻-連續訊號:會有sample rate 或 resample以及濾雜訊的層面要考量<br />
#   圖片-大型多層矩陣:會有色偏曝光、影像旋轉偏移等問題需要考量<br />
#   故個人認為並沒有最複雜的類別，都需要依照想要達成的目標來定義。





