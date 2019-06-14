
['User_id', 'Merchant_id', 'Coupon_id', 
'Discount_rate', 'Distance','Date_received', 
'Date', 'discount_rate', 'discount_man',
'discount_jian', 'discount_type']

dfoff.groupby("Discount_rate").size()
b = dfoff[dfoff["Discount_rate"] == '0.7']

# 研究一下有用優惠卷的消費模式
# 在有用優惠卷的狀況下，分析取得優惠卷與消費日期差距
b=dfoff[~dfoff['Date_received'].isna()] # 篩選掉沒有卷的
b=b[~b['Date'].isna()] # 再篩選掉沒有用卷的
b['interval']=b['Date']-b['Date_received']
b.groupby('interval').size() # print

# 長方條圖
fig, ax = plt.subplots()
b['interval'].hist(ax=ax, bins=200)
ax.set_yscale('log')
# 看單筆數據
b[b['interval']==28.0].shape

# 依照商家分類優惠卷
d = b.groupby('Merchant_id').size().sort_values(ascending=False) #print
d = d[d.values > 300]
d = d.keys()

b['Merchant_id'] = b['Merchant_id'].apply(lambda x: x if x in d else 0 )

c=b[b['Merchant_id'] == 2099]
c['interval'].hist()





#%%
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
import numpy as np
import pandas as pd
from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler


#%%
dfoff = pd.read_csv("./data/ml100marathon/train_offline.csv")
dftest = pd.read_csv("./data/ml100marathon/test_offline.csv")
dftest = dftest[~dftest.Coupon_id.isna()]
dftest.reset_index(drop=True, inplace=True)
print("Training Dataset: ",dfoff.shape)
print("Testing Dataset: ", dftest.shape)
# dfoff.head()

# #%%
# example answer 
# dfsample = pd.read_csv("./data/ml100marathon/sample_submission.csv")

#%%
"""
According to the definition, 
1) buy with coupon within (include) 15 days ==> 1
2) buy with coupon but out of 15 days ==> 0
3) buy without coupon ==> -1 (we don't care)
"""
# def label(row):
#     if np.isnan(row['Date_received']):
#         return -1
#     if not np.isnan(row['Date']):
#         td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
#         if td <= pd.Timedelta(15, 'D'):
#             return 1
#     return 0

# dfoff["label"] = dfoff.apply(label, axis=1)
# dfoff["label"].value_counts()

dfoff = dfoff[~dfoff['Date_received'].isna()] # 篩選掉沒有卷的
dfoff_used = dfoff[~dfoff['Date'].isna()] # 再篩選掉沒有用卷的
# 依照主要商家做 one-hot encode (主要商家目前訂為 有卷有消費數量達到 300)
main_merchant = dfoff_used.groupby('Merchant_id').size().sort_values(ascending=False) #print
main_merchant = main_merchant[main_merchant.values > 300].keys()
print(main_merchant)
dfoff_used['Merchant_id'] = dfoff_used['Merchant_id'].apply(lambda x: x if x in main_merchant else 0 )


#%%
# Generate features - weekday acquired coupon
def getWeekday(row):
    if (np.isnan(row)) or (row==-1):
        return row
    else:
        return pd.to_datetime(row, format = "%Y%m%d").dayofweek+1 # add one to make it from 0~6 -> 1~7

dfoff['weekday'] = dfoff['Date_received'].apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].apply(getWeekday)

# weekend  週末為 1
dfoff['weekend'] = dfoff['weekday'].astype('str').apply(lambda x : 1 if x in [6,7] else 0 ) # apply to trainset
dftest['weekend'] = dftest['weekday'].astype('str').apply(lambda x : 1 if x in [6,7] else 0 ) # apply to testset

weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
print(weekdaycols)

tmpdf = pd.get_dummies(dfoff['weekday'].replace(-1, np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace(-1, np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


# TODO
# 是否加入特殊節日 例如情人節 父親節

#%%
# Generate features - coupon discount and distance
def getDiscountType(row):
    if row == 'nan':
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    # """Convert discount to rate"""
    if row == 'nan':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return np.nan

def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return np.nan

def processData(df):
    
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].astype('str').apply(convertRate)
    df['discount_man'] = df['Discount_rate'].astype('str').apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].astype('str').apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].astype('str').apply(getDiscountType)
    
    # convert distance
    df.loc[df.Distance.isna(), "Distance"] = 99
    return df

#%%
dfoff = processData(dfoff)
dftest = processData(dftest)


#%%
## Naive model
def split_train_valid(row, date_cut="20160416"):
    is_train = True if pd.to_datetime(row, format="%Y%m%d") < pd.to_datetime(date_cut, format="%Y%m%d") else False
    return is_train
    
df = dfoff[dfoff['label'] != -1].copy()
df["is_train"] = df["Date_received"].apply(split_train_valid)
train = df[df["is_train"]]
valid = df[~df["is_train"]]
train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
print("Train size: {}, #positive: {}".format(len(train), train["label"].sum()))
print("Valid size: {}, #positive: {}".format(len(valid), valid["label"].sum()))

original_feature = ['discount_rate',
                    'discount_type',
                    'discount_man', 
                    'discount_jian',
                    'Distance', 
                    'weekday', 
                    'weekday_type'] + weekdaycols
print(len(original_feature),original_feature)

#%%
predictors = original_feature
print(predictors)

def check_model(data, predictors):
    
    classifier = lambda: SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        max_iter=100, 
        shuffle=True, 
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [ 0.001, 0.01, 0.1],
        'en__l1_ratio': [ 0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)
    
    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=folder, 
        n_jobs=-1, 
        verbose=1)
    grid_search = grid_search.fit(data[predictors], 
                                  data['label'])
    
    return grid_search

model = check_model(train, predictors)

#%%
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]


#%%
from sklearn.metrics import roc_auc_score, accuracy_score
auc_score = roc_auc_score(y_true=valid.label, y_score=y_valid_pred[:,1])
acc = accuracy_score(y_true=valid.label, y_pred=y_valid_pred.argmax(axis=1))
print("Validation AUC: {:.3f}, Accuracy: {:.3f}".format(auc_score, acc))

targetset = dftest.copy()
print(targetset.shape)
targetset = targetset[~targetset.Coupon_id.isna()]
targetset.reset_index(drop=True, inplace=True)
testset = targetset[predictors].copy()

y_test_pred = model.predict_proba(testset[predictors])
test1 = testset.copy()
test1['pred_prob'] = y_test_pred[:, 1]
print(test1.shape)

output = pd.concat((targetset[["User_id", "Coupon_id", "Date_received"]], test1["pred_prob"]), axis=1)
print(output.shape)

output.loc[:, "User_id"] = output["User_id"].apply(lambda x:str(int(x)))
output.loc[:, "Coupon_id"] = output["Coupon_id"].apply(lambda x:str(int(x)))
output.loc[:, "Date_received"] = output["Date_received"].apply(lambda x:str(int(x)))
output["uid"] = output[["User_id", "Coupon_id", "Date_received"]].apply(lambda x: '_'.join(x.values), axis=1)
output.reset_index(drop=True, inplace=True)

### NOTE: YOUR SUBMITION FILE SHOULD HAVE COLUMN NAME: uid, label
out = output.groupby("uid", as_index=False).mean()
out = out[["uid", "pred_prob"]]
out.columns = ["uid", "label"]
# out.to_csv("baseline_example.csv", header=["uid", "label"], index=False) # submission format
out.head()

