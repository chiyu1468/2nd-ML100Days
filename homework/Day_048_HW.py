

#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%%
train_data = pd.read_csv('./data/london/train.csv',header = None)
train_label = pd.read_csv('./data/london/trainLabels.csv',header = None)
test_data = pd.read_csv('./data/london/test.csv',header = None)

x_train, x_valid, y_train, y_valid = train_test_split(
    train_data, train_label, test_size=0.25, random_state=4)

#%%
clf = GradientBoostingClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_valid)

acc = metrics.accuracy_score(y_valid, y_pred)
print("Acuuracy: ", acc)

#%%
y_pred = clf.predict(test_data)
# np.savetxt("./data/london/Day_48.csv", y_pred) #, delimiter=",")
result = pd.DataFrame(y_pred)
result.columns = ['Solution']
result.index = result.index + 1
result.to_csv("./data/london/Day_48.csv", index=True, index_label='Id')
