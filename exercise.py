
# coding: utf-8

# In[37]:


# imports
get_ipython().magic(u'matplotlib notebook')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cleantitanic2

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC

# import jtplot submodule from jupyterthemes
from jupyterthemes import jtplot
jtplot.style()


# In[38]:


# init picture params
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

tr_raw = pd.read_csv('./data/train.csv')
te_raw = pd.read_csv('./data/test.csv')
type_dict = None

print tr_raw.info()


# In[39]:


# init model
result_dict = {}
bingo = ['', None, 0, None]
model_dict = {
    'logistic regr': LogisticRegression(),
    'decision tree': DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5),
    'svm rbf   ': SVC()
    #'random forest': RandomForestClassifier(n_estimators=1000,random_state=33)
}

# bagging
#for k, v in model_dict.items():
#    model_dict[k] = BaggingRegressor(v, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

tr_raw.describe()


# In[40]:


tr_raw.head()


# In[41]:


# feature engineering
# https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
if not type_dict:
    [tr_clean, te_clean, type_dict] = cleantitanic2.clean(tr_raw, te_raw)
    
tr_clean


# In[47]:


# label encoder
[tr_clean, te_clean] = map(lambda x:x.apply(LabelEncoder().fit_transform), [tr_clean, te_clean])
[tr_plt, te_plt] = map(lambda x:x.apply(LabelEncoder().fit_transform), [tr_raw, te_raw])

fig=plt.figure('Taitanic', figsize=(8, 4))
tr_plt_female = tr_plt[tr_plt.Sex==1]
tr_plt_y_female = tr_plt_female[['Survived']]
tr_plt_male = tr_plt[tr_plt.Sex==0]
tr_plt_y_male = tr_plt_male[['Survived']]

# real data, female, pclass, age
ax = fig.add_subplot(121, projection='3d')
ax.set_title('female')
ax.scatter(tr_plt_female['Family_Size'], tr_plt_female['Age'], tr_plt_female['Pclass'], c=tr_plt_y_female.values.ravel(), edgecolors='k', cmap=plt.cm.Paired)
ax.set_zticklabels(['1st', '2nd', '3rd'])
ax.set_xlabel('family size')
ax.set_ylabel('age(years)')

# real data, male, pclass, age
bx = fig.add_subplot(122, projection='3d')
bx.set_title('male')
bx.scatter(tr_plt_male['Family_Size'], tr_plt_male['Age'], tr_plt_male['Pclass'], c=tr_plt_y_male.values.ravel(), edgecolors='k', cmap=plt.cm.Paired)
bx.set_zticklabels(['1st', '2nd', '3rd'])
bx.set_xlabel('family size')
bx.set_ylabel('age(years)')

# feature chose
[tr_X, te_X] = map(lambda x:x[['Pclass', 'Sex', 'Age', 'Family_Size', 'Title']], [tr_clean, te_clean])


# In[43]:


# train
tr_y = tr_clean[['Survived']]
for k, v in model_dict.items():
    v.fit(tr_X, tr_y.values.ravel())


# In[44]:


# metrics
fig=plt.figure('Roc', figsize=(3, 3))
cx = fig.add_subplot(111)
cx.set_title('roc')

for k, v in model_dict.items():
    pred_y_cv = cross_val_predict(v, tr_X, tr_y, cv=10)
    acu = metrics.accuracy_score(tr_y, pred_y_cv.astype(np.int32))
    auc = metrics.roc_auc_score(tr_y, pred_y_cv.astype(np.int32))
    if acu > bingo[2]:
        bingo[0] = k
        bingo[1] = v
        bingo[2] = acu
        bingo[3] = pred_y_cv
    print k, "\tACU:", acu, "AUC:", auc,         "MSE:", metrics.mean_squared_error(tr_y, pred_y_cv), "RMSE:", np.sqrt(metrics.mean_squared_error(tr_y, pred_y_cv))
    
    # paint ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(tr_y, pred_y_cv)
    cx.plot(fpr,tpr)

# bingo
print "BINGO:", bingo[0], "ACU:", bingo[2]
#pd.DataFrame([bingo[1].coef_], columns=list(tr_X))


# In[45]:


# predict
pred_y = bingo[1].predict(te_X)

# write csv
id = te_raw[['PassengerId']]
frames = [id, pd.DataFrame(pred_y.astype(np.int32), columns=['Survived'])]
out_y = pd.concat(frames, axis=1, join_axes=[id.index])
out_y.to_csv('out.csv', index=False)

