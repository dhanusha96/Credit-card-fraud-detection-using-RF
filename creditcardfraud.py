import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,average_precision_score,cohen_kappa_score,roc_auc_score, precision_score,recall_score,f1_score,matthews_corrcoef,make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import StackingClassifier
import scipy as sp
import warnings

pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Import tools needed for visualization
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot

import pickle

#import dataset
data = pd.read_csv("creditcard.csv")
print(data)

print(data.columns)

print(data.shape)

print(data.describe())

#distribution of amount
amount = [data['Amount'].values]
sns.distplot(amount)
plt.show()

# distribution of Time
time = data['Time'].values
sns.distplot(time)
plt.show()

# distribution of anomalous features
features = data.iloc[:,0:28].columns

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, c in enumerate(data[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[c][data.Class == 1], bins=50)
    sns.distplot(data[c][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(c))
plt.show()

# Plot histograms of each parameter
data.hist(figsize = (20, 20))
plt.show()

# Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print("Amount details of fradulent transacation")
print(Fraud.Amount.describe())

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


#spliting dataset
X = data.drop(labels='Class', axis=1) # Features
y = data.loc[:,'Class']               # Response
del data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
del X, y

print(X_train.shape)

print(X_test.shape)

X_train.is_copy = False
X_test.is_copy = False


X_train['Time'].describe()

X_train.loc[:,'Time'] = X_train.Time / 3600
X_test.loc[:,'Time'] = X_test.Time / 3600

print(X_train['Time'].max() / 24)

plt.figure(figsize=(12,4), dpi=80)
sns.distplot(X_train['Time'], bins=48, kde=False)
plt.xlim([0,48])
plt.xticks(np.arange(0,54,6))
plt.xlabel('Time After First Transaction (hr)')
plt.ylabel('Count')
plt.title('Transaction Times')
plt.show()

X_train['Amount'].describe()

plt.figure(figsize=(12,4), dpi=80)
sns.distplot(X_train['Amount'], bins=300, kde=False)
plt.ylabel('Count')
plt.title('Transaction Amounts')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.boxplot(X_train['Amount'])
plt.title('Transaction Amounts')
plt.show()

print(X_train['Amount'].skew())


X_train.loc[:,'Amount'] = X_train['Amount'] + 1e-9 # Shift all amounts by 1e-9

X_train.loc[:,'Amount'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(X_train['Amount'], alpha=0.01)

print(maxlog)

print(min_ci, max_ci)

plt.figure(figsize=(12,4), dpi=80)
sns.distplot(X_train['Amount'], kde=False)
plt.xlabel('Transformed Amount')
plt.ylabel('Count')
plt.title('Transaction Amounts (Box-Cox Transformed)')
plt.show()

print(X_train['Amount'].describe())

print(X_train['Amount'].skew())


X_test.loc[:,'Amount'] = X_test['Amount'] + 1e-9 # Shift all amounts by 1e-9

X_test.loc[:,'Amount'] = sp.stats.boxcox(X_test['Amount'], lmbda=maxlog)

#compare distriptive statstics od pca variables

pca_vars = ['V%i' % k for k in range(1,29)]

print(X_train[pca_vars].describe())


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=X_train[pca_vars].mean(), color='darkblue')
plt.xlabel('Column')
plt.ylabel('Mean')
plt.title('V1-V28 Means')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=X_train[pca_vars].std(), color='darkred')
plt.xlabel('Column')
plt.ylabel('Standard Deviation')
plt.title('V1-V28 Standard Deviations')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=X_train[pca_vars].skew(), color='darkgreen')
plt.xlabel('Column')
plt.ylabel('Skewness')
plt.title('V1-V28 Skewnesses')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.distplot(X_train['V8'], bins=300, kde=False)
plt.ylabel('Count')
plt.title('V8')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.boxplot(X_train['V8'])
plt.title('V8')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
plt.yscale('log')
sns.barplot(x=pca_vars, y=X_train[pca_vars].kurtosis(), color='darkorange')
plt.xlabel('Column')
plt.ylabel('Kurtosis')
plt.title('V1-V28 Kurtoses')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=X_train[pca_vars].median(), color='darkblue')
plt.xlabel('Column')
plt.ylabel('Median')
plt.title('V1-V28 Medians')
plt.show()

plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=X_train[pca_vars].quantile(0.75) - X_train[pca_vars].quantile(0.25), color='darkred')
plt.xlabel('Column')
plt.ylabel('IQR')
plt.title('V1-V28 IQRs')
plt.show()

with warnings.catch_warnings():  # Suppress warnings from the matthews_corrcoef function
    warnings.simplefilter("ignore")

# random forest modeleation cr
pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])

param_grid_rf = {'model__n_estimators': [75]}
MCC_scorer = make_scorer(matthews_corrcoef)

grid_rf = GridSearchCV(estimator=pipeline_rf,param_grid=param_grid_rf, scoring=MCC_scorer, n_jobs=-1,pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

print(grid_rf.fit(X_train, y_train))

print(grid_rf.best_score_)


def classification_eval(estimator, X_test, y_test):
    """
    Print several metrics of classification performance of an estimator, given features X_test and true labels y_test.

    Input: estimator or GridSearchCV instance, X_test, y_test
    Returns: text printout of metrics
    """
    y_pred = estimator.predict(X_test)

    # Number of decimal places based on number of samples
    dec = np.int64(np.ceil(np.log10(len(y_test))))

    print('CONFUSION MATRIX')
    print(confusion_matrix(y_test, y_pred), '\n')

    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, y_pred, digits=dec))

    print('SCALAR METRICS')
    format_str = '%%13s = %%.%if' % dec
    print(format_str % ('MCC', matthews_corrcoef(y_test, y_pred)))
    if y_test.nunique() <= 2:  # Additional metrics for binary classification
        try:
            y_score = estimator.predict_proba(X_test)[:, 1]
        except:
            y_score = estimator.decision_function(X_test)
        print(format_str % ('AUPRC', average_precision_score(y_test, y_score)))
        print(format_str % ('AUROC', roc_auc_score(y_test, y_score)))
    print(format_str % ("Cohen's kappa", cohen_kappa_score(y_test, y_pred)))
    print(format_str % ('Accuracy', accuracy_score(y_test, y_pred)))

classification_eval(grid_rf, X_test, y_test)


#save the model
pickle.dump(pipeline_rf, open('model4.pkl', 'wb'))
model = pickle.load(open('model4.pkl', 'rb'))