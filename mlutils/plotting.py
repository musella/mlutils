import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

# -------------------------------------------------------------------------
def corr_matrix(df,figsize=(20,20),fignum=0):
    plt.figure(figsize=figsize)
    corr = df.corr()
    plt.matshow(corr,fignum=fignum, vmin=-1, vmax=1)
    plt.xticks(np.arange(len(corr.columns)),corr.columns,rotation='vertical')
    plt.yticks(np.arange(len(corr.columns)),corr.columns)
    # plt.clims(-1.,1.)
    plt.colorbar()
    
# -------------------------------------------------------------------------
def pca_weights(tn,figsize=(100,30)):
    plt.figure(figsize=figsize)
    plt.matshow(tn.pca_decompose.components_.transpose(),fignum=0)
    plt.yticks(np.arange(len(tn.pca_inputs)),tn.pca_inputs)
    plt.colorbar()

# -------------------------------------------------------------------------
def scatter_matrix(df):
    scatter_matrix(df,figsize=[20,20],c=np.floor(np.log(df['SalePrice'])*10).astype(np.int32))
    
    
# -------------------------------------------------------------------------
def deviance(clf,X_test, y_test):
    # compute test set deviance
    test_score = np.zeros(clf.n_estimators, dtype=np.float64)
    
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
            
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(clf.n_estimators) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(clf.n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

# -------------------------------------------------------------------------
def time_fit(X,y,y_pred,name):
    fig = plt.figure()
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(X, y_pred, 'b-', label=u'Prediction')
    plt.xlabel('time')
    plt.ylabel(name)
    plt.legend(loc='best')
    plt.show()
