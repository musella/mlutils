### import numpy as np
### import pandas as pd
### import os
### 
### import matplotlib.pyplot as plt
### from pandas.tools.plotting import scatter_matrix
### 
### from sklearn.preprocessing import LabelEncoder,Imputer
### import pandas.tseries.converter as converter
### 
### from sklearn.decomposition import PCA
### from sklearn.feature_selection import SelectKBest, mutual_info_regression
### from sklearn import ensemble
### 
### from sklearn.gaussian_process import GaussianProcessRegressor
### from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
### 
### from sklearn.preprocessing import FunctionTransformer
### from sklearn.pipeline import make_pipeline
### 
### from sklearn.metrics import mean_squared_error
### from sklearn import preprocessing
### from sklearn.pipeline import Pipeline
### 
### import cPickle as pickle
### from gzip import open as gopen
### 
### from xgboost.sklearn import XGBRegressor
### 
### from sklearn.model_selection import train_test_split
### from sklearn.model_selection import GridSearchCV
### 
### from joblib import Parallel, delayed
### 
### from glob import glob
### 
### # -------------------------------------------------------------------------
### class trainer(object):
###     
###     
###     # -------------------------------------------------------------------------
###     def __init__(self,df):
###         
###         self.rawdf = df
###         self.df = None
###                 
###     # -------------------------------------------------------------------------
###     def prepare_data(self,fillna=False,onehot=False):
###     
###         df = self.rawdf
### 
###         if onehot:
###             preprocessing.encode(df)
###             
###         self.df = df
###         
###         if fillna:
###             preprocessing.fillna(df)
###             
###     # -------------------------------------------------------------------------
###     def run_pca(self,skip,normalize=False,**kwargs):
###         
### 
###         self.pca_inputs = filter(lambda x: not x in skip, self.df.columns)
###         
###         self.pca = PCA(**kwargs) 
###         self.pca_decompose = self.pca
###         
###         if normalize:
###             self.min_max_scaler = preprocessing.MinMaxScaler()
###             self.pca_decompose = self.pca
###             self.pca = Pipeline( [("normalize",self.min_max_scaler), 
###                                   ("decompose",self.pca_decompose)] )
###             
###         self.pca.fit(self.df[self.pca_inputs])
###         
###         self.pcadf = self.df[skip].copy()
###     
###         pccol = self.pca.transform(self.df[self.pca_inputs])
### 
###         for icol in range(pccol.shape[1]):
###             self.pcadf['Comp%d' % icol] = pccol[:,icol]
###         
###         
###     # -------------------------------------------------------------------------
###     def run_kbest(self,skip,target,**kwargs):
###         
###         if not "score_func" in kwargs:
###             kwargs["score_func"] = mutual_info_regression
### 
###         self.select = SelectKBest(**kwargs)
###         skip.append(target)
###         self.select.fit(self.pcadf[filter(lambda x: x not in skip, self.pcadf)],
###                         self.pcadf[target])
###         
###         self.kbest_comps = map(lambda x: 'Comp%d' % (x[0]), 
###                                filter(lambda y: y[1], enumerate(self.select.get_support())))
###         
###         
###     # -------------------------------------------------------------------------
###     def run_clf(self,target,run_on_pca=False,skip=[],do_lasso=False,use_xgboost=False,
###                 hyper_tune=None,split=True,
###                 **kwargs):
###         
###         if run_on_pca:
###             self.traindf = self.pcadf
###             self.train_feats = self.kbest_comps
###         else:
###             self.traindf = self.df
###             skip.append(target)
###             numcolumns = filter(lambda col:self.df[col].dtype != np.object and col != 'time', self.df.columns)
###             self.train_feats = filter(lambda x: not x in skip, numcolumns)
### 
###         for feat in self.train_feats:
###             print(feat)
###             
###         X, y = self.traindf[self.train_feats], self.traindf[target]
###         if split:
###             X_train, X_test, y_train, y_test = train_test_split(X,y)
###         else:
###             X_train, X_test, y_train, y_test = X, None, y, None
### 
###         self.X_train = X_train
###         self.X_test = X_test
###         self.y_train = y_train
###         self.y_test = y_test
### 
###         if do_lasso:
###             pass
###         else:
###             if use_xgboost:
###                 self.clf = XGBRegressor(**kwargs)
###             else:
###                 self.clf = ensemble.GradientBoostingRegressor(**kwargs)
###             
###         if hyper_tune != None:
###             search_algo = hyper_tune.pop("search_algo",GridSearchCV)
###             self.clf = search_algo(self.clf,**hyper_tune)
###             
### 
###         self.clf.fit(X_train, y_train)
###         if split:
###             self.mse = mean_squared_error(y_test, self.clf.predict(X_test))

