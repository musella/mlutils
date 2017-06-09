import numpy as np
import pandas as pd
import os

import pandas.tseries.converter as converter

from sklearn.preprocessing import FunctionTransformer

# -----------------------------------------------------------------------------------------------------------------
def string_to_datetime(X):
    return converter.DatetimeConverter().convert(pd.to_datetime(X),None,None)

# -----------------------------------------------------------------------------------------------------------------
def datetime_to_week_day(val):
    dti = pd.DatetimeIndex(val)
    return dti.week, dti.dayofweek

# -----------------------------------------------------------------------------------------------------------------
def datetime_to_float(val):
    return np.array(val).astype(np.float64) / 3.1536e+7 # one year in seconds

# -----------------------------------------------------------------------------------------------------------------
def apply_transformer(df,transformer,inp,out,fit=False,raw=False,inline=False,steps=None):
    X = df if inp == None else df[inp]
    if raw:
        X = X.values
    if fit:
        transformer.fit(X)
    if inline or type(out) == str:
        df[out] = transformer.transform(X)
    else:
        Xtr = transformer.transform(X)
        if type(Xtr) == list:
            pairs = zip(out,Xtr)
        else:
            pairs = zip(out,Xtr.T)
        for col, val in pairs:
            df[col] = val
    if steps != None:
        steps.append( (inp,out,transform) )
    return df

# -----------------------------------------------------------------------------------------------------------------
def encode_time_column(df,src,dst='time',add_day_week=False,steps=None,add_float=False):
    apply_transformer(df,
                      FunctionTransformer(string_to_datetime,validate=False),
                      [src],dst,steps=steps)
    if add_day_week:
        apply_transformer(df,
                          FunctionTransformer(datetime_to_week_day,validate=False),
                          None,['%s_week' % dst,'%s_day' % dst],steps=steps)
        
    if add_float:
        apply_transformer(df,
                          FunctionTransformer(datetime_to_float),
                          [dst],float_time = '%_float' % time,steps=steps)
        
# -----------------------------------------------------------------------------------------------------------------
def encode_label_column(df,name):
    le = LabelEncoder()
    return name,"encoded_%s" % name,le.fit_transform(df[name].values.ravel()),le

# -----------------------------------------------------------------------------------------------------------------
def encode_obj_columns(df,n_jobs=4,skip_cols=[],skip_patterns=[],steps=None):
    objcolumns = filter(lambda col: (df[col].dtype == np.object) and not (col in skip_cols) and not any(pat in col for pat in skip_patterns), df.columns)
    
    enccolumns = Parallel(n_jobs=n_jobs,verbose=20)(delayed(encode_column)(df,name) for name in objcolumns)
    names = []
    for inp,name,vals,encoder in enccolumns:
        df[name] = vals
        if steps != None:
            steps.append( (name,inp,encoder) )
        names.append(name)
                          
    return names

# -----------------------------------------------------------------------------------------------------------------
def run_imputer(vals,strategy):
    imputer = Imputer(strategy=strategy,axis=0,copy=False)
    return imputer.fit_transform(vals),imputer

# -----------------------------------------------------------------------------------------------------------------
def fillna(df,n_jobs=4,strategy='median',skip_cols=[],skip_patterns=[],steps=None):
    numcolumns = filter(lambda col: (df[col].dtype != np.object) and not (col in skip_cols) and not any(pat in col for pat in skip_patterns), df.columns)
    
    results = Parallel(n_jobs=n_jobs,verbose=20)(delayed(run_imputer)(df[[name]]) for name in numcolumns)
    
    for icol,col in enumerate(numcolumns):
        df[col] = results[icol][0]
        if steps != None:
            steps.append( (col,col,results[icol][1]) )

