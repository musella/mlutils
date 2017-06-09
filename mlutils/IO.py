import cPickle as pickle
from gzip import open as gopen
from glob import glob

import os

inputFolder = "../data"

def read_data(fname):
    df = pd.read_csv(os.path.join(IO.inputFolder,fname))
    return df

def to_pickle(name,obj):
    with gopen('%s.pkl.gz' % name,'w+') as out:
        pickle.dump(obj,out)
        out.close()

def read_pickle(name):
    with gopen('%s.pkl.gz' % name,'r') as fin:
        obj = pickle.load(fin)
        fin.close()
        return obj
