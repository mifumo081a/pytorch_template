# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:33:29 2021

@author: tsakailab

4GB以上のデータをdumpしようとするとプロトコルを指定する必要がある。
"""

import pickle
import os
import zipfile

def read_as_pickle(fname,path):
    """
    pickle data 読み取り用関数
    
    Parameters
    ----------
        fname : str
            読み込まれるファイル名
            xxx.pklを想定
        path : str
            読み込まれるファイルの直上パス
    """

    if (fname[-4:] != '.pkl'):
        fname = fname+'.pkl'
    print(os.path.join(path, fname))
    if not os.path.exists(os.path.join(path, fname)):
        print('error! no such files!')
        return
    with open(os.path.join(path, fname),'rb') as f:
        return pickle.load(f)
    

def save_as_pickle(data,fname,path=None):
    """
    
    Parameters
    ----------
        data : object
            保存されるデータ
        fname : str
            保存時の名称
        path : str
            保存先の直上パス
            非設定時はカレントディレクトリに保存
    """
    if(path == None):
        with open(fname+'.pkl','wb') as f:
            pickle.dump(data,f,protocol=4)
    else:
        with open(os.path.join(path, fname+'.pkl'),'wb') as f:
            pickle.dump(data,f,protocol=4)
