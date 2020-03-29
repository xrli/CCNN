import os
import sys
import pandas as pd
from glob import glob
from keras.models import load_model
import keras_metrics
from data import load_pfd
from model import test
from keras.utils.generic_utils import CustomObjectScope

#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

if __name__ == '__main__':


    
    ## load pfd file
    path = ['PICS-ResNet_data/test_data/pulsar', 'PICS-ResNet_data/test_data/rfi']
    fpath_list = []
    fname_list = []
    test_data = []
    print('loading pfd ......')
    # obtain the path and filename of all the pfd file
    for path_item in path:
        path_item_temp = os.path.join(path_item, '*.pfd')
        fpath_list.extend(glob(path_item_temp))

    # load subplots from pfd
    for f in fpath_list:
        test_data.append(load_pfd(f))
        fname_list.append(f.split('/')[-1])
        
    # save data to dataframe
    test_data = pd.DataFrame(test_data, index=fname_list)
    
    print('testing ......')
    # load model from the code path
    with CustomObjectScope({'binary_precision': keras_metrics.precision(), 'binary_recall':keras_metrics.recall()}):
        model = load_model('trained_model/H-CCNN.h5', compile=False)
    
        # running test
        y_prod = test(model, test_data)

        # save predict result
        result = pd.DataFrame([test_data.index, y_prod[:, -1]]).T
        result.to_csv('result.txt', header=None, index=None)