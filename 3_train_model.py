import numpy
import pandas
from pomegranate import BayesianNetwork
from pomegranate import HiddenMarkovModel
from sklearn.externals import joblib
import _pickle as cPickle
import pickle
import scipy.io
import numpy as np
import random as rm

#####################################################################################################initialize
vox_sel_num=10
pix_sel_num=7
n=112
npart=112
ntrial=100

# partition = numpy.load(open('partition.npy', 'rb'))
I = scipy.io.loadmat('./six_nine_data/I.mat')
images=I['I']
partition=numpy.reshape(images,[npart,ntrial,pix_sel_num])
selected_voxels = numpy.load(open('selected_voxels.npy', 'rb'))
predicted_pixel=numpy.zeros([npart,ntrial,pix_sel_num])
##############################################

for i in range(1,n):
    ind=vox_sel_num
    selected_partition=partition[i,:,:]
    data=numpy.concatenate((selected_voxels,selected_partition),axis=1)


    #################################################################################################################network
    model = HiddenMarkovModel.from_samples(data,algorithm='exac')


    for j in range(1,pix_sel_num):
        ind=j+vox_sel_num
        print(ind)
        data_p = numpy.concatenate((selected_voxels, selected_partition), axis=1)
        print(data_p.shape)
        data_p[:,ind-1]=numpy.nan
        print(data_p[0:99,10:16])
        predicted=model.predict(data_p)
        predicted_arr=numpy.array(predicted)
        predicted_pixel[i,:,j]=predicted_arr[:,ind]
        print(predicted_arr[:,ind])
        print('part:')
        print(i)
        print('pixel:')
        print(j)

####################################################################################################save prediction
numpy.array(predicted_pixel).dump(open('predicted_pixel.npy','wb'))
