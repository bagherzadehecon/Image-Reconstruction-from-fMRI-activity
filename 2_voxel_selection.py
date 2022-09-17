
import numpy
import scipy.io
from numpy import mean
import numpy as np
from scipy.stats import ks_2samp
import pandas
from statsmodels.formula.api import ols

x = scipy.io.loadmat('./six_nine_data/X.mat')
y = scipy.io.loadmat('./six_nine_data/Y.mat')
I = scipy.io.loadmat('./six_nine_data/I.mat')
structural= scipy.io.loadmat('./six_nine_data/structural.mat')
funcloc= scipy.io.loadmat('./six_nine_data/funcloc.mat')
prior= scipy.io.loadmat('./six_nine_data/prior.mat')
masks= scipy.io.loadmat('./six_nine_data/masks.mat')

images=I['I']
labels=y['Y']
line_label=labels.ravel()
beta=x['X']
##############################################################################################################initialize
upper=1
lower=0.1
npixel=7
vox_sel_num=10
npart=112
############################################################################################## threshholding beta values
threshold=mean(beta)
# print(threshold)

binary_x=np.where(beta>threshold, upper, lower)
####################################################################################KOLMOGOROV–SMIRNOV TEST with 2sample
data=numpy.concatenate((binary_x,line_label[:,numpy.newaxis]),axis=1)
# print(df.shape)
df=pandas.DataFrame(data)
lengthx=binary_x.shape[1] #3092
ntrial=binary_x.shape[0]  #100

pval=np.zeros([1,lengthx])

for i in range(1,lengthx):
    # if(sum(df[i]) >0):
        # print(sum(df[i]))
        mod = ols('df[[i]] ~ df[[lengthx]]', data=df).fit()
        pval[0, i] = ((mod.pvalues)[1])
        # print(pval.shape)

index=numpy.argsort(pval)
start=np.where(pval[0,index]>0)[1]
print(start)
selected_voxels=binary_x[:,index[0,start[1]:start[1]+vox_sel_num]]
print(selected_voxels.shape)
# print(pval[0,index[0,0:vox_sel_num]])
# x=(selected_voxels.astype(bool).sum(axis=0)) #count nonzero elements of each column of data frame
np.array(selected_voxels).dump(open('selected_voxels.npy', 'wb'))
















