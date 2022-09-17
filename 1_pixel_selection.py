
import numpy
import scipy.io

from statsmodels.formula.api import ols
import pandas
I = scipy.io.loadmat('./six_nine_data/I.mat')
images=I['I']
y = scipy.io.loadmat('./six_nine_data/Y.mat')
labels=y['Y']
line_label=labels.ravel()

#########################################################################################################initialize


pix_sel_num=7
lengthi=images.shape[1] #784
ntrial=100
npart=112
npixel=7


#########################################################################################################pixel selection
pval=numpy.zeros([1,lengthi])
# print(images.shape)
# print(line_label.shape)

data=numpy.concatenate((images,line_label[:,numpy.newaxis]),axis=1)
# print(df.shape)
df=pandas.DataFrame(data)


for i in range(1,lengthi):

        mod = ols('df[[i]] ~ df[[784]]' ,data=df).fit()
        pval[0,i]=((mod.pvalues)[1])
        # print(pval.shape)




index=numpy.argsort(pval)
sorted_img=images[:,index[0,:]]
print(sorted_img.shape)
# print(sorted_img.astype(bool).sum(axis=0)) #count nonzero elements of each column of data frame

partition= numpy.zeros([npart,ntrial,npixel])
print(partition.shape)

partition=numpy.reshape(sorted_img,[npart,ntrial,npixel])

numpy.array(partition).dump(open('partition.npy', 'wb'))
numpy.array(index).dump(open('index.npy', 'wb'))


