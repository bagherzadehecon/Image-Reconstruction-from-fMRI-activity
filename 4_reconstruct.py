import numpy
import scipy.io
import matplotlib.pyplot as plt
###########################################################################################################initialize
index = numpy.load(open('index.npy', 'rb'))
predicted_pixel = numpy.load(open('predicted_pixel.npy', 'rb'))
npart=112
ntrial=100
npixel=7
vox_sel_num=10
pix_sel_num=7
size=28
reconstructed_img=numpy.zeros([ntrial,npart*npixel])

I = scipy.io.loadmat('./six_nine_data/I.mat')
images=I['I']
###########################################################################################################reconstruct
print(predicted_pixel.shape)
reconstructed_img=numpy.reshape(predicted_pixel,[ntrial,npart*npixel])
# print(reconstructed_img[20,:])

# for i in range (1,npart*npixel):
#     reconstructed_img[:,index[0,i]]=reconstructed_img[:,i]


# ###########################################################################################################visualize
plt.imshow(numpy.reshape(reconstructed_img[5,:],[size,size]))
plt.show()
plt.imshow(numpy.reshape(numpy.transpose(images[60,:]),[size,size]))
plt.show()