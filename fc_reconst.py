import numpy
import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


caps_f=np.load('caps2_output_value.npy')

capsule_feature=np.concatenate((caps_f[:,0,6,:,0],caps_f[:,0,9,:,0]),1)
print('capsule_feature',capsule_feature.shape)
selected_voxel=np.load('selected_voxels.npy')
print('selected_voxel',selected_voxel)


#parameters
Number_hidden_1=256
Number_hidden_2=128
Number_hidden_3=32
iterations_train=8
iterations_test=2
batch_size=10
landa=0.0001
epochs=20000


x = tf.placeholder(dtype=tf.float32, shape=(None, 100), name="input")
y = tf.placeholder(dtype=tf.float32, shape=(None, 32), name="label")

with tf.name_scope(name="Hidden_layer1"):
    w1 = tf.Variable(tf.random_normal(shape=(100, Number_hidden_1), mean=0, stddev=2, seed=1), name="W")
    b1 = tf.Variable(tf.zeros([Number_hidden_1]), name="B")

    hidden_layer1_input = tf.matmul(x, w1) + b1
    relu1 = tf.nn.relu(hidden_layer1_input)

with tf.name_scope(name="Hidden_layer2"):
    w2 = tf.Variable(tf.random_normal(shape=(Number_hidden_1, Number_hidden_2), mean=0, stddev=2, seed=1), name="W")
    b2 = tf.Variable(tf.zeros([Number_hidden_2]), name="B")

    hidden_layer2 = tf.matmul(relu1, w2) + b2
    relu2 = tf.nn.relu(hidden_layer2)


with tf.name_scope(name="Hidden_layer3"):
    w3 = tf.Variable(tf.random_normal(shape=(Number_hidden_2, Number_hidden_3), mean=0, stddev=2, seed=1), name="W")
    b3 = tf.Variable(tf.zeros([Number_hidden_3]), name="B")

    hidden_layer3 = tf.matmul(relu2, w3) + b3
    out3=hidden_layer3


with tf.name_scope("xent"):
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))
    loss = tf.reduce_mean(tf.losses.mean_squared_error(y,out3))

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(landa).minimize(loss)


tf.summary.scalar("mse", loss)

merge = tf.summary.merge_all()
filewriter = tf.summary.FileWriter("learning monitoring")

sess = tf.Session()
filewriter.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(sess=sess,save_path="save/")


selected_voxel_tr, selected_voxel_val ,capsule_feature_tr,capsule_feature_val,indices_train,indices_test = train_test_split(selected_voxel,capsule_feature,np.array(range(100)),test_size=1/10, random_state=0)


for j in range(epochs):
    t_loss=0
    for i in range(iterations_train):

                batch_xs = selected_voxel_tr[i*batch_size:(i+1)*batch_size,:]
                batch_ys = capsule_feature_tr[i*batch_size:(i+1)*batch_size,:]

                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                b = (sess.run(merge, feed_dict={x: batch_xs, y: batch_ys}))

                filewriter.add_summary(b, i)

                train_loss,feature = sess.run((loss,out3), feed_dict={x: batch_xs, y: batch_ys})
                t_loss+=train_loss

                if (i == 0):
                    feature_train = feature

                else:
                    feature_train = np.concatenate((feature_train, feature), axis=0)

                if(i==iterations_train-1):
                    np.save('first_out_tr.npy', feature_train)


    print(" epoch %5i iteration %5i (train_loss) is %g" % (j,i, t_loss/iterations_train))


    v_loss=0
    for i in range(iterations_test):

                batch_xs = selected_voxel_val[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = capsule_feature_val[i * batch_size:(i + 1) * batch_size, :]

                # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                # saver.save(sess=sess,save_path="save/")

                c = (sess.run(merge, feed_dict={x: batch_xs, y: batch_ys}))

                filewriter.add_summary(c, i)

                validation_loss,feature = sess.run((loss,out3 ), feed_dict={x: batch_xs, y: batch_ys})
                v_loss+=validation_loss


                if(i==0):
                    feature_tot=feature

                else:
                    feature_tot=np.concatenate((feature_tot,feature),axis=0)


                if(i==iterations_test-1):
                    np.save('first_out_ts.npy',feature_tot)

    print(" epoch %5i iteration %5i (validation_loss) is %g" % (j, i, v_loss/iterations_test))


print(feature_tot.shape)
np.save('indices_test.npy',indices_test)
np.save('indices_train.npy',indices_train)
saver.save(sess=sess,save_path="first_part")