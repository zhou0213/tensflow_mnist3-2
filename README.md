# tensflow_mnist3-2

'''
这份代码是作业1、2、3、4的综合版本，把四个作业融合在一起了。
    执行首先会挑选出0~9中各自的前8个数     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>作业一
    随后会生成一个  数字与标号的对照表
    然后针对list中的顺序，把images存入创建的文件夹中，并命名      >>>>>>>>>>>>>>>>>>>>>>>>>>>>>作业四
    写一个判断函数，如果已经生成了存储图片的目录则直接选择想要可视化的图片的标号 >>>>>>>>>>>>>>>作业二
                    如果没有生成存储图片的目录，则直接创建该目录，创建完成会重新执行判断函数，可视乎标号的图片
    对mnist数据集进行100次训练，并生成权重图   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>作业三
'''

import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time
mnist = input_data.read_data_sets('input_data/', one_hot=True)
list = []
x2d = np.array(mnist.test.labels)
#挑选出0~9中，各8个数字
for i in  range(10):
    count = 0
    for j in range(10000):
        if(x2d[j,i]==1):
            list.append(j)
            count+=1
            if(count==8):
                break
x3d = np.array(list)
a = 0
print("对照表如下：——————————————————————————————————————")
for i in range(10):
    print(i,">>>>>",x3d[a:a+8:1])
    a+=8
#x3d.shape=(10,8)
#print(x3d)
#根据生成数字组成的列表，将图片顺序存储到文件夹中
def create (charge=False):
    if not charge:
        os.mkdir('test_picture/')
        b = 0
        for i in range(10):
            title = ("%s" % i)
            number = os.path.join('test_picture/', title)
            os.mkdir(number)
            count = 0
            for a in range(80):
                plt.imshow(mnist.test.images[list[a + b]].reshape(28, 28))
                plt.savefig(r'test_picture/%s\No%s.png' % (i, list[a + b]))
                count += 1
                if (count > 7):
                    b += 8
                    break
        print("您挑选出的目录已经创建成功！！！")


#判断时候有保存图片的文件夹，没有则创建
def func():
    print("当前文件夹存在的目录：")
    mulu = os.listdir()
    count2 = 0
    for i in range(len(mulu)):

        print("-------", mulu[i])
        if (mulu[i] == 'test_picture'):


#选择自己想可视化的图片编号，并进行可视化

            print("存在目录，即将显示图像~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            nn = input("按照对照表，输出需要打印的图片编号：")
            a = int(nn)
            plt.imshow(mnist.test.images[a].reshape(28, 28))
            plt.show()
            path = create(True)
            break
        else:
            count2 += 1
            if (count2 == len(mulu)):
                print("不存在目录，正在创建需要的目录````````~~~~~~~~~~```````````~~~~~~~~~请稍等")
                path = create(False)
                return func()

func()

            #下面画的是权重图
            
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
accuracies = []

#with tf.Session() as sess:
tf.global_variables_initializer().run()
for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



#画出权重图
for i in range(10):
 plt.subplot(2, 5, i+1)    #分成子图，（2x5），逐个放入
 weight = sess.run(W)[:,i]    #一列代表一个数值
 plt.title(i)
 plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('Greys_r'))      #camp参数、灰度图
 frame1 = plt.gca()      #获得子图
 frame1.axes.get_xaxis().set_visible(False)
 frame1.axes.get_yaxis().set_visible(False)
plt.show()












