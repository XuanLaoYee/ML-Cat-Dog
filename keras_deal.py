#This code uses keras framework based on tensorflow2.0 version
#You run better with jupyter
#When you run this code you need to run keras_data_deal.py first
#In fact, the accuracy of the current version is not high, it can only be about 90%.
# If you want to improve the accuracy, adjust the parameters and re-run.
# The amount of data here is large, and it takes about 1.5h to run it once.

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Dense,Dropout,Activation,Flatten
import os
import pickle
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

model = Sequential() #建立一个线性堆叠模型

model.add(Conv2D(filters=50,kernel_size=(3,3),
                 input_shape=(100,100,3),
                 activation='relu',
                padding='same')) #建立卷积层，处理100*100的图像,有50个过滤器

model.add(Dropout(rate=0.25)) #每次训练时都会放弃25%的神经元防止过拟合
model.add(MaxPooling2D(pool_size=(2,2))) #建立池化层1，执行缩减采样，把100*100的图像缩小成50*50的,仍然是50个
model.add(Conv2D(filters=100,kernel_size=(3,3),
                activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层2，执行缩减采样，把50*50的图像缩小成25*25的，仍然是100个
model.add(Flatten())
model.add(Dropout(rate=0.25)) #建立平坦层
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25)) #建立隐藏层
model.add(Dense(2,activation='softmax')) #建立输出层
#下面开始进行训练
model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])#训练前要进行设置
def load_data(filename):
    '''从batch文件中读取图片信息'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='ascii')
        return data['data'],data['label'],data['filenames']

data_dir = './keras_batch_files'
train_filenames = [os.path.join(data_dir, 'train_batch')]
test_file = [os.path.join(data_dir, 'test_batch')]
img_train,label_train,filename = load_data('./src/keras_batch_files/train_batch')
img_train_array=np.array(img_train,dtype=float)
label_train_array=np.array(label_train,dtype=int)
# label_train_array.shape
img_train_array = img_train_array.reshape(img_train_array.shape[0],100,100,3)/255
img_train_array.shape
label_train_array=label_train_array.reshape(len(label_train_array),-1)
# label_train_array.shape
label_train_array = np_utils.to_categorical(label_train_array,num_classes=2)
# label_train_array.shape
train_history = model.fit(img_train_array,label_train_array,
                   validation_split=0.2,
                epochs=10,batch_size=200,verbose=1)
plt.plot(train_history.history['val_loss'])
plt.plot(train_history.history['loss'])

#下面开始预测
img_test,label_test,filename_test = load_data('./src/keras_batch_files/test_batch')
img_test_array=np.array(img_test,dtype=float)
label_test_array=np.array(label_test,dtype=int)
img_test_array = img_test_array.reshape(img_test_array.shape[0],100,100,3)/255

prediction=model.predict_classes(img_test_array) #数值预测x
print(prediction[80:100])
print(label_test_array[80:100])
