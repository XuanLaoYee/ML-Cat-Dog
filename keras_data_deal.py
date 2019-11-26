import cv2 as cv
import os
import numpy as np
import random
import pickle
import time
#注释请参考data_deal.py

start_time=time.time()

data_dir='./picture'
keras_data_path='./keras_batch_files'

os.makedirs(keras_data_path, exist_ok=True)
all_data_files = os.listdir(os.path.join(data_dir, 'train/'))

random.shuffle(all_data_files)

all_train_files = all_data_files[:20000]
all_test_files = all_data_files[20000:]

train_data = []
train_label = []
train_filenames = []

test_data = []
test_label = []
test_filenames = []

for each in all_train_files:
    img = cv.imread(os.path.join(data_dir,'train/',each),1)
    resized_img = cv.resize(img, (100,100))

    img_data = np.array(resized_img)
    train_data.append(img_data)
    if 'cat' in each:
        train_label.append(0)
    elif 'dog' in each:
        train_label.append(1)
    else:
        raise Exception('%s is wrong train file'%(each))
    train_filenames.append(each)

for each in all_test_files:
    img = cv.imread(os.path.join(data_dir,'train/',each), 1)
    resized_img = cv.resize(img, (100,100))

    img_data = np.array(resized_img)
    test_data.append(img_data)
    if 'cat' in each:
        test_label.append(0)
    elif 'dog' in each:
        test_label.append(1)
    else:
        raise Exception('%s is wrong test file'%(each))
    test_filenames.append(each)

print(len(train_data), len(test_data))

# 制作100个batch文件
start = 0
end = 20200

batch_data = train_data[start: end]
batch_label = train_label[start: end]
batch_filenames = train_filenames[start: end]

all_data = {
    'data': batch_data,
    'label': batch_label,
    'filenames': batch_filenames,
}

with open(os.path.join(keras_data_path, 'train_batch'), 'wb') as f:
    pickle.dump(all_data, f)


# 制作测试文件
all_test_data = {
    'data':test_data,
    'label':test_label,
    'filenames':test_filenames,
    'name':'test batch 1 of 1'
}

with open(os.path.join(keras_data_path, 'test_batch'), 'wb') as f:
    pickle.dump(all_test_data, f)
end_time = time.time()
print('制作结束, 用时{}秒'.format(end_time - start_time))
