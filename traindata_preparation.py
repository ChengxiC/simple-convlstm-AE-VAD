import numpy
import torch
from os.path import isdir, isfile, exists, join
from os import mkdir, makedirs, listdir
import numpy as np
from PIL import Image


train_data_path = r'E:\data\UCSD\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train/'  # the path you need to change
train_datapoint_path = './train/'

if not exists(train_datapoint_path):
    makedirs(train_datapoint_path)

train_id = 0  # to track the processing data
strides = [1, 2, 3]  # the sampling rate

for file in sorted(listdir(train_data_path)):

    dir_path = join(train_data_path, file)

    if isdir(dir_path):
        print('train_id_' + str(train_id))
        video = []  # to store the datapoint

        for frame in sorted(listdir(dir_path)):
            img_path = join(dir_path, frame)
            if str(img_path)[-3:] == 'tif':  # if it is a tif image
                # then: resize => normalization => unsqueeze
                img = Image.open(img_path).resize(size=[227, 227])
                img = np.array(img, dtype=numpy.float32) / 255.0  # to [0, 1]
                img = np.expand_dims(img, axis=0)  # (227, 227) => (1, 227, 227)
                video.append(img)

        video_len = len(video)
        count = 0
        temp = np.zeros(shape=[10, 1, 227, 227])
        for stride in strides:
            for idx in range(0, video_len, stride):
                temp[count, ...] = video[idx]
                count += 1

                if count == 10:  # 10 frames for a datapoint
                    torch.save(torch.tensor(temp, dtype=torch.float32), train_datapoint_path + 'train_id_' + str(train_id) + '.pt')
                    train_id += 1
                    count = 0

print('train data preparation is finished!')





