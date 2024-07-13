from os import mkdir, makedirs, listdir
from os.path import isdir, isfile, exists, join
import numpy
from PIL import Image
import numpy as np
import torch


test_data_path = r'E:\data\UCSD\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test/'  # the path you need to change
test_datapoint_path = './test/'
test_id = 0

if not exists(test_datapoint_path):
    makedirs(test_datapoint_path)

for files in sorted(listdir(test_data_path)):
    dir_path = join(test_data_path, files)

    if isdir(dir_path):
        print('test_id_' + str(test_id))
        video = []  # to store the frames in idr_path
        for frame in sorted(listdir(dir_path)):
            img_path = join(dir_path, frame)

            if str(img_path)[-3:] == 'tif':
                # resize => normalization => unsqueeze
                img = Image.open(img_path).resize(size=[227, 227])
                img = np.array(img, dtype=numpy.float32) / 255.0  # turn to numpy and normalization
                img = np.expand_dims(img, axis=0)  # (227, 227) => (1, 227, 227)
                video.append(img)

        video_len = len(video)
        count = 0
        temp = np.zeros(shape=[10, 1, 227, 227])
        for idx in range(0, video_len):
            temp[count, ...] = video[idx]
            count += 1
            if count == 10:
                torch.save(torch.tensor(temp, dtype=torch.float32), test_datapoint_path + 'test_id_' + str(test_id) + '.pt')
                test_id += 1
                count = 0

print('the test data preparation is finished!')


