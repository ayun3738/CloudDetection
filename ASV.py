import cv2
import os
import numpy as np

base = './Cloud_Classification-14'

for label_name in ['train', 'test', 'valid']:
    print(f'{label_name} start')

    folder_paths = base + '/' + label_name
    folder_path = folder_paths + '/images'
    files = os.listdir(folder_path)
    # samples
    for file in files:
        if file.split('.')[-1] == 'jpg':
            # sample
            # full_path = 'E:/alpaco/Cloud/Cloud_Classification-1/train/images/4a0f769_jpg.rf.daae64127387769bf6153384a10c8519.jpg'
            # img = cv2.imread(full_path)
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # h, s, v = cv2.split(hsv_img)
            # img_adap_mean = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)

            # image1 = np.concatenate((img_adap_mean, s, v), axis=1)
            # cv2.imshow('to_HSV', image1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # main
            full_path = folder_path + '/' + file
            img = cv2.imread(full_path)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            img_adap_mean = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)

            img_ASV = np.concatenate((img_adap_mean[:, :, np.newaxis], s[:, :, np.newaxis], v[:, :, np.newaxis]), axis=-1)
            # cv2.imshow('to_ASV', img_ASV)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            cv2.imwrite(full_path, img_ASV)
            
            
    print(f'{label_name} {len(files)} end')