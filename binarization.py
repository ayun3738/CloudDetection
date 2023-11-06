import cv2
from roboflow import Roboflow
import os
import numpy as np

# cloud_640s
# rf = Roboflow(api_key="odXiNgbNhpl15seknp9L")
# project = rf.workspace("alpaco-od").project("cloud_classification-7edfv")
# dataset = project.version(4).download("yolov8")

# base = './Cloud_Classification-4'
# for label_name in ['train', 'test', 'valid']:
#     print(f'{label_name} start')

#     folder_paths = base + '/' + label_name
#     folder_path = folder_paths + '/images'
#     files = os.listdir(folder_path)
#     for file in files[:6]:
#         if file.split('.')[-1] == 'jpg':
#             full_path = folder_path + '/' + file

#             img = cv2.imread(full_path, 0)

#             ret, img_bin50 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

#             # ret, img_bin100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#             ret, img_bin150 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
#             # # 하나의 이미지의 다수의 조명 상태 : 주변 밝기도 살펴보고 이진화
#             # img_adap_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
#             # img_adap_gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
#             # image1 = np.concatenate((img, img_bin, img_adap_mean, img_adap_gaus), axis=1)

#             # # 노이즈 제거 방법
#             # ret, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             # img_blur = cv2.GaussianBlur(img, (5,5), 0)
#             # ret, img_otsu_g = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             # image2 = np.concatenate((img, img_otsu, img_otsu_g), axis=1)

                
#             # 하나의 이미지의 다수의 조명 상태 : 주변 밝기도 살펴보고 이진화
#             img_adap_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
#             # ret, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[:, np.newaxis]
#             image1 = np.concatenate((img, img_bin50, img_adap_mean, img_bin150), axis=1)
#             img_bin50 = img_bin50[:, :, np.newaxis]
#             img_bin150 = img_bin150[:, :, np.newaxis]
#             img_adap_mean = img_adap_mean[:, :, np.newaxis]
#             # img_bin = np.concatenate((img_bin50, img_adap_mean, img_bin150), axis=-1)

#             # cv2.imshow('binary', img_bin)
#             cv2.imshow('otsu', image1)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             # cv2.imwrite(full_path, img_bin)

            
#     print(f'{label_name} end')


# test
full_path = 'Cloud_Classification-14/train/images/1b5e7d1_jpg.rf.570c39209a4aadf436c383b65d50ea0b.jpg'
img = cv2.imread(full_path, 0)

ret, img_bin50 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

# ret, img_bin100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
ret, img_bin150 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

img_adap_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
image1 = np.concatenate((img_bin50, img_adap_mean, img_bin150), axis=1)

cv2.imshow('otsu', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()