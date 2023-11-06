import cv2
import os
import numpy as np

# base = './Cloud_Classification-4'

# for label_name in ['train', 'test', 'valid']:
#     print(f'{label_name} start')

#     folder_paths = base + '/' + label_name
#     folder_path = folder_paths + '/images'
#     files = os.listdir(folder_path)
#     # samples
#     for file in files[:2]:
#         if file.split('.')[-1] == 'jpg':
#             full_path = folder_path + '/' + file

#             img = cv2.imread(full_path)
#             hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             h, s, v = cv2.split(hsv_img)
#             image1 = np.concatenate((h, s, v), axis=1)
            
#             s_mask = cv2.inRange(s, 50, 256)
#             v_mask = cv2.inRange(v, 50, 256)

#             s_masked = s_mask*s
#             v_masked = v_mask*v

#             h = h[:, :, np.newaxis]
#             s_masked = s_masked[:, :, np.newaxis]
#             v_masked = v_masked[:, :, np.newaxis]
#             img_masked = np.concatenate((h, s_masked, v_masked), axis=-1)
#             img_masked = cv2.cvtColor(img_masked, cv2.COLOR_HSV2BGR)

#             image2 = np.concatenate((s_mask, v_mask), axis=1)

#             cv2.imshow('source', img)
#             cv2.imshow('to_HSV', image1)
#             cv2.imshow('SV_mask', image2)
#             cv2.imshow('SV_masked', img_masked)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

    # write
    # for file in files:
    #     if file.split('.')[-1] == 'jpg':
    #         full_path = folder_path + '/' + file

    #         img = cv2.imread(full_path)
    #         hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #         # h, s, vhsv_
    #         # image1 = np.concatenate((img, hsv_img), axis=1)

    #         # cv2.imshow('to_HSV', image1)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         cv2.imwrite(full_path, hsv_img)
            
    # print(f'{label_name} end')



# test
full_path = 'datasets/Cloud_Classification-14/train/images/1b5e7d1_jpg.rf.570c39209a4aadf436c383b65d50ea0b.jpg'
img = cv2.imread(full_path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)
image1 = np.concatenate((h, s, v), axis=1)

cv2.imshow('otsu', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()