# from BB_UNet_UNet.Dataset_Helpers import class2one_hot, Make_CSV
import os


path = 'E:/alpaco/Cloud/Cloud_Classification-14'


for tt in ['train', 'valid', 'test']:
    txt_folder = os.path.join(path, tt, 'labels')
    txt_paths = os.listdir(txt_folder)
    for txt in txt_paths:
        txt_str = '' 
        txt = os.path.join(txt_folder, txt)
        with open(txt, 'r') as f:
            for readread in f.readlines():

                if readread.split(' ')[0] == '3':
                    readread = '2' + readread[1:]
                txt_str += readread
        with open(txt, 'w') as w:
            w.write(txt_str)
