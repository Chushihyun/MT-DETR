import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm,trange

weather=['train_clear','val_clear','test_clear','light_fog','dense_fog','snow']
time=['day','night']
sensor=['camera','lidar','radar']

src_dir='splits'
tar_dir='time_image'

if not os.path.exists(tar_dir):
    os.makedirs(tar_dir)


for w in weather:
    for t in time:
        txt_path=f'{src_dir}/{w}_{t}.txt'
        f = open(txt_path)
        lines = f.readlines()
        for line in tqdm(lines):
            img_path=line.replace(",","_").rstrip()
            img_path=f'{img_path}.png'
            # print(img_path)
            tar_path=osp.join(tar_dir,img_path)

            if t == 'day':
                img=np.ones((1024,1920,3))*255
            elif t == 'night':
                img=np.zeros((1024,1920,3))
            else:
                print("error")

            cv2.imwrite(tar_path, img)
            # print(tar_path)

