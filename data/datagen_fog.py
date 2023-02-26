import cv2
import numpy as np
import argparse
import random
import os
import os.path as osp
from sklearn.feature_extraction import image
# import imutils
from tqdm import tqdm,trange

# argument parser
'''
dataNum:                  How many samples you want to synthesize
load_image_path:          Path to load background image
load_rain_path:           Path to load rain streak
load_depth_path:          Path to load depth information
save_input_image_path:    Path to save images with rain and haze
save_gt_image_path:       Path to save clean ground truth images
save_gtNohaze_image_path: Path to save no haze (rainy) images
save_gtNoRain_image_path: Path to save no rain (hazy) images
save_depth_path:          Path to save depth information
rainType:                 How many rain streaks you want to overlay on images
ang:                      Angle for random rotating [-ang:ang]
'''
def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_image_path", type=str, default="cam_stereo_left_lut/", help='path to load images')
    parser.add_argument("--load_depth_path", type=str, default="depth_image/", help='path to load depth info')
    parser.add_argument("--save_image_path", type=str, default="foggy_camera/", help='path to save ground truth images')
    parser.add_argument("--light_min", type=float, default=0.3)
    parser.add_argument("--light_max", type=float, default=0.8)
    parser.add_argument("--beta_min", type=float, default=1.3)
    parser.add_argument("--beta_max", type=float, default=1.3)
    parser.add_argument("--beta_range", type=float, default=0.3)
    parser.add_argument("--train_only", type=bool, default=False)
    parser.add_argument("--target_image_path", type=str, default="foggy_camera/", help='path to load images')
    opt = parser.parse_args()
    return opt



# depth to transmission formula
def depthToTransmission(depth, b_min, b_max):
    depth=depth/255.0
    beta = np.random.uniform(b_min, b_max)
    # print(beta)
    trans = np.exp(-beta * depth)
    return trans

def light_effect(img,airlight,night):
    if night==1:
        # rgb to gray
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # gaussian
        gray=cv2.GaussianBlur(gray,(21,21),0)
        # threshold
        light=gray>205
        brightness_0=light*((gray-205)/50.0)

        brightness_gau=cv2.GaussianBlur(brightness_0,(25,25),0)
        brightness=np.maximum(brightness_0,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(25,25),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(45,45),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(45,45),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(65,65),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(65,65),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(21,21),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(21,21),0)
        brightness=np.maximum(brightness,brightness_gau)

        brightness_gau=cv2.GaussianBlur(brightness,(21,21),0)
        brightness=np.maximum(brightness,brightness_gau)

        # adjust airlight
        atlight = airlight*np.ones(img.shape)[:,:,0]
        atlight = atlight+ (0.95-airlight)*brightness

        return cv2.merge([atlight,atlight,atlight]), cv2.merge([brightness_0,brightness_0,brightness_0]), cv2.merge([brightness,brightness,brightness])
    else:
        atlight = airlight*np.ones(img.shape)[:,:,0]

        return cv2.merge([atlight,atlight,atlight]), None, None


    

def add_fog(image,depth,trans,airLight,night):

    # trans = cv2.merge([trans, trans, trans])
    light, b0, b=light_effect(image, airLight, night)



    image = image / 255.0
    image = image.astype('float32')

    # start adding haze
    constant = np.ones(image.shape)

    hazyimage = image * trans + light * (constant - trans)

    return hazyimage, light, b0, b


def get_valid_list():
    spilt_path=f"splits/train_clear_day.txt"
    kitti_names = open(spilt_path,'r')
    kitti_names_contents = kitti_names.readlines()  
    valid_day=[]          
    for class_name in kitti_names_contents:
        valid_day.append(class_name.replace(",","_").rstrip()+'.png')
    kitti_names.close()

    spilt_path=f"splits/train_clear_night.txt"
    kitti_names = open(spilt_path,'r')
    kitti_names_contents = kitti_names.readlines()  
    valid_night=[]          
    for class_name in kitti_names_contents:
        valid_night.append(class_name.replace(",","_").rstrip()+'.png')
    kitti_names.close()

    return valid_day, valid_night
    
def main():
    opt = Parser()

    # check dirs exist or not
    if not os.path.isdir(opt.save_image_path):
        os.makedirs(opt.save_image_path)
    print(f'save image at {opt.save_image_path}')

    # load dir and count
    images_list = os.listdir(opt.load_image_path)
    datasize = len(images_list)

    valid_day_list, valid_night_list=get_valid_list()


    # start synthesizing loop
    for i in trange(datasize):
        file_name=images_list[i]

        if (file_name in valid_day_list) or (file_name in valid_night_list):
            if file_name in valid_night_list:
                night=1
            elif file_name in valid_day_list:
                night=0
            else:
                print("wrong")

            
            # load image/depth path
            image_path=osp.join(opt.load_image_path,file_name)
            depth_path=osp.join(opt.load_depth_path,file_name)

            # load image/depth
            image=cv2.imread(image_path)
            depth=cv2.imread(depth_path)

            # cv2.imwrite(osp.join(opt.save_image_path,'image.png'), image)
            # cv2.imwrite(osp.join(opt.save_image_path,'depth.png'), depth)


            # convert depth to transmission
            trans = depthToTransmission(depth, 1.0, 1.6)

            # cv2.imwrite(osp.join(opt.save_image_path,'trans.png'), trans*255)

            if night==0:
                airLight = np.random.uniform(0.4, 0.75)
            elif night==1:
                airLight = np.random.uniform(0.3, 0.65)

            # start adding
            hazyimage,light,b0,b=add_fog(image,depth,trans,airLight,night)

            # save
            cv2.imwrite(osp.join(opt.save_image_path,file_name), hazyimage*255)

        else:
            continue



if __name__ == "__main__":
    main()

