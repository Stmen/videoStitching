import cv2
import numpy as np
import os
import sys
from  panorama import Stitcher

#work_dir = os.getcwd()
base_dir = ".."
data_dir = base_dir + "/data"
data_dir_fingerpoint = data_dir + "/fingerpoint"
data_dir_football = data_dir + "/football"

def conver_debase_to_img():
    data_lists= os.listdir(data_dir_fingerpoint)
    for list in data_lists:
        if  not list.endswith(".debase"):
            continue

        abs_path = data_dir_fingerpoint + "/" + list
        name = list.rstrip(".debase")

        commond = "python imageconv.py  -v  --force  --normalize 0 --width 160 --height 160  --input-type u32 --output-type png --path "+ data_dir_fingerpoint  + " " + abs_path
        print(commond)
        code = os.system(commond)
def normailzation(data):
    data = data.astype(np.float32)
    max = np.max(data)
    min = np.min(data)
    range = max - min
    return (data - min) / range  *255

def test():
    data_lists = os.listdir(data_dir_fingerpoint)
    for list in data_lists :
        if  not list.endswith(".debase"):
            continue
        name = list.rstrip(".debase")
        name = data_dir_fingerpoint + "/" + name
        abs_path = data_dir_fingerpoint + "/" + list
        debase = np.fromfile(abs_path,dtype=np.uint32)
        print(debase)
        print(type(debase))
        debase = debase.reshape(160,160)
        h,w = debase.shape
        debase = debase.astype(np.float32)
        #debase_uint8 = debase / 256
        debase_uint8 = normailzation(debase)
        debase_uint8 = debase_uint8.astype(np.uint8)
        #debase_img = cv2.Mat(h,w,debase_uint8)

        cv2.imwrite(name + ".jpg",debase_uint8)

        #print(debase.dtype)
        #print(debase)
        #print(type(debase))
        #img = cv2.Mat(debase)
        #print(img)

def get_finger_img_uint8(path,mode):
    debase = np.fromfile(path, dtype=np.uint32)
    debase = debase.reshape(160, 160)
    if mode == "nor" :
        debase_uint8 = normailzation(debase)
    elif mode == "cut":
        debase_uint8 = debase / 256
    elif mode ==  "zou":
        name = path
        debase_uint8 = cv2.imread(name + ".png")
    debase_uint8 = debase_uint8.astype(np.uint8)
    return debase_uint8
def test2():
    fingerpoint1 = get_finger_img_uint8(data_dir_fingerpoint + "/fingerpoint3.debase","nor")
    fingerpoint2 = get_finger_img_uint8(data_dir_fingerpoint + "/fingerpoint4.debase","nor")
    orb = cv2.SIFT_create()
    kp1 = orb.detect(fingerpoint1)
    kp2 = orb.detect(fingerpoint2)

    kp1 ,des1 = orb.compute(fingerpoint1,kp1)
    kp2, des2 = orb.compute(fingerpoint2,kp2)

    outimg1 = cv2.drawKeypoints(fingerpoint1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(fingerpoint2, keypoints=kp2, outImage=None)
    outimg3 = np.hstack([outimg1, outimg2])
    cv2.imwrite("../data/fingerpoint/normailzation.png", outimg3)
def test3():
    fingerpoint1 = get_finger_img_uint8(data_dir_fingerpoint + "/fingerpoint4.debase","nor")
    fingerpoint2 = get_finger_img_uint8(data_dir_fingerpoint + "/fingerpoint4.debase","nor")
    stitcher = Stitcher()
    result = stitcher.stitch(fingerpoint1,fingerpoint2)
    cv2.imwrite(data_dir_fingerpoint + "/stitch.png",result)
def get_football_img_uint8(path):
    return cv2.imread(path)

def test4():
    football_left   = get_football_img_uint8(data_dir_football + "/Left_0001.jpg")
    football2_right = get_football_img_uint8(data_dir_football + "/Right_0001.jpg")
    stitcher = Stitcher()
    result = stitcher.stitch([football_left,football2_right])

def test5():
        football_left = get_football_img_uint8(data_dir_football + "/Left_0001.jpg")
        football2_right = get_football_img_uint8(data_dir_football + "/Right_0001.jpg")
        #orb = cv2.SIFT_create()
        orb = cv2.ORB_create()
        kp1 = orb.detect(football_left)
        kp2 = orb.detect(football2_right)

        kp1, des1 = orb.compute(football_left, kp1)
        kp2, des2 = orb.compute(football2_right, kp2)

        outimg1 = cv2.drawKeypoints(football_left, keypoints=kp1, outImage=None)
        outimg2 = cv2.drawKeypoints(football2_right, keypoints=kp2, outImage=None)
        outimg3 = np.hstack([outimg1, outimg2])
        cv2.imwrite("../data/football/test.png", outimg3)
def test6():
    football_left   = get_football_img_uint8(data_dir_football + "/Left_0001.jpg")
    football2_right = get_football_img_uint8(data_dir_football + "/Right_0001.jpg")
    fingerpoint1 = get_football_img_uint8(data_dir_fingerpoint + "/../test/test3.jpg")
    fingerpoint2 = get_football_img_uint8(data_dir_fingerpoint + "/../test/test4.jpg")
    stitcher = Stitcher()
    result = stitcher.stitch(fingerpoint1,fingerpoint2)
    cv2.imwrite(data_dir_fingerpoint + "/../test/stitch.png",result)
    result = stitcher.stitch(football_left,football2_right)
    cv2.imwrite(data_dir_fingerpoint + "/football/stitch.png",result)
def main( args):
    print(cv2.__version__)
    test6()


if __name__ == "__main__" :
    main(sys.argv[ 1:])



