import cv2

import numpy as np
import os
import sys

import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=2,
        showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA,"bgr")
        (kpsB, featuresB) = self.detectAndDescribe(imageB,"bgr")

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

   def detectAndDescribe(self,image,mode) :
        # convert the image to grayscale
        if mode == "gray":
            gray = image
        elif mode == "bgr":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detecter  = cv2.SIFT_create()
            kps = detecter.detect(gray)

            # extract features from the image
            #extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = detecter.compute(gray,kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        #kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            #ptsA = np.float32([kpsA[i] for (_, i) in matches])
            #ptsB = np.float32([kpsB[i] for (i, _) in matches])

            ptsA = [kpsA[i] for (_, i) in matches]
            ptsB = [kpsB[i] for (i, _) in matches]

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

#work_dir = os.getcwd()
base_dir = ""
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
    orb = cv2.ORB_create()
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
    result, vis = stitcher.stitch([fingerpoint1,fingerpoint2],showMatches=False)
def get_football_img_uint8(path):
    return cv2.imread(path)

def test4():
    football_left   = get_football_img_uint8(data_dir_football + "/Left_0001.jpg")
    football2_right = get_football_img_uint8(data_dir_football + "/Right_0001.jpg")
    stitcher = Stitcher()
    result, vis = stitcher.stitch([football_left,football2_right],showMatches=True)

def test5():
        football_left = get_football_img_uint8(data_dir_football + "/Left_0001.jpg")
        football2_right = get_football_img_uint8(data_dir_football + "/Right_0001.jpg")
        orb = cv2.ORB_create()
        kp1 = orb.detect(football_left)
        kp2 = orb.detect(football2_right)

        kp1, des1 = orb.compute(football_left, kp1)
        kp2, des2 = orb.compute(football2_right, kp2)

        outimg1 = cv2.drawKeypoints(football_left, keypoints=kp1, outImage=None)
        outimg2 = cv2.drawKeypoints(football2_right, keypoints=kp2, outImage=None)
        outimg3 = np.hstack([outimg1, outimg2])
        cv2.imwrite("../data/football/test.png", outimg3)
def main( args):
    print(cv2.__version__)
    test5()


if __name__ == "__main__" :
    main(sys.argv[ 1:])



