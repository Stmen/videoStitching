import cv2
import numpy as np


##指纹图像不需要透视变换等，只需要刚体变换
def fingerpoint_image_align(image_input_1,image_input_2):
    img2  = image_input_1
    img1  = image_input_2

    if img1 is None or img2 is None:
        print(' images is empty!')
        return None
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None) # 提取img1特征点
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None) # 提取img2特征点
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2) #匹配img1与img2的特征点
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7

    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m) #寻找最佳匹配特征对


    #-- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


    points1=np.zeros((len(good_matches),2),dtype=float)
    points2=np.zeros((len(good_matches),2),dtype=float)

    print("keypoints1 =", len(keypoints1))
    print("keypoints2 =", len(keypoints2))
    print("match points =",len(good_matches))
    if len(good_matches) < 3:
        return None
    for i in range(len(good_matches)):
        points1[i,:]=keypoints1[good_matches[i].queryIdx].pt
        points2[i,:]=keypoints2[good_matches[i].trainIdx].pt # 最佳匹配特征点位置

    ##刚体变换
    h, mask = cv2.estimateAffinePartial2D(points1, points2) #根据最佳匹配特征点生成将img1对齐到img2的几何变换矩阵

    height, width= img2.shape
    img_align = cv2.warpAffine(img1, h, (width, height)) # 根据前述生成的几何变换对img1进行变换

    img_align_concat = np.concatenate([img2, img_align], axis=1)
    return img_align_concat ,img_matches
    #img1Reg即为将img1对齐到img2的新图像

def fingerpoint_image_align_self(img1,img2):

    #check input
    if img1 is None or img2 is None:
        print(' images is empty!')
        return False

    #detect keypoints
    detector = cv2.SIFT_create()
    keypoints1,descriptors1 = detector.detectAndCompute(img1,None)
    keypoints2,descriptors2 = detector.detectAndCompute(img2,None)


    matcher = cv2.DescriptorMatcher_create("BruteForce")
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    ratio_thresh = 0.7
    matches_step_1 = []
    distance_match = False
    # 距离匹配跟RANSAC随机抽样一致性匹配有出来的对齐图像会有一些偏差
    # 距离匹配后匹配数量大幅度减少。随机抽样没有匹配数据了
    for m,n in knn_matches:
        if distance_match == True:
            if m.distance < ratio_thresh * n.distance:
                matches_step_1.append(m)
        else:
            matches_step_1.append(m)

    #刚体变换最少三个匹配点
    if len(matches_step_1) < 3:
        print("matches points = ",matches_step_1)
        return False

    points1=np.zeros((len(matches_step_1),2),dtype=float)
    points2=np.zeros((len(matches_step_1),2),dtype=float)

    for i in range(len(matches_step_1)):
        points1[i,:]=keypoints1[matches_step_1[i].queryIdx].pt
        points2[i,:]=keypoints2[matches_step_1[i].trainIdx].pt # 最佳匹配特征点位置

    #通过匹配点，找到刚体变换矩阵
    #RANSAC过滤掉离群点
    h, mask = cv2.estimateAffinePartial2D(points1, points2,method =cv2.RANSAC)
    height, width= img2.shape

    #将img2对img1进行对齐变换
    img2_align = cv2.warpAffine(img2, h, (width, height))

    #画出匹配图像特征点
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_step_1, img_matches,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img_concat = np.concatenate([img1, img2], axis=1)
    img_align  = np.concatenate([img1, img2_align], axis=1)

    return (img_concat,img_matches,img_align)










