import numpy as np
import cv2 as cv

# 讀取圖像
sourse_image = cv.imread('source.jpg')
target_image = cv.imread('target.jpg')
destination_image = cv.imread('destination.jpg')

# 轉成灰階
gray_source = cv.cvtColor(sourse_image, cv.COLOR_BGR2GRAY)
gray_target = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
gray_d = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)

# create SIFT
sift = cv.SIFT_create()

# SIFT檢測源圖和目標圖中的特徵點和描述子
kp_src, des_src = sift.detectAndCompute(gray_source, None)
kp_dst, des_dst = sift.detectAndCompute(gray_target, None)
kp_d, des_d = sift.detectAndCompute(gray_d, None)

# 用RANSAC找到最佳Homography Matrix
def find_best_homography(kps_src, des_src, kps_dst, des_dst, ransac_threshold=5.0, ransac_iterations=1000):
    # 用FLANN進行特徵點匹配
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des_src, des_dst, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None

    src_pts = np.float32([kps_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 用RANSAC計算Homography Matrix
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_threshold)
    return homography, mask

homography, mask = find_best_homography(kp_src, des_src, kp_dst, des_dst)

if homography is not None:
    # 計算目標圖的四個頂點座標
    h, w = destination_image.shape[:2]
    d_corners = np.float32([[0, 0], [0, h], [w, h], [w , 0]]).reshape(-1, 1, 2)

    # 使用透視變換計算目標圖在源圖四個頂點的位置
    src_corners = cv.perspectiveTransform(d_corners, np.linalg.inv(homography))

    # 使用透視變換對齊目標圖與源圖
    dst_aligned = cv.warpPerspective(target_image, np.linalg.inv(homography), (gray_source.shape[1], gray_source.shape[0]))
    d_aligned = cv.warpPerspective(destination_image, np.linalg.inv(homography), (gray_source.shape[1], gray_source.shape[0]))
    
    # 將目標圖貼到源圖上
    result = sourse_image.copy()
    cv.fillConvexPoly(result, np.int32(src_corners), (0, 0, 0))
    result = result + d_aligned

    # 輸出結果圖
    cv.imwrite('result.jpg', result)
else:
    print("Homography Matrix not found.")