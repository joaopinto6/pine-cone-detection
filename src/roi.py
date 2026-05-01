import cv2
import numpy as np

def canopy_roi_bgr(img_bgr, cfg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(cfg["roi"]["hsv_green_lower"], np.uint8)
    hi = np.array(cfg["roi"]["hsv_green_upper"], np.uint8)
    roi = cv2.inRange(hsv, lo, hi)

    k = cfg["roi"]["close_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    # after morphology
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        roi = np.where(labels==largest, 255, 0).astype("uint8")
    
    if cfg["roi"]["dilate_iter"] > 0:
        roi = cv2.dilate(roi, kernel, iterations=cfg["roi"]["dilate_iter"])

    return roi  # 255 on tree, 0 elsewhere
