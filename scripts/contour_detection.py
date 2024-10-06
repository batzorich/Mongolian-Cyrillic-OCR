import cv2
from scripts.preprocess import apply_preprocess
import numpy as np

def get_contour_precedence(contour, cols):
    tolerance_factor = 61
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def sort_contours_by_y(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]    
    sorted_indices = sorted(range(len(bounding_boxes)), key=lambda k: bounding_boxes[k][1])
    
    return [contours[i] for i in sorted_indices]

def detect_cntr(img_path):
    image = apply_preprocess(img_path)

    ret,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    contours, h = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    image_area = image.shape[0] * image.shape[1]
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    largest_contour_area = w * h
    threshold_area = 0.95 * image_area
    if largest_contour_area >= threshold_area:
        contours = tuple(x for x in contours if x is not largest_contour)

    min_area = 1000
    contours_sorted = sort_contours_by_y(contours)

    filtered_contours = [contour for contour in contours_sorted if min_area <= cv2.contourArea(contour)]

    cropped_images = []
    for i, ctr in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(ctr)
        cropped_img = image[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
    return cropped_images
