import cv2
import numpy as np
from scripts.mn_ocr_paddle.PaddleOCR.tools.infer.predict_rec import predict_custom

def convert_gray_to_rgb(gray_image):
    rgb_image = np.zeros_like(gray_image[:, :, np.newaxis]) * 255
    
    rgb_image = np.stack([gray_image] * 3, axis=-1)
    
    return rgb_image

# Apply preprocessing that is not done in skew correction
def apply_preprocess(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

def thresholding(image, inv=True):
    if inv:
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return thresh

def cropp_line(line_image):
    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)
    height = line_image.shape[0]
    index = 0       
    while vertical_projection[index] == height:
        index += 1
    if index > 2:
        index -= 2
    line_image = line_image[:, index:]
    
    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)
    
    index = line_image.shape[1] - 1
    while vertical_projection[index] == height:
        index -= 1
    if index < line_image.shape[1] - 1:
        index += 2
    
    line_image = line_image[:, :index]
    return line_image

def word_segmentation(line_image, display_result=False):
    line = cropp_line(line_image)
    dst = cv2.fastNlMeansDenoising(line, None, 12, 7, 21)
    thresh = thresholding(dst)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1) 
    binary = line > dilated
    vertical_projection = np.sum(binary, axis=0)
    
    height = line.shape[0]
    whitespace_lengths = []
    current_whitespace = 0
    for vp in vertical_projection:
        if vp == height:
            current_whitespace += 1
        elif current_whitespace:
            whitespace_lengths.append(current_whitespace)
            current_whitespace = 0

    if current_whitespace:
        whitespace_lengths.append(current_whitespace)

    avg_white_space_length = np.mean(whitespace_lengths)*1.5 if whitespace_lengths else 0
    
    divider_indexes = [0]
    current_whitespace = 0
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            current_whitespace += 1
        else:
            if current_whitespace > avg_white_space_length:
                divider_indexes.append(index - current_whitespace // 2)
            current_whitespace = 0

    if display_result:
        line_copy = line.copy()
        mask = np.zeros_like(line_copy, dtype=bool)
        mask[:, divider_indexes] = True
        line_copy[mask] = 0
    
    divider_indexes.append(len(vertical_projection))

    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))
    
    words = [line[:, window[0]:window[1]] for window in dividers]

    return words

def word_parse(line_images_all):
    
    total_text_result = []
    for i in range(len(line_images_all)):
        words = word_segmentation(line_images_all[i], True)
        line_text = ""
        for word in words:
            rgb_word = convert_gray_to_rgb(word)
            line_text = line_text + (predict_custom(rgb_word, "./scripts/mn_ocr_paddle/trained_inference/", False)) + " "
        total_text_result.append(line_text)
    
    return total_text_result