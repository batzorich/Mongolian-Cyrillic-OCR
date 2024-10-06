from scripts.contour_detection import detect_cntr
from scripts.line_segmentation import line_parse
from scripts.word_segmentation_and_prediction import word_parse

def predict(img_path):
    contour_images = detect_cntr(img_path)
    
    line_images_all = line_parse(contour_images)

    total_text_result = word_parse(line_images_all)

    return total_text_result
    