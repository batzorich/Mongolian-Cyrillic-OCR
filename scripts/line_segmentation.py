import cv2
import numpy as np

def line_segmentation_horizontal_projection(gray_image):
    _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    horizontal_projection = np.sum(binary, axis=1)

    threshold = (np.max(horizontal_projection) - np.min(horizontal_projection)) / 100
    lines = []
    start = None

    for i, value in enumerate(horizontal_projection):
        if value > threshold and start is None:  # Start of a line
            start = i
        elif value <= threshold and start is not None:  # End of a line
            end = i
            lines.append((start, end))
            start = None

    if start is not None:
        lines.append((start, len(horizontal_projection)))

    line_images = []
    for (start, end) in lines:
        x, y, w, h = 0, int(start), int(gray_image.shape[1]), int(end-start)
        line_images.append(gray_image[y:y+h, x:x+w])

    return line_images

def line_parse(contour_images):
    line_images_all = []
    for i in range(len(contour_images)):
        line_images = line_segmentation_horizontal_projection(contour_images[i])
        if line_images != None:
            for single_line in line_images:
                if single_line.shape[0] > 10: #change
                    line_images_all.append(single_line)
    return line_images_all