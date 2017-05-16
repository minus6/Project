# Zeiss
# Image pre-processing

import glob
import numpy as np
from PIL import Image
from pytesseract import image_to_string
import cv2

scaling = {}

for file in glob.glob('images/*.jpg'):
    image = cv2.imread(file) # read image
    width = image.shape[1]
    image = image[:, width / 2: width] # crop the right half of the image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) # create kernel
    dilated = cv2.dilate(thresh, kernel, iterations = 10) # dilate

    # cv2.imshow('dilated', cv2.resize(dilated, (683, 520)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours

    # for each contour found
    for contour in contours:
        # get rectangle bounding contour
        x, y, w, h = cv2.boundingRect(contour)

        # discard areas that are too large
        if h > 110 or w > 1000:
            continue

        # discard areas that are too small
        if h < 50 or w < 50:
            continue

        label = image_to_string(Image.fromarray(image[y: y + h, x: x + w])) # OCR

        try:
            scale = int(filter(str.isdigit, label)) # extract scale and convert to int
            scaling[file] = scale # add scale to dictionary
        except ValueError:
            print("could not extract the scale dammit :(")
            scaling[file] = 0

        mask = np.zeros(image.shape[:2], np.uint8) # construct mask to be used for inpainting
        mask[y: y + h, x: x + w] = 255

        new = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) # inpaint and create new image

        cv2.imwrite(file[:-4] + '_new.jpg', new) # write new image to disc

print scaling  # print scales dictionary
