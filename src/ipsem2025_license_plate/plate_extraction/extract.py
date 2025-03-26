import os

import cv2 as cv
import torch
from PIL import Image
from yolov5 import YOLOv5


def predict(input_folder, output_folder):
    model = YOLOv5("best.pt")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv.imread(image_path)

        if image is None:
            print(f"Nie można wczytać obrazu: {filename}")
            continue
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, corrected_plate)
        print(f"Przetworzono: {filename}, zapisano do {output_path}")
