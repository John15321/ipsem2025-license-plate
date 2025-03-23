import argparse
import os

import cv2 as cv
import numpy as np
from tools import (
    platePerspectiveUnwarping,
    platePerspectiveUnwarpingWithSuperPixel,
    platePerspectiveUnwarpingWithWhite,
)


def process_images(input_folder, output_folder, method="white"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv.imread(image_path)

        if image is None:
            print(f"Nie można wczytać obrazu: {filename}")
            continue

        plate_xmin = 0
        plate_ymin = 0
        plate_ymax, plate_xmax = image.shape[:2]

        if method == "white":
            corrected_plate = platePerspectiveUnwarpingWithWhite(
                image, plate_xmin, plate_ymin, plate_xmax, plate_ymax
            )
        elif method == "superpixel":
            corrected_plate = platePerspectiveUnwarpingWithSuperPixel(
                image, plate_xmin, plate_ymin, plate_xmax, plate_ymax
            )
        else:
            corrected_plate = platePerspectiveUnwarping(
                image, plate_xmin, plate_ymin, plate_xmax, plate_ymax
            )

        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, corrected_plate)
        print(f"Przetworzono: {filename}, zapisano do {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Przetwarzanie obrazów tablic rejestracyjnych."
    )
    parser.add_argument(
        "input_folder", type=str, help="Ścieżka do folderu wejściowego z obrazami."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Ścieżka do folderu wyjściowego dla przetworzonych obrazów.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["white", "superpixel", "default"],
        default="white",
        help="Metoda przetwarzania obrazu.",
    )

    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder, method=args.method)
