import argparse
import os
import shutil
import sys
import logging  # Added logging
from tqdm import tqdm  # Added tqdm for progress bar

import cv2
import pytesseract

from .tools import FTYPE, STYPE, PlateExtractor, platePerspectiveUnwarpingWithWhite

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("generate_dataset_with_labels.log"),
    ],
)

def recognize_text(image_path):
    # Load the image
    logging.debug(f"Recognizing text from image: {image_path}")
    image = cv2.imread(image_path)

    # OCR - text recognition
    custom_config = r"--oem 3 --psm 10"  # Mode for single character
    text = pytesseract.image_to_string(image, config=custom_config)

    return text.strip()


def create_ocr_dataset(folder_path, output_folder):
    """
    Function to create an OCR dataset by moving images to a new folder
    and saving recognized characters in separate .txt files in the labels folder.
    """
    logging.info(f"Creating OCR dataset in folder: {output_folder}")
    dataset = []

    # Create the main ocr-dataset folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create images and labels folders within ocr-dataset
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    image_counter = 1  # Counter for file names (e.g., 1.png, 2.png, ...)

    # Iterate through all folders in the main folder
    for folder_name in tqdm(os.listdir(folder_path), desc="Processing folders"):
        folder_path_full = os.path.join(folder_path, folder_name)

        # Check if the folder contains .png files
        if os.path.isdir(folder_path_full):
            for filename in os.listdir(folder_path_full):
                if filename.endswith(".png"):
                    image_path = os.path.join(folder_path_full, filename)
                    logging.debug(f"Processing image: {image_path}")

                    # Recognize text from the image
                    recognized_text = recognize_text(image_path)

                    if recognized_text:
                        # logging.info(f"Recognized text: {recognized_text} - {image_counter}.png")
                        # Create a unique file name
                        new_filename = f"{image_counter}.png"
                        new_image_path = os.path.join(images_folder, new_filename)

                        # Move the image to the images folder
                        shutil.copy(image_path, new_image_path)

                        # Save the recognized text in a .txt file
                        label_filename = f"{image_counter}.txt"
                        label_path = os.path.join(labels_folder, label_filename)

                        with open(label_path, "w") as label_file:
                            label_file.write(recognized_text)

                        # Increment the counter for the next image
                        image_counter += 1

    logging.info(f"Dataset saved to {output_folder}")
    logging.info(f"Images moved to folder: {images_folder}")
    logging.info(f"Text files saved in folder: {labels_folder}")


def main():
    # Generating our instance
    extractor = PlateExtractor()

    logging.info("Starting OCR dataset generation process")
    default_input_path = "test-plates"  # Default folder with images
    default_output_path = "ocr-dataset"  # Default output folder

    # Command line arguments configuration
    parser = argparse.ArgumentParser(description="OCR Dataset Creator")
    parser.add_argument(
        "--input",
        type=str,
        default=default_input_path,
        help="Path to the input folder with images (default: 'test-plates')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output_path,
        help="Path to the output folder (default: 'ocr-dataset')",
    )

    # Parse arguments
    args = parser.parse_args()

    # Folder for perspective-corrected plates
    corrected_plates_folder = "CORRECTED_PLATES"
    if not os.path.exists(corrected_plates_folder):
        os.makedirs(corrected_plates_folder)
        logging.debug(f"Created folder: {corrected_plates_folder}")

    # First, apply perspective correction to the whole license plates
    for filename in tqdm(os.listdir(args.input), desc="Correcting plates"):
        file_path = os.path.join(args.input, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            logging.debug(f"Processing file: {file_path}")
            # Read the image
            image = cv2.imread(file_path)

            # Apply perspective correction to whole plate
            plate_xmin = 0
            plate_ymin = 0
            plate_ymax, plate_xmax = image.shape[:2]

            corrected_image = platePerspectiveUnwarpingWithWhite(
                image, plate_xmin, plate_ymin, plate_xmax, plate_ymax
            )

            # Save the corrected image
            corrected_path = os.path.join(corrected_plates_folder, filename)
            cv2.imwrite(corrected_path, corrected_image)

    logging.info("Extracting characters from license plates")
    extractor.apply_extraction_onpath(
        input_path=corrected_plates_folder, ftype=FTYPE.SINGLECHAR, stype=STYPE.BINARY
    )

    # Call the function to create the OCR dataset
    create_ocr_dataset("OUTPUT_SINGLE", args.output)

    # Get the script directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to folders to be deleted
    folder_paths = [
        "OUTPUT_BOX",
        "OUTPUT_SINGLE",
        corrected_plates_folder,
    ]

    for folder_path in folder_paths:
        # Check if the folder exists and delete it
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Folder '{folder_path}' has been deleted.")
        else:
            logging.warning(f"Folder '{folder_path}' does not exist.")
