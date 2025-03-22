# scripts

## Description

correct_perspective.py is a Python script that automatically unwarps perspective of images containing license plate.

generate_dataset_with_labels.py is a Python script that automatically processes images containing license plates, separates it into single letters, recognizes text using the Tesseract OCR library, and organizes them into a structured dataset. The script utilizes the `lpce` tool for license plate extraction. It moves images to a new folder and saves the recognized characters in corresponding text files.

## Requirements

Tesseract OCR must be installed and configured on your system. 

## Usage

correct_perspective.py cript can be executed from the command line, optionally specifying input and output folder paths:

```bash
python plate_unwarp.py <input_folder> <output_folder> --method <method>
```

- 'input_folder': Path to the folder containing images of license plates.
- 'output_folder': Path to save the processed images.
- --method: (Optional) Image processing method. Options:
    - white (default): Uses white region detection.
    - superpixel: Uses a superpixel-based approach.
    - default: Uses a generic unwarping method.

generate_dataset_with_labels.py script can be executed from the command line, optionally specifying input and output folder paths:

```bash
python generate_dataset_with_labels.py --input "path_to_input_folder" --output "path_to_output_folder"
```

If no paths are provided, the default values will be used:
- `test-plates` as the input folder
- `ocr-dataset` as the output folder

### Example Run:

```bash
python correct_perspective.py ./test-plates ./processed-plates --method white
```

```bash
python generate_dataset_with_labels.py --input "./test-plates" --output "./ocr_dataset"
```

## How the correct_perspective.py Script Works

1. Reads each image from the input folder.
2. Detects the license plate using contour analysis.
3. Applies the selected unwarping method.
4. Saves the corrected plate images to the output folder.
5. Logs processing results in the console.

## How the generate_dataset_with_labels.py Script Works

1. Uses `lpce` to extract license plate characters from images.
2. Scans the extracted character images for `.png` files.
3. Recognizes text on each image using `pytesseract`.
4. Copies images to the `images` folder within the output directory.
5. Creates `.txt` files in the `labels` folder, saving the recognized characters.
6. Deletes intermediate output folders (`OUTPUT_BOX` and `OUTPUT_SINGLE`) after processing.
7. Notifies the user upon completion and provides paths to the processed data.

