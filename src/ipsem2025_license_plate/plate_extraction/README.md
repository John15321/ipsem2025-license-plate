# License Plate Extraction

This module provides tools for detecting and extracting license plates from images using YOLOv5.

## Using `ipsem2025-plate-recognizer` CLI Tool

The license plate detection functionality is available as a command-line tool through the `ipsem2025-plate-recognizer` command, which is an entry point to `detect.py`.

### Basic Usage

```bash
ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_SOURCE --device DEVICE
```

Where:
- `MODEL_PATH`: Path to your trained model weights (e.g., `./models/best.pt`)
- `INPUT_SOURCE`: Path to input image, video, or directory (e.g., `./example_imgs/car.jpg`)
- `DEVICE`: Computation device (`"cpu"` or CUDA device like `"0"`)

### Common Parameters

- `--weights MODEL_PATH`: Path to the model weights file (required)
- `--source INPUT_SOURCE`: Input source (image file, video, directory, webcam, etc.)
- `--device DEVICE`: Computation device ("cpu" or cuda device like "0" or "0,1,2,3")
- `--conf-thres CONFIDENCE`: Confidence threshold for detections (default: 0.25)
- `--iou-thres IOU`: IoU threshold for NMS (default: 0.45)

### Output Options

- `--save-txt`: Save results to *.txt files
- `--save-conf`: Save confidence scores in text files
- `--save-crop`: Save cropped license plate images
- `--view-img`: Display detection results in a window
- `--project OUTPUT_DIR`: Directory to save results (default: runs/detect)
- `--name EXP_NAME`: Name of the experiment (default: exp)

### Examples

1. **Basic detection with custom output directory:**
   ```bash
   ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_IMAGE --device DEVICE --project OUTPUT_DIR --name RUN_NAME
   ```

2. **Extract license plate crops:**
   ```bash
   ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_DIR --device DEVICE --save-crop
   ```

3. **Process all images in a directory with higher confidence threshold:**
   ```bash
   ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_DIR --device DEVICE --conf-thres 0.4
   ```

4. **Show results while processing:**
   ```bash
   ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_VIDEO --device DEVICE --view-img
   ```

5. **Save detection coordinates to text files:**
   ```bash
   ipsem2025-plate-recognizer --weights MODEL_PATH --source INPUT_DIR --device DEVICE --save-txt
   ```

### Output Structure

By default, results are saved to `OUTPUT_DIR/EXP_NAME` directory where:
- Images with detection visualizations are saved in the root of this directory
- If `--save-txt` is used, text files containing detections are in the `labels/` subdirectory
- If `--save-crop` is used, cropped license plates are in the `crops/license_plate/` subdirectory

### Example with Real Values

```bash
ipsem2025-plate-recognizer --weights ./models/best.pt --source ./example_imgs/car.jpg --device "cpu" --save-crop --conf-thres 0.5 --project ./results --name test_run1
```

## Using `extract.py`

The `extract.py` module provides a simplified interface for batch processing images to extract license plates.

**Note:** This functionality is currently being expanded.