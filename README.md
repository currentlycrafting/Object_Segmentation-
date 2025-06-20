# Real-time Object Segmentation with YOLOv8-seg

---

This project demonstrates **real-time object instance segmentation** using the **Ultralytics YOLOv8-seg model** and **OpenCV**. It processes a video file, identifies objects, segments them with unique colors, and displays the results with bounding boxes and confidence scores.

## ✨ Features

* **Real-time Segmentation:** Processes video frames in real-time to perform object instance segmentation.
* **YOLOv8n-seg Model:** Utilizes the efficient `yolov8n-seg.pt` pre-trained model for fast and accurate segmentation.
* **Mask Visualization:** Overlays colored segmentation masks on detected objects, making individual instances clear.
* **Bounding Box and Labeling:** Draws bounding boxes and displays class labels with confidence scores for each detected object.
* **Configurable Input:** Easily change the input video file.

---

## 🛠️ Requirements

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* `ultralytics` library
* `opencv-python` library
* `numpy` library

You can install the necessary Python packages using pip:

```bash
pip install ultralytics opencv-python numpy

## 🚀 Setup & Usage

1.  Save the provided Python code as a `.py` file (e.g., `segmentation_app.py`).

2. Ensure your input video file (default: `park.mp4`) is in the same directory as the Python script. If your video is elsewhere, update the `input_video_path` variable in the script to point to its location:

    ```python
    input_video_path = "path/to/your/video.mp4"
    ```

3.  Run the script:
    Execute the Python script from your terminal:

    ```bash
    python segmentation_app.py
    ```

    A new window will open, displaying the video feed with segmented objects.

---
Customizing the Model

You can experiment with different YOLOv8 segmentation models (e.g., `yolov8s-seg.pt`, `yolov8m-seg.pt` for higher accuracy but potentially lower speed) by changing the `model_name` variable in the script:
model_name = 'yolov8m-seg.pt' # For a larger, more accurate model
