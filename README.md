Real-time Object Segmentation with YOLOv8-seg
This project demonstrates real-time object instance segmentation using the Ultralytics YOLOv8-seg model and OpenCV. It processes a video file, identifies objects, segments them with unique colors, and displays the results with bounding boxes and confidence scores.

✨ Features
Real-time Segmentation: Processes video frames in real-time to perform object instance segmentation.
YOLOv8n-seg Model: Utilizes the efficient yolov8n-seg.pt pre-trained model for fast and accurate segmentation.
Mask Visualization: Overlays colored segmentation masks on detected objects, making individual instances clear.
Bounding Box and Labeling: Draws bounding boxes and displays class labels with confidence scores for each detected object.
Configurable Input: Easily change the input video file.
🛠️ Requirements
Before you begin, ensure you have the following installed:

Python 3.8+
ultralytics library
opencv-python library
numpy library
You can install the necessary Python packages using pip:

Bash

pip install ultralytics opencv-python numpy
🚀 Setup & Usage
Save the code:
Save the provided Python code as a .py file (e.g., segmentation_app.py).

Place your input video:
Ensure your input video file (default: park.mp4) is in the same directory as the Python script. If your video is elsewhere, update the input_video_path variable in the script to point to its location:

Python

input_video_path = "path/to/your/video.mp4"
Run the script:
Execute the Python script from your terminal:

Bash

python segmentation_app.py
A new window will open, displaying the video feed with segmented objects.

Quit the application:
Press the q key to close the display window and exit the application.

⚙️ Customizing the Model
You can experiment with different YOLOv8 segmentation models (e.g., yolov8s-seg.pt, yolov8m-seg.pt for higher accuracy but potentially lower speed) by changing the model_name variable in the script:

Python

model_name = 'yolov8m-seg.pt' # For a larger, more accurate model
💡 How it Works
The script leverages the following components:

ultralytics.YOLO: Initializes a YOLOv8-seg model, pre-trained to detect and segment a wide variety of objects.
cv2.VideoCapture: Reads frames sequentially from the specified input video file.
Model Inference: For each frame, model(frame) performs object detection and instance segmentation, returning results that include detected objects, their bounding boxes, and crucial segmentation masks.
Mask Processing:
Raw masks are extracted from results[0].masks.data.
These masks are resized using cv2.resize to match the original frame's dimensions.
A boolean mask is created to precisely identify the pixels belonging to each segmented object.
A unique, random color is generated for each detected object instance.
The output_frame is then updated by blending the generated color with the original frame pixels, but only where the boolean mask is True, creating a semi-transparent colored overlay.
Bounding Box and Labeling: Bounding box coordinates, class labels, and confidence scores are extracted and drawn onto the output_frame using OpenCV functions, providing clear visual identification of each object.
Real-time Display: The cv2.imshow function continuously displays the processed frames, providing a real-time visualization of the segmentation results.
🤝 Contributing
Feel free to fork this repository, suggest improvements, or open issues. Contributions are always welcome!
