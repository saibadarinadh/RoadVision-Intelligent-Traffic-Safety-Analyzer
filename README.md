# RoadVision - Intelligent Traffic Safety Analyzer

RoadVision is an intelligent traffic safety analyzer that processes videos to detect lanes, potholes, and people. It also uses computer vision techniques to reduce headlight glare, enhancing the visibility and accuracy of detections.

## Features

- **Lane Detection**: Detects lane markings in videos to assist in driving tasks and road safety.
- **Pothole Detection**: Identifies potholes to enhance road maintenance and user safety.
- **Person Detection**: Detects people in videos for applications in surveillance and safety.
- **Headlight Glare Reduction**: Uses computer vision to reduce headlight glare, improving the visibility of the road and objects.



## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RoadVision.git
    cd RoadVision
    ```

2. Install the required dependencies:
    ```sh
    pip install -r req.txt
    ```

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Upload a video on the upload page and wait for the processing to complete.

4. View the processed video with lane, pothole, and person detections, along with reduced headlight glare.

## File Descriptions

- **app.py**: The main Flask application file that handles video uploads, processing, and rendering.
- **best.pt, lane.pt, yolov8n.pt**: Pre-trained YOLO models for pothole, lane, and person detection respectively.
- **deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel**: Files related to the MobileNet-SSD model.
- **templates/**: Contains HTML templates for the home and upload pages.
- **uploads/**: Directory where uploaded videos are stored.
- **outputs/**: Directory where processed videos are saved.


## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov5)
- [Bootstrap](https://getbootstrap.com/)
