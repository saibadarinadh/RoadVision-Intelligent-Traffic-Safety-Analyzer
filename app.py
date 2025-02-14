from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from flask import Response
app = Flask(__name__)

# Configure upload and output folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the YOLO models
model_lane = YOLO('lane.pt')  # Model for lane detection
model_pothole = YOLO('best.pt')  # Model for pothole detection
model_person = YOLO('yolov8n.pt')  # Model for person detection

# Function to reduce headlight glare
def reduce_headlight_glare(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blur = cv2.GaussianBlur(v, (15, 15), 0)
    mask = cv2.inRange(blur, 200, 255)
    v[mask != 0] = v[mask != 0] - 150
    hsv_modified = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

# Function to mask out the region of interest for lane detection
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)

# Function to draw the filled polygon between the lane lines
def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    cv2.fillPoly(line_img, poly_pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)

# The lane detection pipeline
def lane_detection_pipeline(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    left_line_x, left_line_y, right_line_x, right_line_y = [], [], [], []

    if lines is None:
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(height * (3 / 5))
    max_y = height

    left_x_start = left_x_end = 0
    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

    right_x_start = right_x_end = 0
    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

    return draw_lane_lines(
        image,
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    )

# Function to detect multiple vehicles in the video and return their positions and widths
def detect_vehicles(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = (0, 120, 70)
    upper_red = (10, 255, 255)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    car_positions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:
            car_positions.append((x, w))

    return car_positions

# Function to update squares in the grid based on detected vehicles' positions and widths
def update_squares(frame, car_positions):
    frame_with_squares = frame.copy()
    for car_x, car_width in car_positions:
        cv2.rectangle(frame_with_squares, (car_x, 0), (car_x + car_width, 60), (0, 255, 0), 2)
    return frame_with_squares




def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for uniform processing
        frame_resized = cv2.resize(frame, (1280, 720))

        # Process the frame
        frame_glare_reduced = reduce_headlight_glare(frame_resized)
        lane_frame = lane_detection_pipeline(frame_glare_reduced)
        
        # Run pothole detection
        result_pothole = model_pothole(frame_glare_reduced, stream=True)
        for info in result_pothole:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                if confidence > 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(lane_frame, f'Pothole {confidence}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Run person detection
        result_person = model_person(frame_glare_reduced, stream=True)
        for info in result_person:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                if confidence > 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(lane_frame, f'Person {confidence}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update vehicle detection
        car_positions = detect_vehicles(frame_glare_reduced)
        lane_frame_with_squares = update_squares(lane_frame, car_positions)

        # Convert the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', lane_frame_with_squares)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(process_video(os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# def process_video(input_video_path):
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return None

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
#     output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize frame for uniform processing
#         frame_resized = cv2.resize(frame, (1280, 720))

#         # Step 1: Reduce headlight glare
#         frame_glare_reduced = reduce_headlight_glare(frame_resized)

#         # Step 2: Run lane detection pipeline
#         lane_frame = lane_detection_pipeline(frame_glare_reduced)

#         # Step 3: Run the YOLO model for pothole detection
#         result_pothole = model_pothole(frame_glare_reduced, stream=True)

#         # Draw bounding boxes and labels for pothole detection
#         for info in result_pothole:
#             boxes = info.boxes
#             for box in boxes:
#                 confidence = box.conf[0]
#                 confidence = math.ceil(confidence * 100)
#                 Class = int(box.cls[0])

#                 if confidence > 50:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(lane_frame, f'Pothole {confidence}%', (x1, y1 - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Step 4: Run the YOLO model for person detection
#         result_person = model_person(frame_glare_reduced, stream=True)
        
#         for info in result_person:
#             boxes = info.boxes
#             for box in boxes:
#                 confidence = box.conf[0]
#                 confidence = math.ceil(confidence * 100)
#                 Class = int(box.cls[0])

#                 if confidence > 50:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(lane_frame, f'Person {confidence}%', (x1, y1 - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Step 5: Detect vehicles in the frame
#         car_positions = detect_vehicles(frame_glare_reduced)

#         # Step 6: Update squares for detected vehicles
#         lane_frame_with_squares = update_squares(lane_frame, car_positions)

#         # Write the processed frame to the output video
#         output_video.write(lane_frame_with_squares)

#     cap.release()
#     output_video.release()
#     cv2.destroyAllWindows()
#     return output_video_path
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(request.url)

    if video_file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
        video_file.save(video_path)
        return redirect(url_for('video_feed'))


if __name__ == '__main__':
    app.run(debug=True)
