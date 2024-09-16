import cv2
import time
from ultralytics import YOLOv10
import math

custom_labels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

# Load the pre-trained YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

# Open the video file
cap = cv2.VideoCapture('trafficvideo.mov')  # Replace with your video file

# Define codec and create VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_speed.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Variables to track previous positions and time for speed calculation
previous_positions = {}  # Store previous positions for objects (key: label + id)
previous_time = None      # Store the time of the previous frame

def calculate_speed(prev_pos, curr_pos, time_diff):
    """Calculate speed based on position change and time difference."""
    if time_diff == 0:
        return 0
    x1, y1 = prev_pos
    x2, y2 = curr_pos
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Calculate pixel distance
    speed = distance / time_diff  # Speed in pixels per second
    return speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Capture the current time for speed calculation
    current_time = time.time()

    # Perform object detection on the current frame
    results = model.predict(source=frame, save=False)

    # YOLOv10 returns a list of predictions, so we iterate through the list
    for result in results:
        boxes = result.boxes.xyxy  # Access the bounding boxes (xmin, ymin, xmax, ymax)
        confs = result.boxes.conf  # Access the confidence scores
        labels = result.boxes.cls  # Access the class IDs

        # Iterate over the detected boxes and draw them
        for i, (box, conf, label) in enumerate(zip(boxes, confs, labels)):
            xmin, ymin, xmax, ymax = map(int, box)

            # Get the label name from the custom_labels dictionary
            label_name = custom_labels.get(int(label), 'Unknown')

            # Calculate the center of the bounding box (for displacement calculation)
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            current_position = (center_x, center_y)

            # Calculate time difference (delta time) between frames
            if previous_time is not None:
                time_diff = current_time - previous_time
            else:
                time_diff = 0

            # If the object has been detected previously, calculate speed
            obj_id = f"{label_name}_{i}"  # Unique identifier for the object
            if obj_id in previous_positions:
                prev_position = previous_positions[obj_id]
                speed = calculate_speed(prev_position, current_position, time_diff)
            else:
                speed = 0

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display the label, confidence score, and speed
            cv2.putText(frame, f'{label_name} {conf:.2f} Speed: {speed:.2f} px/s',
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Store the current position for the next frame
            previous_positions[obj_id] = current_position

    # Write the processed frame to the output video file
    out.write(frame)

    # Update previous time for the next frame
    previous_time = current_time

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
