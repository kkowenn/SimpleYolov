import cv2
from ultralytics import YOLOv10  

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
cap = cv2.VideoCapture('trafficvideo.mov')  # Replace video file

# Define codec and create VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

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

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display the label and confidence score
            cv2.putText(frame, f'{label_name} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the processed frame to the output video file
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
