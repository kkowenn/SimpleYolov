from ultralytics import YOLOv10

# Load the pre-trained YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

# Specify the source image URL
source = 'cars.png'

# Perform inference on the image and save the result
model.predict(source=source, save=True)
