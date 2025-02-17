import torch
import cv2
import numpy as np
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to filter out text detections
def filter_detections(detections):
    filtered_detections = []
    for det in detections:
        if det['name'] not in ['text', 'handwriting']:
            filtered_detections.append(det)
    return filtered_detections

# Directories
input_folder = 'voynich_pages'
output_folder = 'results'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Filter out text detections
    filtered_results = filter_detections(results.pandas().xyxy[0].to_dict(orient="records"))

    # Draw bounding boxes on the image
    for det in filtered_results:
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        label = det['name']
        confidence = det['confidence']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image
    result_image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(result_image_path, image)

    # Optionally display the result image
    # cv2.imshow('Object Detection', image)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()