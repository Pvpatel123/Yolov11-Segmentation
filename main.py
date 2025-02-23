import cv2
import numpy as np
import matplotlib.pyplot as plt
from YOLO11Segmentation import YOLOv11Segmentation

# Initialize the YOLOv11 segmentation model
model = YOLOv11Segmentation("yolo11n-seg.pt")

# Load an image
image_path = "/home/parth/Desktop/Project/Frustum/KITTI_DATASET/training/image_2/001000.png"  # Change to actual path
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Error loading image from path: {image_path}")

# Perform segmentation using the correct method
results = model.segment(image)

# Loop through results and apply segmentation mask
for result in results:
    annoted_frame = result.plot()  # Overlay segmentation mask

# Convert BGR to RGB for visualization
annoted_frame = cv2.cvtColor(annoted_frame, cv2.COLOR_BGR2RGB)

# Display the segmented image
plt.figure(figsize=(10, 6))
plt.imshow(annoted_frame)
plt.axis("off")
plt.show()

