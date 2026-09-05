import cv2
from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26n.pt")

im1 = Image.open("bus.jpg")  # PIL image
results = model.predict(source=im1, save=True)  # save plotted images
