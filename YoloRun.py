from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch

#torch.serialization.add_safe_globals([DetectionModel])
model_yolo = YOLO(r"yolov8n.pt")

# Train with your annotated dataset

model_yolo.train(
    data="solar_panel_dataset.yaml",
    epochs=100,           # Train longer
    imgsz=640,
    batch=8,
    lr0=0.001,            # Higher learning rate
    patience=20,
    device="cpu",             # Use GPU if available
    optimizer="SGD",      # Default fine
    augment=True,         # Enable strong augmentations
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
)