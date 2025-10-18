import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import random
from cv2 import resize
from glob import glob
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

img_height = 299
img_width = 299
train_ds = tf.keras.utils.image_dataset_from_directory(
  'SampleImages-resized',
  validation_split=0.2,
  subset='training',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'SampleImages-resized',
  validation_split=0.2,
  subset='validation',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)
  
class_names = train_ds.class_names
print(len(class_names))
print(class_names)
train_ds

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

img_height = 224
img_width = 224
base_model = tf.keras.applications.VGG16(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model = load_model("solar_panel_condition_model_1.keras")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")

model = load_model("solar_panel_condition_model.keras")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")

model = load_model("solar_panel_condition_model_finalversion_7.keras")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")

base_model.trainable = True

fine_tune_at = len(base_model.layers) - 40 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False  # keep first few frozen (low-level features)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoch = 12
history = model.fit(train_ds, validation_data=val_ds, epochs=epoch,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
             restore_best_weights=True,
            patience=4,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
)

history_df = pd.DataFrame(history.history)

history_df.to_csv("solar_defect_history_improving.csv", index=False)

model.save("solar_panel_condition_model_finalversion_improving.keras")
print("✅ Model saved successfully!")