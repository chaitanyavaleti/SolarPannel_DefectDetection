import tensorflow as tf
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

model = load_model("solar_panel_condition_model_finalversion_improving.keras")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")