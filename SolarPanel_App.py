import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.utils import custom_object_scope
from ultralytics import YOLO
import cv2
from keras.layers import Lambda

# --- App Title ---
st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🔆 Solar Panel Condition Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image of a solar panel to classify its condition.")

# --- Load model (cached for performance) ---
@st.cache_resource
def load_models():
    with custom_object_scope({'SlicingOpLambda': Lambda}):
        classifier = tf.keras.models.load_model("solar_panel_condition_model_80.keras")
    detector = YOLO("best.pt")
    return classifier, detector

model, detector = load_models()

# --- Define your class names (must match training) ---
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Display uploaded image
    img_height, img_width = 299,299
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_height, img_width))

    # --- Preprocess image ---
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 

   # Columns layout
    col1, col2 = st.columns([1, 2])  # left column smaller, right column bigger
    
    # Show original image in smaller column
    with col1:
        st.image(image, caption="Original Image", width=200)  # shrink size
    
    # Show prediction results in the larger column
    with col2:
        logits = model.predict(img_array)
        predictions = tf.nn.softmax(logits).numpy()
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        
        st.subheader("🧭 Prediction Results")
        st.markdown(f"**Predicted Condition:** {predicted_class}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Optional: display probabilities as a table
        import pandas as pd
        df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": (predictions[0]*100).round(2)
        }).sort_values(by="Probability (%)", ascending=False)
        st.table(df)


    # --- Maintenance Recommendation ---
    st.subheader("🧰 Maintenance Recommendation")
    recommendations = {
        "Clean": "✅ Panel is clean. No action needed.",
        "Dusty": "🧹 Light cleaning recommended to maintain efficiency.",
        "Bird-drop": "🐦 Clean droppings to avoid shading and hotspots.",
        "Electrical-damage": "⚡ Electrical damage detected! Schedule maintenance immediately.",
        "Physical-Damage": "🧩 Physical damage detected! Replace or repair panel.",
        "Snow-Covered": "❄️ Remove snow to restore performance."
    }
    st.info(recommendations.get(predicted_class, "No recommendation available."))


    # --- YOLO Obstruction Detection (optional) ---
    st.subheader("🧩 Obstruction Detection (YOLOv8)")
    resized_img = image.resize((640, 640), resample=Image.Resampling.BILINEAR)


    with st.spinner("Running object detection..."):
        resized_array = np.array(resized_img)
        results = detector.predict(resized_array, conf=0.001, iou=0.3,  show=False, verbose=True)
        #results[0].show()

    print("Number of detections:", len(results[0].boxes))
    boxes = results[0].boxes
    result_img = resized_array.copy()

    # Filter boxes for only the target class
    filtered_indices = [i for i in range(len(boxes)) if detector.names[int(boxes.cls[i])] == predicted_class]

    if len(filtered_indices) > 0:
        st.write(f"✅ {len(filtered_indices)} '{predicted_class}' Detections Found:")
        for i in filtered_indices:
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            label = detector.names[cls_id]
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            color = (0, 255, 0)  # green box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            #st.write(f"- **{label}** ({conf:.2f}) — Box: {xyxy}")

        st.image(result_img, caption="Detected Obstructions", use_container_width=True)
    else:
        st.write(f"✅ No '{predicted_class}' detected.")