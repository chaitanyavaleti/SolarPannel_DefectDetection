# Solar Panel Defect Detection App

This Streamlit application allows users to upload images of solar panels and automatically predicts the panel’s condition. The app can also optionally detect defects and obstructions using bounding boxes.

---

## Features

- **Classifies Solar Panel Conditions**:  
  - Bird-drop  
  - Clean  
  - Dusty  
  - Electrical-damage  
  - Physical-Damage  
  - Snow-Covered  

- **Displays Prediction Confidence**  
- **Optional Bounding Boxes** for detected defects (requires object detection model)  
- **Supports real-time image upload** for quick inspection  

---

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/solar-panel-detection.git
cd solar-panel-detection
```

**2. Create and activate a Python virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3.Install dependencies:**
```bash
pip install -r requirements.txt
```

**Requirements**

Python >= 3.10
TensorFlow 2.15
Streamlit
Pillow
NumPy
Pandas (optional, for charts)


**Usage**

1.Place your trained classification model (solar_panel_condition_model.keras) in the project folder.

2. Run the Streamlit app:

streamlit run SolarPanel_App.py

3. Upload a solar panel image.

4. View the predicted condition, confidence, and optionally bounding boxes.


## 👤 Created By

Chaitanya Valeti (MAE4 [AIML-C-WD-E-B18)
Built as a Mini project for the **AIML (Artificial Intelligence & Machine Learning)** domain at GUVI (HCL Tech).
