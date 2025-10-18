@echo off
echo =========================================
echo  Setting up Solar Panel Detection Env
echo =========================================

:: Optional: create virtual environment
python -m venv venv
call venv\Scripts\activate

echo.
echo --- Upgrading pip ---
python -m pip install --upgrade pip setuptools wheel

echo.
echo --- Installing TensorFlow (CPU version) ---
pip install tensorflow==2.15.0 tensorflow-intel==2.15.0 keras==2.15.0

echo.
echo --- Installing YOLO (Ultralytics) ---
pip install ultralytics==8.2.50

echo.
echo --- Installing Streamlit and supporting packages ---
pip install streamlit pandas pillow matplotlib opencv-python

echo.
echo --- Fixing protobuf and gRPC dependencies ---
pip install --upgrade grpcio grpcio-status protobuf==4.25.8

echo.
echo --- Optional: For GPU acceleration (if CUDA installed) ---
:: pip install tensorflow[and-cuda]

echo.
echo ✅ Environment setup complete!
echo You can now run: streamlit run SolarPanel_App.py
pause
