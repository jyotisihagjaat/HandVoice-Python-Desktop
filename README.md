# HandVoice — Python Desktop Gesture Communicator

A webcam desktop app: show a one-hand pose and the meaning appears directly in the camera window. No browser or Live Server required.

## Recommended setup (Windows)
1. Extract this ZIP.
2. Install **Python 3.12 (64-bit)** from python.org. During installation, enable the Python launcher (`py`).
   - MediaPipe's pinned wheel is intended for Python 3.11/3.12. Your existing Python 3.13 may not have a compatible wheel for this version.
3. Double-click `RUN.bat`.
4. On first run it creates a local `.venv`, installs packages, then opens the webcam app.
5. Hold one hand clearly in front of the camera. Press **Q** or **Esc** to close.

## Included
- Python 3 desktop app (OpenCV window)
- MediaPipe real-time hand landmarks
- Smoothed gesture classification to reduce flicker
- Large on-screen phrase and gesture label
- FPS display and hand-detected status
- 20 gesture/phrase mappings

## Troubleshooting
- Close apps that may be using the webcam (Teams, Meet, Zoom, Camera).
- Windows: Settings → Privacy & security → Camera → allow desktop apps to access your camera.
- If the wrong camera opens, edit `cv2.VideoCapture(0, ...)` in `main.py` to use camera index `1`.
- First installation requires internet access to download Python packages. After installation, the app runs locally.

## Accuracy note
This is a gesture-to-phrase prototype, not an official ASL or Indian Sign Language translator. It uses approximate static landmark rules; some signs overlap and recognition depends on hand orientation, camera angle, lighting and individual anatomy. Open palm, fist, peace, thumbs up/down, pointing and call-me are the primary intended gestures. The remaining phrase mappings are experimental and may overlap. For real accessibility use, a validated sign-language model and user testing are necessary.

## Privacy
Frames are processed locally on your computer. This project does not upload or save webcam video.
