# YOLO Light Effect and Video Generator

This Streamlit application allows you to create a video with dynamic headlight and indicator effects overlayed on an image. You can customize the blink interval and add audio to generate a final video.

## Features
- **Object Detection**: Uses a YOLO model to detect headlights and indicators in the uploaded image.
- **Dynamic Overlays**: Applies blinking effects for headlights and indicators based on user-defined intervals.
- **Video Generation**: Creates a video from the image with the overlays and combines it with uploaded audio.
- **Customizable Settings**: Choose video duration and blink interval for precise control.

---

## Requirements

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Streamlit
- OpenCV
- Pillow
- NumPy
- ffmpeg-python
- ultralytics (for YOLO model)

### YOLO Model
- Train a YOLO model to detect headlights and indicators and save the weights (`best.pt`).
- Update the path to your YOLO weights in the script: `C:\Users\harsh.gahlot\PycharmProjects\yolo10\indicator\runs\detect\train3\weights\best.pt`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gahlotharsh/headlight_indicator_blinking.git
   cd yolo-light-effect-generator

This version includes:
1. **`pip install -r requirements.txt`** instructions under installation.
2. Detailed steps to install and verify `ffmpeg` on different platforms.
3. Verification step for `ffmpeg` installation using the `ffmpeg -version` command.

```bash
   Streamlit run streamlit.py
