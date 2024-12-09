import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import ffmpeg
import tempfile
import os

# Set up streamlit
st.title("YOLO Light Effect and Video Generator")
st.write("Upload your image, headlight overlay, indicator overlay, and an audio file to create a video.")

# File uploaders
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
headlight_file = st.file_uploader("Upload Headlight Overlay", type=["png"])
indicator_file = st.file_uploader("Upload Indicator Overlay", type=["png"])
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

# Video duration input (in seconds)
video_duration = st.slider("Select Video Duration (in seconds)", min_value=0, max_value=30, value=10)

# Blink interval input (in seconds)
blink_interval = st.slider("Select Blink Interval (in seconds)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Parameters
fps = 30  # frames per second

# Check if files are uploaded and the submit button is clicked
if image_file and headlight_file and indicator_file and audio_file:

    submit_button = st.button("Generate Video")

    if submit_button:
        # Load YOLO model
        model = YOLO(r"C:\Users\harsh.gahlot\PycharmProjects\yolo10\indicator\runs\detect\train3\weights\best.pt")

        # Load images
        image = Image.open(image_file).convert("RGBA")
        headlight_image = Image.open(headlight_file).convert("RGBA")
        indicator_image = Image.open(indicator_file).convert("RGBA")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)

        # Perform detection
        results = model.predict(source=image_cv, conf=0.1)
        boxes = results[0].boxes.xyxy.numpy().tolist() if results and len(results[0].boxes) > 0 else []
        classes = results[0].boxes.cls.numpy().tolist() if results and len(results[0].boxes) > 0 else []

        # Map class IDs to lights
        HEADLIGHT_CLASS = 0  # Replace with your YOLO model's class ID for headlights
        INDICATOR_CLASS = 1  # Replace with your YOLO model's class ID for indicators

        # Helper function to apply light effect
        def apply_light_effect(img, light_img, box, scale=1.0):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            resized_light = light_img.resize((int(w * scale), int(h * scale)))
            x_center = x1 + (w - resized_light.width) // 2
            y_center = y1 + (h - resized_light.height) // 2
            img.paste(resized_light, (x_center, y_center), resized_light)
            return img

        # Create frames for the video
        frames = []
        total_frames = fps * video_duration  # total number of frames based on the selected video duration
        for frame_num in range(total_frames):
            frame = image.copy()
            is_headlight_on = (frame_num // (fps * blink_interval)) % 2 == 0

            for box, cls in zip(boxes, classes):
                if cls == HEADLIGHT_CLASS and is_headlight_on:
                    frame = apply_light_effect(frame, headlight_image, box)
                elif cls == INDICATOR_CLASS and not is_headlight_on:
                    frame = apply_light_effect(frame, indicator_image, box)

            frames.append(cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR))

        # Save frames to temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_video:
            temp_video_path = temp_video.name
            video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps,
                                           (image.width, image.height))
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.read())

        # Prepare output directory and final output path
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = os.path.join(output_dir, "final_output_video_with_audio.avi")

        # Combine video and audio
        video_input = ffmpeg.input(temp_video_path)
        audio_input = ffmpeg.input(temp_audio_path)

        try:
            ffmpeg.output(video_input, audio_input, final_output_path, vcodec="copy", acodec="aac").run(
                overwrite_output=True)
            st.success("Video with audio has been generated!")

            # Provide download link for the final output video
            with open(final_output_path, "rb") as file:
                st.download_button("Download Final Video", file, file_name="final_output_video_with_audio.avi")

        except ffmpeg.Error as e:
            st.error(f"An error occurred during the video processing: {e}")

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(temp_audio_path)

else:
    st.warning("Please upload all required files (image, overlays, audio).")
