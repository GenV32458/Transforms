import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# IMAGE: upload a JPG or PNG file
print("Upload an image")
uploaded = files.upload()
image_name = next(iter(uploaded))
image = cv2.imread(image_name)
if image is None:
    raise ValueError("Could not read the uploaded image.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fft = np.fft.fftshift(np.fft.fft2(gray))
spectrum = 20 * np.log(np.abs(fft) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(image_rgb); plt.title("Original Image"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale Image"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(spectrum, cmap="gray"); plt.title("Image Fourier Spectrum"); plt.axis("off")
plt.tight_layout(); plt.show()

# VIDEO: upload an MP4, AVI, or another OpenCV-supported video
print("Upload a video")
uploaded_video = files.upload()
video_name = next(iter(uploaded_video))
cap = cv2.VideoCapture(video_name)
if not cap.isOpened():
    raise ValueError("Could not open the uploaded video.")

success, frame = cap.read()
cap.release()
if not success:
    raise ValueError("Could not read a frame from the video.")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_fft = np.fft.fftshift(np.fft.fft2(gray_frame))
frame_spectrum = 20 * np.log(np.abs(frame_fft) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(frame_rgb); plt.title("First Video Frame"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gray_frame, cmap="gray"); plt.title("Grayscale Video Frame"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(frame_spectrum, cmap="gray"); plt.title("Video Frame Fourier Spectrum"); plt.axis("off")
plt.tight_layout(); plt.show()
