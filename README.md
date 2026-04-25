This is a high-performance computer vision pipeline built to turn a standard smartphone into a localized AI tracking node. This project is a proof of concept designed to test and refine tracking and identification logic before it is applied to a Eufy security camera system and, eventually, an autonomous drone project further down the line.

The system streams live video from a mobile device over a local Wi-Fi network to a PC. A dual-model approach handles real-time detection: a hardware-accelerated YOLOv8 model for general object tracking and a FaceNet pipeline for specific human identification. I built this for local networks specifically to ensure near-zero latency and total data privacy by keeping all processing off the cloud.

Project Roadmap
Phase 1 (Current): Proof of concept using mobile hardware as a wireless IP node with integrated Facial Recognition.

Phase 2 (Planned): Integration with Eufy S330 hardware via local RTSP streaming.

Phase 3 (Planned): Deployment of optimized tracking logic to an autonomous drone system.

Key Features
Mobile-to-PC Streaming: Uses mobile hardware as a wireless IP camera to test tracking algorithms.

Localized Facial Recognition: Identify authorized individuals vs. "Unknown" entities using Euclidean distance matching.

Hardware-Accelerated AI: Utilizes PyTorch and NVIDIA CUDA to run inference on a local RTX 3080 GPU.

Zero-Cloud Architecture: 100 percent of the processing stays on the local machine.

Secure Environment Configuration: Decouples network credentials and hardware paths from the codebase using .env files.

Tech Stack
Language: Python 3.12

Vision and AI: OpenCV, Ultralytics YOLOv8, FaceNet (MTCNN & InceptionResnetV1)

Network: Local WLAN (HTTP/RTSP protocol)

Environment: python-dotenv, PyTorch (CUDA 12.1)

Step 1: Phone App Setup (The Camera Node)
To send the video feed from your phone to your PC, you need an app that broadcasts your camera over your local Wi-Fi.

Download DroidCam (available on iOS and Android).

Connect your phone to the same Wi-Fi network as your PC.

Open the app and find the Wi-Fi IP Address and Port (e.g., http://10.0.0.149:4747).

Keep the app open so the phone continues to broadcast the stream.

Step 2: PC Setup (The AI Brain)
Clone the repository:

Bash
git clone https://github.com/BennyDogfish/cuda-rtsp-tracker.git
cd cuda-rtsp-tracker
Install the 3080-Optimized environment:
To ensure the RTX 3080 handles the dual-AI load correctly, install the dependencies in this specific order:

Bash
# 1. Install GPU-Optimized Torch
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2. Install AI Models (using --no-deps to prevent version conflicts)
pip install ultralytics facenet-pytorch --no-deps

# 3. Install Supporting Libraries
pip install "numpy<2.0.0" "Pillow<10.3.0" python-dotenv opencv-python
Step 3: Configuration (Network & Faces)
This project uses .env files and local directories to separate private data from the logic.

Network: Create a file named .env in the main folder and add your phone's URL:

Plaintext
CAMERA_URL="http://10.0.0.149:4747/video"
Authorized Faces: Create a folder named known_faces. Drop a .jpg or .png of any person you want the AI to recognize. Name the file after the person (e.g., steve.jpg).

Step 4: Run the Tracker
This project is modular. You can run the general object tracker or the specific face identification node independently.

To run Object Tracking (YOLOv8):

Bash
python virtual_tracker.py
To run Facial Recognition (FaceNet):

Bash
python face_tracker.py
Controls:

Press the 'q' key while the window is active to shut down the pipeline and close the connection.