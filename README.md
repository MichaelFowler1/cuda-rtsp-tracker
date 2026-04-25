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
