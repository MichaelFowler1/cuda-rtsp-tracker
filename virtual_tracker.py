import cv2
import threading
import time
import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
camera_url_env = os.getenv("CAMERA_URL", "0")
URL = int(camera_url_env) if camera_url_env.isdigit() else camera_url_env

MODEL_NAME = "yolov8l.pt"

class HighThroughputVision:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        
        # Force 1080p resolution for better clarity and detection accuracy
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Minimize buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        self.frame = None
        self.stopped = False
        
        print("Initializing YOLO pipeline with CUDA support...")
        self.model = YOLO(MODEL_NAME)
        
        # Warm up the model to initialize CUDA cores
        self.model.track(source="https://ultralytics.com/images/bus.jpg", device="cuda", half=True, verbose=False)

    def start(self):
        threading.Thread(target=self.grab_frames, daemon=True).start()
        return self

    def grab_frames(self):
        while not self.stopped:
            # Continuous grab for lowest possible latency
            ret = self.cap.grab()
            if ret:
                success, frame = self.cap.retrieve()
                if success:
                    self.frame = frame

    def run_inference(self):
        print("High-throughput node active. Press 'q' to quit.")
        
        while not self.stopped:
            if self.frame is not None:
                # Using stream=True to optimize memory and keep data on the GPU
                results = self.model.track(
                    source=self.frame,
                    device="cuda",
                    persist=True,
                    imgsz=1088,
                    conf=0.7,
                    iou=0.5,
                    max_det=40,    # Limit detections per frame to prevent rendering lag
                    half=True,
                    stream=True,   
                    verbose=False
                )
                
                for r in results:
                    # Render bounding boxes and display
                    cv2.imshow("YOLOv8 Inference", r.plot())
                    break 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vision = HighThroughputVision(URL).start()
    time.sleep(1.5) # Allow camera buffer to fill
    try:
        vision.run_inference()
    finally:
        vision.stop()

if __name__ == "__main__":
    main()