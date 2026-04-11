import cv2
import threading
import time
import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- CONFIGURATION ---
# Safely fetch the URL without hardcoding it. Falls back to default webcam (0) if not found.
URL = os.getenv("CAMERA_URL", "0")
# Convert to integer if it's falling back to the default '0' camera
URL = int(URL) if URL.isdigit() else URL

MODEL_NAME = "yolov8m.pt" 

class HighThroughputVision:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        # Increase internal buffer for high-speed transfer
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.frame = None
        self.stopped = False
        
        print("🏎️ Overclocking Pipeline for RTX 3080...")
        self.model = YOLO(MODEL_NAME)
        # Force a heavy warmup to initialize all CUDA cores
        self.model.track(source="https://ultralytics.com/images/bus.jpg", device="cuda", half=True, verbose=False)

    def start(self):
        threading.Thread(target=self.grab_frames, daemon=True).start()
        return self

    def grab_frames(self):
        while not self.stopped:
            # Absolute highest speed capture
            ret = self.cap.grab()
            if ret:
                success, frame = self.cap.retrieve()
                if success:
                    self.frame = frame

    def run_inference(self):
        # We use the 'stream=True' generator to keep data on the GPU
        print("✅ High-Throughput Node Active. Press 'q' in the window to quit.")
        
        while not self.stopped:
            if self.frame is not None:
                # stream=True is the key to getting that GPU usage up
                # The 'stream=True' generator keeps data on the GPU
                results = self.model.track(
                    source=self.frame,
                    device="cuda",
                    persist=True,
                    imgsz=640,
                    conf=0.3,      # Keeps the AI highly sensitive
                    iou=0.5,
                    max_det=40,    # Capped at 40 to prevent CPU drawing lag
                    classes=[0, 2, 3, 5, 7], # THE FIX: Only look for specific targets
                    half=True,
                    stream=True,   
                    verbose=False
                )
                
                for r in results:
                    # plot() is fast, but imshow is slow. 
                    # We only display if the window is ready.
                    cv2.imshow("3080 High-Throughput", r.plot())
                    break 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vision = HighThroughputVision(URL).start()
    time.sleep(1.5)
    try:
        vision.run_inference()
    finally:
        vision.stop()

if __name__ == "__main__":
    main()