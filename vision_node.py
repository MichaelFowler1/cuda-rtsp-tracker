import cv2
import threading
import time
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("CAMERA_URL", "0")
URL = int(URL) if URL.isdigit() else URL
MODEL_NAME = "yolov8m.pt" 

class UltimateVisionNode:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Prevents hanging if the camera momentarily disconnects
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000) 
        
        self.frame = None
        self.new_frame_ready = False # THE FIX: Stops the GPU from double-processing
        self.lock = threading.Lock() # THE FIX: Stops screen tearing
        self.stopped = False
        
        print("🏎️ Initializing CUDA kernels...")
        self.model = YOLO(MODEL_NAME)
        self.model.predict("https://ultralytics.com/images/bus.jpg", device="cuda", half=True, verbose=False)

    def start(self):
        threading.Thread(target=self.grab_frames, daemon=True).start()
        return self

    def grab_frames(self):
        while not self.stopped:
            try:
                ret = self.cap.grab()
                if ret:
                    success, latest_frame = self.cap.retrieve()
                    if success:
                        # Lock prevents the 3080 from reading while we are writing
                        with self.lock: 
                            self.frame = latest_frame
                            self.new_frame_ready = True
                else:
                    # If Wi-Fi drops a packet, don't spin the CPU to 100%
                    time.sleep(0.005) 
            except Exception as e:
                print(f"⚠️ Network hiccup: {e}")
                time.sleep(0.01)

    def run_inference(self):
        print("✅ Bulletproof Node Active. Press 'q' to quit.")
        
        while not self.stopped:
            current_frame = None
            
            # Safely extract the frame
            with self.lock:
                if self.new_frame_ready and self.frame is not None:
                    current_frame = self.frame.copy() 
                    self.new_frame_ready = False # Tell the GPU to wait for the next frame!

            # ONLY run inference if we actually have a brand new image
            if current_frame is not None:
                results = self.model.track(
                    source=current_frame,
                    device="cuda",
                    persist=True,
                    imgsz=640,
                    conf=0.4,
                    iou=0.45,
                    classes=[0, 2, 3, 5, 7], 
                    half=True,
                    verbose=False
                )
                
                cv2.imshow("3080 Real-Time", results[0].plot())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vision = UltimateVisionNode(URL).start()
    time.sleep(1.0)
    try:
        vision.run_inference()
    finally:
        vision.stop()

if __name__ == "__main__":
    main()