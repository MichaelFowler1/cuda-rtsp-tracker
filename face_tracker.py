import cv2
import torch
import threading
import time
import os
from dotenv import load_dotenv
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load environment variables
load_dotenv()

camera_url_env = os.getenv("CAMERA_URL", "0")
CAMERA_URL = int(camera_url_env) if camera_url_env.isdigit() else camera_url_env

class FaceTracker:
    def __init__(self, src, known_faces_dir="known_faces"):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.frame = None
        self.stopped = False
        
        # Setup device for inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing face tracker on {self.device.type.upper()}...")

        # MTCNN for face detection and cropping
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # ResNet for generating facial embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.names = []
        self.embeddings = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        print(f"Loading reference faces from '{directory}'...")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created '{directory}' directory. Add images here to detect specific faces.")
            return

        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0]
                
                img = Image.open(path).convert('RGB')
                face_tensor = self.mtcnn(img)
                
                if face_tensor is not None:
                    # Grab the first face found in the image
                    face_unsqueeze = face_tensor[0].unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    with torch.no_grad():
                        emb = self.resnet(face_unsqueeze)
                    
                    self.embeddings.append(emb)
                    self.names.append(name)
                    print(f"Loaded reference for: {name}")

        if self.embeddings:
            # Group embeddings into a single tensor for faster distance calculations
            self.embeddings = torch.cat(self.embeddings)
        else:
            self.embeddings = None

    def start(self):
        threading.Thread(target=self.grab_frames, daemon=True).start()
        return self

    def grab_frames(self):
        while not self.stopped:
            ret = self.cap.grab()
            if ret:
                success, frame = self.cap.retrieve()
                if success:
                    self.frame = frame

    def run_inference(self):
        print("Tracker active. Press 'q' to exit.")
        
        while not self.stopped:
            if self.frame is not None:
                # Convert to PIL Image for MTCNN
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Detect faces
                boxes, _ = self.mtcnn.detect(pil_img)
                faces_tensor = self.mtcnn(pil_img)

                if boxes is not None and faces_tensor is not None:
                    faces_tensor = faces_tensor.to(self.device)
                    
                    # Generate embeddings for current frame
                    with torch.no_grad():
                        current_embeddings = self.resnet(faces_tensor)

                    # Compare against known faces
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(b) for b in box]
                        name = "Unknown"
                        color = (0, 0, 255) # Red for unknown

                        if self.embeddings is not None:
                            # Check euclidean distance between current face and known faces
                            dists = torch.cdist(current_embeddings[i].unsqueeze(0), self.embeddings)
                            min_dist_idx = torch.argmin(dists).item()
                            min_dist = dists[0][min_dist_idx].item()

                            # Using 0.8 as the distance threshold
                            if min_dist < 0.8:
                                name = self.names[min_dist_idx]
                                color = (0, 255, 0) # Green for known

                        # Draw bounding box and label
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(self.frame, (x1, y1 - 35), (x2, y1), color, cv2.FILLED)
                        cv2.putText(self.frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Face Tracker", self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vision = FaceTracker(CAMERA_URL).start()
    time.sleep(1.5) # Give the camera buffer a moment to fill
    try:
        vision.run_inference()
    finally:
        vision.stop()

if __name__ == "__main__":
    main()