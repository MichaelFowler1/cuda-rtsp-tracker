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

URL = os.getenv("CAMERA_URL", "0")
URL = int(URL) if URL.isdigit() else URL

class DedicatedFaceTracker:
    def __init__(self, src, known_faces_dir="known_faces"):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.frame = None
        self.stopped = False
        
        # Hardware Check: Ensure CUDA is active for the RTX 3080
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🏎️ Booting Dedicated Face Pipeline on {self.device.type.upper()}...")

        # MTCNN: The AI that draws bounding boxes around faces
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # ResNet: The AI that converts faces into 512-dimensional embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Database for specific faces
        self.names = []
        self.embeddings = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        print(f"📂 Loading authorized faces from '{directory}'...")
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"⚠️ Created '{directory}' folder. Add images (e.g., 'Bob.jpg') to recognize specific faces!")
            return

        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0]
                
                # Load image and let MTCNN crop the face
                img = Image.open(path).convert('RGB')
                face_tensor = self.mtcnn(img)
                
                if face_tensor is not None:
                    # Isolate the first face found in the reference image
                    face_unsqueeze = face_tensor[0].unsqueeze(0).to(self.device)
                    
                    # Generate the mathematical embedding for this person
                    with torch.no_grad():
                        emb = self.resnet(face_unsqueeze)
                    
                    self.embeddings.append(emb)
                    self.names.append(name)
                    print(f"✅ Learned face: {name}")

        if self.embeddings:
            # Stack all known faces into a single tensor for hyper-fast GPU matrix math
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
        print("✅ Dedicated Face Node Active. Press 'q' to quit.")
        
        while not self.stopped:
            if self.frame is not None:
                # MTCNN expects PIL images in RGB format
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Step 1: Detect face coordinates and crop them
                boxes, _ = self.mtcnn.detect(pil_img)
                faces_tensor = self.mtcnn(pil_img)

                if boxes is not None and faces_tensor is not None:
                    faces_tensor = faces_tensor.to(self.device)
                    
                    # Step 2: Generate embeddings for all faces on screen
                    with torch.no_grad():
                        current_embeddings = self.resnet(faces_tensor)

                    # Step 3: Match against our list of specific people
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(b) for b in box]
                        name = "Unknown"
                        color = (0, 0, 255) # Red for unknown

                        if self.embeddings is not None:
                            # Calculate the mathematical distance between live face and database
                            dists = torch.cdist(current_embeddings[i].unsqueeze(0), self.embeddings)
                            min_dist_idx = torch.argmin(dists).item()
                            min_dist = dists[0][min_dist_idx].item()

                            # 0.8 threshold for matching
                            if min_dist < 0.8:
                                name = self.names[min_dist_idx]
                                color = (0, 255, 0) # Green for known

                        # Draw the UI
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(self.frame, (x1, y1 - 35), (x2, y1), color, cv2.FILLED)
                        cv2.putText(self.frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("3080 Face Recognition", self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vision = DedicatedFaceTracker(URL).start()
    time.sleep(1.5) # Allow camera buffer to fill
    try:
        vision.run_inference()
    finally:
        vision.stop()

if __name__ == "__main__":
    main()