import cv2
import time
from ultralytics import RTDETR

# 1. Initialize AI Model into RTX 3080 VRAM
print("Waking up the RTX 3080 CUDA Cores...")
model = RTDETR("rtdetr-l.pt")

# 2. The Cleaned RTSP Link (Using your locked-in credentials)
# Note: Ensure you clicked "Confirm" in the Eufy App to lock these in!
rtsp_url = "rtsp://Rn7QViF2yvqVYLiK:Jx2wRNamuIpI4Luy@10.0.0.45/live0"

print("\n--- VISION NODE ONLINE ---")
print("Status: Armed and Patrolling. Waiting for Eufy motion trigger...\n")

while True:
    # Use FFMPEG and a minimal buffer to catch the stream instantly
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # This heartbeat proves the code is running and waiting for the camera
        print(f"[{time.strftime('%H:%M:%S')}] Camera asleep. 3080 standing by...", end="\r")
        cap.release()
        time.sleep(1) # Aggressive 1-second check
        continue

    print("\n\n[!!!] CONNECTION ESTABLISHED. ENGAGING AI TRACKING...")

    while True:
        success, frame = cap.read()
        if not success:
            print("\n[i] Stream closed by camera. Returning to Patrol Mode.")
            break 

        # Process the frame on your RTX 3080
        # imgsz=640 is the sweet spot for speed and accuracy
        results = model(frame, imgsz=640, device=0, classes=[0], conf=0.5, verbose=False)

        # Draw the Targeting HUD
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Neon Green Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # On-Screen System Status
        cv2.putText(frame, "RTX 3080 | AI ACTIVE", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Eufy AI Vision Node", frame)

        # Press 'q' to gracefully shut down
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("System shutting down...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Clean up the connection between motion events
    cap.release()
    cv2.destroyAllWindows()