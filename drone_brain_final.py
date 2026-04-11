import cv2
import time
from djitellopy import Tello
from ultralytics import RTDETR # <-- Changed import

# --- CONFIGURATION & SMOOTHING ---
WIDTH, HEIGHT = 960, 720
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# PID Gains
YAW_GAIN = 0.15 
FB_GAIN = 0.25      # Increased slightly because we are smoothing it now
UD_GAIN = 0.15

# --- SLEW RATE LOGIC ---
smooth_fb = 0.0     # This stores the "current" speed
MAX_RAMP = 2.0      # How much speed can change per frame (lower = smoother)
MAX_SPEED = 40      # Safety cap

# 1. INITIALIZE MODEL (Ultralytics)
model = RTDETR("rtdetr-l.pt") # Or use YOLO("yolov8n.pt") if you prefer YOLO

# 2. INITIALIZE TELLO
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

def get_smooth_movement(box, current_fb):
    x1, y1, x2, y2 = box
    obj_center_x = (x1 + x2) / 2
    obj_center_y = (y1 + y2) / 2
    obj_width = x2 - x1
    
    # 1. Calculate the "Target" speed the AI wants right now
    target_fb = (200 - obj_width) * FB_GAIN
    
    # 2. RAMP LOGIC: Don't jump to target_fb, move toward it by MAX_RAMP
    if current_fb < target_fb:
        current_fb = min(current_fb + MAX_RAMP, target_fb)
    elif current_fb > target_fb:
        current_fb = max(current_fb - MAX_RAMP, target_fb)
    
    # 3. Standard YAW and UP/DOWN
    yaw = int((obj_center_x - CENTER_X) * YAW_GAIN)
    up_down = int(-(obj_center_y - CENTER_Y) * UD_GAIN)
    
    # 4. Final Safety Clamps
    fb_out = int(max(-MAX_SPEED, min(MAX_SPEED, current_fb)))
    yaw_out = int(max(-60, min(60, yaw)))
    
    return [0, fb_out, up_down, yaw_out], current_fb

try:
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Tracking mode handles ID persistence automatically
        results = model.track(frame, device="cuda", persist=True, classes=[0], verbose=False)
        
        movement = [0, 0, 0, 0]
        
        # Check if we have boxes AND the tracker has assigned an ID
        if results[0].boxes and results[0].boxes.id is not None:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            
            # USE THE SMOOTHING FUNCTION
            movement, smooth_fb = get_smooth_movement(box, smooth_fb)
            
            # Visual Feedback
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        else:
            # If person lost, bleed off the speed gradually
            smooth_fb *= 0.9 
            if abs(smooth_fb) < 1: smooth_fb = 0
            movement = [0, int(smooth_fb), 0, 0]

        tello.send_rc_control(movement[0], movement[1], movement[2], movement[3])
        cv2.imshow("Tello Tracker + Smooth-Slew", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
finally:
    tello.send_rc_control(0, 0, 0, 0) # Stop motors before landing
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()