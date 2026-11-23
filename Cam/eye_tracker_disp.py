import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURATION ---
SMOOTHING_FACTOR = 0.1  # Lower = Smoother but slower (0.01 to 1.0)
RING_SIZE_MULTIPLIER = 10 # Scale the jitter ring

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Create a black window to represent the "Screen"
screen_w, screen_h = 960, 540 # Half 1080p for visibility
cv2.namedWindow('Gaze Projection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gaze Projection', screen_w, screen_h)

# --- INDICES ---
# Left Eye Indices (Eye corners + Iris Center)
# Left Corner: 33, Right Corner: 133, Iris Center: 468
EYE_LEFT_CORNER = 33
EYE_RIGHT_CORNER = 133
IRIS_CENTER = 468

# --- CALIBRATION VARIABLES ---
# We track the Min and Max "Eye Ratios" we see to map them to the screen
x_min, x_max = 1.0, 0.0 # Start inverted so they expand
y_min, y_max = 1.0, 0.0

# --- SMOOTHING BUFFERS ---
recent_gaze_x = []
recent_gaze_y = []
BUFFER_SIZE = 10 

def get_relative_eye_position(landmarks, width, height):
    """
    Returns a ratio (0.0 to 1.0) of where the iris is inside the eye.
    0.0 = Left/Top, 1.0 = Right/Bottom
    """
    # Get coordinates
    left = landmarks[EYE_LEFT_CORNER]
    right = landmarks[EYE_RIGHT_CORNER]
    iris = landmarks[IRIS_CENTER]
    
    # 1. Calculate Horizontal Ratio (Simple projection on X axis for speed)
    # Note: For real robustness, we should use vector projection, but X-dist works for 2D heads.
    eye_width = abs(right.x - left.x)
    if eye_width == 0: return 0.5, 0.5 # Prevent divide by zero
    
    # How far is iris from the left corner?
    iris_dist_x = iris.x - left.x
    ratio_x = iris_dist_x / eye_width
    
    # 2. Calculate Vertical Ratio
    # We use the eye corners as the "Horizon" and measure deviation
    eye_center_y = (left.y + right.y) / 2.0
    # Normalize by width because eye height changes with blinking
    ratio_y = (iris.y - eye_center_y) / eye_width 
    
    return ratio_x, ratio_y

print("GAZE TRACKER RUNNING")
print("STEP 1: Move your head to center the camera.")
print("STEP 2: Look at the Top-Left of the 'Gaze Projection' window.")
print("STEP 3: Look at the Bottom-Right of the 'Gaze Projection' window.")
print("The system auto-expands calibration as you look around. Press 'r' to reset calibration.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape
    
    # Create the Blank Screen
    projection_screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # 1. Get Raw Eye Ratios
            raw_x, raw_y = get_relative_eye_position(landmarks, img_w, img_h)
            
            # 2. Auto-Calibration (Expand the known range as you look around)
            # This "learns" the extent of your eye movement
            if raw_x < x_min: x_min = raw_x
            if raw_x > x_max: x_max = raw_x
            if raw_y < y_min: y_min = raw_y
            if raw_y > y_max: y_max = raw_y
            
            # 3. Normalize (Map Raw Ratio to 0.0 - 1.0 range based on calibration)
            # Avoid divide by zero
            denom_x = (x_max - x_min) if (x_max - x_min) > 0 else 0.001
            denom_y = (y_max - y_min) if (y_max - y_min) > 0 else 0.001
            
            norm_x = (raw_x - x_min) / denom_x
            norm_y = (raw_y - y_min) / denom_y
            
            # 4. Map to Screen Coordinates
            screen_x = int(norm_x * screen_w)
            screen_y = int(norm_y * screen_h)
            
            # 5. Smoothing & Jitter Calculation
            recent_gaze_x.append(screen_x)
            recent_gaze_y.append(screen_y)
            
            if len(recent_gaze_x) > BUFFER_SIZE:
                recent_gaze_x.pop(0)
                recent_gaze_y.pop(0)
            
            # Average (The smoothed point)
            avg_x = int(np.mean(recent_gaze_x))
            avg_y = int(np.mean(recent_gaze_y))
            
            # Standard Deviation (The "Probability Ring" Size)
            std_x = np.std(recent_gaze_x)
            std_y = np.std(recent_gaze_y)
            radius = int((std_x + std_y) * RING_SIZE_MULTIPLIER)
            
            # --- DRAWING ---
            
            # A. Draw on Camera Feed (Feedback)
            h, w = img_h, img_w
            cx, cy = int(landmarks[IRIS_CENTER].x * w), int(landmarks[IRIS_CENTER].y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)
            
            # Display Raw Ratios
            cv2.putText(image, f"Ratio X: {raw_x:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Ratio Y: {raw_y:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # B. Draw on Projection Screen
            # The "Probability Ring" (Cyan, Semi-transparent fill would be cool, but circle is fast)
            # We constrain it so it doesn't crash the script if radius is huge or negative
            safe_radius = max(5, min(radius, 200)) 
            cv2.circle(projection_screen, (avg_x, avg_y), safe_radius, (255, 255, 0), 2)
            
            # The "Gaze Point" (Red Dot)
            cv2.circle(projection_screen, (avg_x, avg_y), 8, (0, 0, 255), -1)
            
            # Calibration Grid (Visual aid)
            cv2.rectangle(projection_screen, (50, 50), (screen_w-50, screen_h-50), (50, 50, 50), 1)
            cv2.putText(projection_screen, "Look at corners to Calibrate", (int(screen_w/2)-150, int(screen_h/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('MediaPipe Eye Tracking', image)
    cv2.imshow('Gaze Projection', projection_screen)

    key = cv2.waitKey(5) & 0xFF
    if key == 27: # ESC
        break
    if key == ord('r'): # Reset Calibration
        x_min, x_max = 1.0, 0.0
        y_min, y_max = 1.0, 0.0
        recent_gaze_x = []
        recent_gaze_y = []
        print("Calibration Reset!")

cap.release()
cv2.destroyAllWindows()