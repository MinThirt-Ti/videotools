import cv2
import mediapipe as mp
import math

# --- 1. SETUP & CONFIGURATION ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Face Mesh (Eyes)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Hand Tracking
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)

# --- 2. HELPER FUNCTIONS ---

def calculate_ear(landmarks, indices, w, h):
    """Calculates Eye Aspect Ratio (EAR) to detect blinking."""
    coords = []
    for i in indices:
        pt = landmarks[i]
        coords.append((pt.x * w, pt.y * h))
        
    v1 = math.dist(coords[1], coords[5])
    v2 = math.dist(coords[2], coords[4])
    hor = math.dist(coords[0], coords[3])
    return (v1 + v2) / (2.0 * hor)

def count_fingers(hand_landmarks):
    """
    Counts how many fingers are 'up'.
    Logic: Compares the Y-coordinate of the Finger Tip vs the PIP (Knuckle).
    Note: In image coords, Y=0 is top. So Tip < PIP means 'Up'.
    """
    fingers = []
    
    # Thumb (Logic is different: compare X usually, but for simplicity here we check offset)
    # We'll use a simplified check for thumb: is Tip to the left/right of IP joint depending on hand?
    # For robustnes in this snippet, we will focus on the 4 main fingers first.
    
    # Finger Tip Indices: [Index, Middle, Ring, Pinky] -> [8, 12, 16, 20]
    # Finger PIP Indices: [6, 10, 14, 18]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    # Check 4 Fingers
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1) # Finger is Open
        else:
            fingers.append(0) # Finger is Closed
            
    # Simple Thumb Check (Tip y < MCP y) - basic vertical check
    if hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y:
        fingers.append(1)
    else:
        fingers.append(0)
        
    return fingers

# Indices
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

print("MEGA TRACKER: Face + Eyes + Hands + Gestures. Press 'ESC' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    # Flip & Convert
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # --- 3. INFERENCE ---
    face_results = face_mesh.process(rgb_image)
    hand_results = hands.process(rgb_image)

    # --- 4. PROCESSING FACE ---
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw Mesh
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                None, mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # BLINK LOGIC
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_CONTOUR, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_CONTOUR, w, h)
            
            # Draw Status
            if left_ear < 0.22 and right_ear < 0.22:
                cv2.putText(image, "** BLINK **", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw Irises (Just center points for clean look)
            def to_pixel(idx):
                pt = face_landmarks.landmark[idx]
                return (int(pt.x * w), int(pt.y * h))
            
            cv2.circle(image, to_pixel(468), 3, (0, 255, 0), -1)
            cv2.circle(image, to_pixel(473), 3, (255, 0, 255), -1)

    # --- 5. PROCESSING HANDS & GESTURES ---
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            
            # Draw Skeleton
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # GESTURE LOGIC
            finger_state = count_fingers(hand_landmarks)
            total_fingers = sum(finger_state)
            
            # Determine Gesture Name
            gesture = "Unknown"
            if total_fingers == 5:
                gesture = "OPEN PALM"
            elif total_fingers == 0:
                gesture = "FIST"
            elif total_fingers == 2 and finger_state[0] == 1 and finger_state[1] == 1: 
                # Index (0 in our list is Index) + Thumb (4 in list is Thumb)? 
                # Note: My list order in count_fingers was [Index, Middle, Ring, Pinky, Thumb]
                # So Index is index 0.
                gesture = "PEACE" # Rough approximation
            elif total_fingers == 1 and finger_state[0] == 1:
                gesture = "POINTING"
                
            # Draw Gesture Text near the hand (Wrist point)
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(image, gesture, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Mega Tracker', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()