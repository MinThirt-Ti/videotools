import cv2
import numpy as np

def display_webcam_native_resolution():
    arbn = 
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Attempt to set to 4K or max cam size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam Native Resolution: {width} x {height}")
    
    cv2.namedWindow("Webcam Feed", width, height)
    cv2.moveWindow("Webcam Feed", 0,0)
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def minth():
    display_webcam_native_resolution()
    
    return None

if __name__ == "__main__":
    minth()
