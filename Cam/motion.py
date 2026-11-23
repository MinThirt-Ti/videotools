import cv2
import numpy as np

def display_webcam_native_resolution():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam Native Resolution: {width} x {height}")
    
    # cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    
    previous_frame = None  # Store the previous frame
    alpha = 0.5 # Weight for previous frame. Adjust for more/less overlap
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Could not read frame.")
            break

        if previous_frame is not None:
            # Calculate the absolute difference between frames
            diff = cv2.absdiff(frame, previous_frame)

            cv2.imshow("Webcam Feed", diff)
        else:
            cv2.imshow("Webcam Feed", frame) # First Frame

        previous_frame = frame.copy() # Store the current frame as previous for the next iteration

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    display_webcam_native_resolution()
