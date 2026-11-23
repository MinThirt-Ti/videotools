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
    
    ret, prev_frame = cap.read()  # Read the first frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Could not read frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # Adjust parameters
        
        threshold = 2.0  # Adjust this threshold value
        
        flow_im = np.zeros_like(frame)
        
        for y in range(0, height, 10):  # Sample flow vectors for display
            for x in range(0, width, 10):
                fx, fy = flow[y, x]
                
                magnitude = np.sqrt(fx**2 + fy**2) # Calculate vector magnitude
                
                if magnitude > threshold:
                    cv2.line(flow_im, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1)
        
        alpha = 0.5
        combined = cv2.addWeighted(flow_im, alpha, frame, 1 - alpha, 0)
        
        cv2.imshow("Webcam Feed", combined)  # Display the frame
        
        prev_gray = gray.copy() # Update previous frame

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    display_webcam_native_resolution()
