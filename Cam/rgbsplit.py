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
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Could not read frame.")
            break

        # Split the frame into RGB channels
        blue, green, red = cv2.split(frame)

        # Create blank images for single channels
        blank = np.zeros_like(blue) # Use blue as reference for size and type

        # Create RGB images for each channel (only one color will be present)
        red_image = cv2.merge([blank, blank, red])
        green_image = cv2.merge([blank, green, blank])
        blue_image = cv2.merge([blue, blank, blank])

        # Concatenate the images horizontally
        combined_frame = np.concatenate([red_image, green_image, blue_image], axis=1)

        # Resize the combined frame to fit within the original window size if necessary
        # combined_frame = cv2.resize(combined_frame, (width*3, height)) # or other scaling factors

        cv2.imshow("Webcam Feed", combined_frame)

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    display_webcam_native_resolution()
