import cv2
from ultralytics import YOLO
import time

def run_yolo():
    # Load model (downloads automatically if not found)
    # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    model = YOLO("yolov8n.pt") 

    # Initialize Webcam (0 is usually default)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Resolution settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("--- YOLOv8 ENGAGED ---")
    print("Press 'q' to terminate.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        start_time = time.perf_counter()

        # Inference
        # conf=0.5 sets the confidence threshold
        results = model(frame, conf=0.5, verbose=False)

        # Plot results directly onto the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        fps = 1 / (time.perf_counter() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo()