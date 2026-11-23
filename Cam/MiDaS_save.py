import cv2
import numpy as np

print("Importing torch...")
try:
    import torch
    print("Successfully imported torch.")
except Exception as e:
    print(e)
    exit()

def process_video(video_path, output_path):
    model_type = "MiDaS_small"
    print(f"Loading {model_type}...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        print(f"Successfully loaded {model_type}.")
    except Exception as e:
        print(e)
        return
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WHERE TF IS CUDA?!")
        return
    
    model.to(device)
    model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam Native Resolution: {width} x {height}")
    print(f"Video FPS: {fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))
    
    
    # cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    
    grid_size = (9, 9)
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            print("Error: Could not read frame.")
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        threshold_value = 127  # Adjust this value to control the threshold
        _, thresholded_depth = cv2.threshold(normalized_depth_map, threshold_value, 255, cv2.THRESH_BINARY)
        
        grid_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background

        cell_height = height // grid_size[0]
        cell_width = width // grid_size[1]
        
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height

                cell_area = cell_width * cell_height
                roi = thresholded_depth[y1:y2, x1:x2]  # Region of interest (cell)
                white_pixel_count = np.count_nonzero(roi)  # Count white pixels in cell
                percentage_filled = (white_pixel_count / cell_area) * 100

                if percentage_filled > 50:
                    grid_frame[y1:y2, x1:x2] = (0, 0, 255)  # Red cell
                else:
                    grid_frame[y1:y2, x1:x2] = (0, 0, 0)  # Black cell

                # Draw grid lines (optional)
                cv2.rectangle(grid_frame, (x1, y1), (x2, y2), (100, 100, 100), 1) # Gray grid lines
        
        contours, _ = cv2.findContours(thresholded_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_with_contours = frame.copy()  # Important: copy the frame to draw on it
        cv2.drawContours(frame_with_contours, contours, -1, (0, 0, 255), 2)  # Red contours
        
        depth_map_color = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
        
        catted = np.concatenate((frame_with_contours, depth_map_color, grid_frame), 1)
        
        cv2.imshow("Feed", catted)
        out.write(catted)

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    out.release()  # Release the video writer
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    video_path = "120eb2n4.mp4"  # Replace with your video file path
    output_path = "processed_vid.mp4"  # Replace with desired output path
    process_video(video_path, output_path)
