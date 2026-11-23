import cv2
import numpy as np

def display_MiDaS():
    print("Importing torch...")
    try:
        import torch
        print("Successfully imported torch.")
    except Exception as e:
        print(e)
        return
    
    model_type = "MiDaS_small"
    print(f"Loading {model_type}...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        print(f"Successfully loaded {model_type}.")
    except Exception as e:
        print(torch.hub.get_dir())
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
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.namedWindow("Webcam Feed", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Webcam Feed", 0,0)
    
    print(f"Webcam Native Resolution: {width} x {height}")
    
    grid_size = (9, 9)
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Could not read frame.")
            break
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
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
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        depth_map = prediction.cpu().numpy()
        
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        normalized_depth_map_color = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
        
        # dm_tmax = np.max(depth_map)
        # if dm_max == None:
            # dm_max = dm_tmax
        
        # if dm_tmax > dm_max:
            # dm_max = dm_tmax
            # print(f"Max = {dm_max}")
        
        # dm_tmin = np.min(depth_map)
        # if dm_min == None:
            # dm_min = dm_tmin
        
        # if dm_tmin < dm_min:
            # dm_min = dm_tmin
            # print(f"Min = {dm_min}")
        
        # dm = (depth_map / dm_max) * 255
        # dm_flat = dm.astype(np.uint8)
        # depth_map_color = cv2.applyColorMap(dm_flat, cv2.COLORMAP_BONE)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        threshold_value = 127  # Adjust this value to control the threshold
        _, thresholded_depth = cv2.threshold(normalized_depth_map, threshold_value, 255, cv2.THRESH_BINARY)
        
        mask = normalized_depth_map > threshold_value
        mask = mask.astype(np.uint8)
        
        # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        masking = np.zeros_like(frame)
        masking[mask == 1] = [255, 255, 255]
        
        masked_frame = frame.copy()
        masked_frame[mask == 0] = [0,0,0]
        
        # contours, _ = cv2.findContours(thresholded_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # frame_with_contours = frame.copy()  # Important: copy the frame to draw on it
        # cv2.drawContours(frame_with_contours, contours, -1, (0, 0, 255), 2)  # Red contours
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        # grid_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background

        # cell_height = height // grid_size[0]
        # cell_width = width // grid_size[1]
        
        # for row in range(grid_size[0]):
            # for col in range(grid_size[1]):
                # x1 = col * cell_width
                # y1 = row * cell_height
                # x2 = (col + 1) * cell_width
                # y2 = (row + 1) * cell_height

                # cell_area = cell_width * cell_height
                # roi = thresholded_depth[y1:y2, x1:x2]  # Region of interest (cell)
                # white_pixel_count = np.count_nonzero(roi)  # Count white pixels in cell
                # percentage_filled = (white_pixel_count / cell_area) * 100

                # if percentage_filled > 50:
                    # grid_frame[y1:y2, x1:x2] = (0, 0, 255)  # Red cell
                # else:
                    # grid_frame[y1:y2, x1:x2] = (0, 0, 0)  # Black cell

                # # Draw grid lines (optional)
                # cv2.rectangle(grid_frame, (x1, y1), (x2, y2), (100, 100, 100), 1) # Gray grid lines
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        overlay = cv2.addWeighted(masking, 0.5, frame, 0.7, 0)
        catted = np.concatenate((masked_frame, normalized_depth_map_color, overlay), 1)
        cv2.imshow("Webcam Feed", catted)
        
        # cv2.imshow("Webcam Feed", depth_map_color)
        # cv2.imshow("Webcam Feed2", normalized_depth_map_color)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        
        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    display_MiDaS()
