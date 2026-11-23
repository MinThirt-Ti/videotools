def display_screen():
    try:
        import cv2
        import numpy as np
        from PIL import ImageGrab
        import torch
        
        model_type = "MiDaS_small"
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("WHERE TF IS CUDA?!")
            return False
        model.to(device)
        model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
    except Exception as e:
        print(e)
        return False
    
    scale = 0.25
    
    while True:
        screenshot = ImageGrab.grab()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        new_height, new_width, _ = frame.shape
        
        input_batch = transform(frame).to(device)
        with torch.no_grad():
            prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(new_height,new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        normalized_depth_map_color = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
        
        catted = np.concatenate((frame, normalized_depth_map_color), 1)
        
        cv2.imshow("Depth Feed", catted)
        # cv2.imshow("Screen Feed", frame)  # Display the frame

        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()  # Close all windows
    return True

def main():
    q = display_screen()
    
    if q:
        print("Safely Exited")
    else:
        print("Exited with Error")

if __name__ == "__main__":
    main()