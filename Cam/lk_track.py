import cv2
import numpy as np

def run_dense_relative_tracking():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Configuration ---
    # Sampling step: Draw a line every X pixels (Lower = denser "field")
    step = 10  
    
    # Threshold: Magnitude of relative motion required to draw a line
    # Increase if you see too much noise/static
    min_magnitude = 2.0  
    
    # Trail Length: 0.90 = short, 0.98 = long, 1.0 = infinite
    fade_factor = 0.94

    # --- Initialization ---
    ret, prev_frame = cap.read()
    if not ret: return
    
    prev_frame = cv2.flip(prev_frame, 1)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Persistent mask for trails
    mask = np.zeros_like(prev_frame)
    
    # Pre-define grid coordinates for faster looping
    h, w = prev_frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)

    print("Dense Relative Motion Tracker Started.")
    print("Press 'q' to quit. Press 'c' to clear screen.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, 
                                            flags=0)

        # 2. Calculate Global Motion (Camera Shake)
        # We take the median of the flow to estimate how the "background" moved.
        # We downsample [::4, ::4] for speed (calculating median of full HD is slow)
        cam_flow_x = np.median(flow[::4, ::4, 0])
        cam_flow_y = np.median(flow[::4, ::4, 1])

        # 3. Fade the Trail Mask
        # Multiply by fade_factor to slowly dim old lines
        mask = (mask.astype(float) * fade_factor).astype(np.uint8)

        # 4. Extract Flow Vectors at Grid Points
        fx, fy = flow[y, x].T

        # 5. Subtract Camera Motion (Relative Motion)
        rel_fx = fx - cam_flow_x
        rel_fy = fy - cam_flow_y

        # Calculate magnitude of RELATIVE vectors
        # (This avoids drawing lines for the background moving)
        mag_sq = rel_fx**2 + rel_fy**2
        
        # Filter: Only draw lines where relative motion > threshold
        # We use a boolean mask 'vis' to select only the moving points
        vis = mag_sq > (min_magnitude ** 2)
        
        # 6. Draw the Lines on the Mask
        # We iterate only over points that passed the filter (vis)
        # This is much faster than looping over every single point
        pts_x = x[vis]
        pts_y = y[vis]
        vec_x = rel_fx[vis]
        vec_y = rel_fy[vis]
        
        for i in range(len(pts_x)):
            px, py = pts_x[i], pts_y[i]
            dx, dy = vec_x[i], vec_y[i]
            
            # Draw line (Green)
            cv2.line(mask, (px, py), (int(px+dx), int(py+dy)), (0, 255, 0), 1)

        # 7. Combine Frame and Mask
        img = cv2.add(frame, mask)
        
        cv2.imshow('Dense Relative Motion', img)
        
        prev_gray = gray
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            mask = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dense_relative_tracking()