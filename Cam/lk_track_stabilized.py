import cv2
import numpy as np

def run_stabilized_flow():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Parameters ---
    # Processing resolution (lower = faster FPS & less noise)
    process_w, process_h = 640, 480
    
    # Feature params for stabilization (finding background anchors)
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    # LK Flow params for tracking anchors
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Dense Flow Params (Farneback)
    farneback_params = dict(pyr_scale=0.5, levels=3, winsize=15, 
                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Visualization params
    step = 10         # Grid spacing for drawing lines
    fade_factor = 0.90 # 0.90 = fast fade, 0.98 = long trails
    min_magnitude = 3.0 # Threshold to ignore residual stabilization noise

    # --- Init ---
    ret, prev_frame_raw = cap.read()
    if not ret: return
    
    prev_frame = cv2.resize(prev_frame_raw, (process_w, process_h))
    prev_frame = cv2.flip(prev_frame, 1)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Grid for drawing
    h, w = prev_frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    
    mask = np.zeros_like(prev_frame)

    print("Stabilized Flow Tracker Started.")
    print("This method aligns the background to ignore camera translation.")
    print("Press 'q' to quit. Press 'c' to clear.")

    while True:
        ret, frame_raw = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame_raw, (process_w, process_h))
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 1. STABILIZATION STEP ---
        # Find points in previous frame to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        warped_prev_gray = prev_gray # Default fallback if stabilization fails

        if p0 is not None and len(p0) > 10:
            # Track points to current frame
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            
            # Select good points
            good_prev = p0[st==1]
            good_curr = p1[st==1]
            
            if len(good_curr) > 10:
                # Find the transformation matrix (Homography) that maps Prev -> Curr
                # RANSAC is robust against outliers (moving objects)
                H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Warp the PREVIOUS frame to align with the CURRENT frame
                    # This simulates what the previous frame would look like if the camera 
                    # had already moved to the current position.
                    warped_prev_gray = cv2.warpPerspective(prev_gray, H, (w, h))

        # --- 2. DENSE FLOW CALCULATION ---
        # Compute flow between Warped_Previous and Current.
        # Since the background is aligned, flow there should be ~0.
        flow = cv2.calcOpticalFlowFarneback(warped_prev_gray, gray, None, **farneback_params)

        # --- 3. VISUALIZATION ---
        mask = (mask.astype(float) * fade_factor).astype(np.uint8)

        fx, fy = flow[y, x].T
        
        # Calculate Magnitude
        mag_sq = fx**2 + fy**2
        
        # Filter noise
        vis = mag_sq > (min_magnitude ** 2)
        
        pts_x = x[vis]
        pts_y = y[vis]
        vec_x = fx[vis]
        vec_y = fy[vis]
        
        for i in range(len(pts_x)):
            px, py = pts_x[i], pts_y[i]
            dx, dy = vec_x[i], vec_y[i]
            
            # Green lines
            cv2.line(mask, (px, py), (int(px+dx), int(py+dy)), (0, 255, 0), 1)

        # Composite
        combined = cv2.add(frame, mask)
        cv2.imshow('Stabilized Flow', combined)

        prev_gray = gray.copy()
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            mask = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stabilized_flow()