import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import time

def run_vision_system():
    # 1. Hardware Initialization
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- HARDWARE ACCELERATION: {device.upper()} ---")

    # 2. Model Loading (Depth Only)
    print("Loading DepthAnything...")
    depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    # 3. Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 4. View Modes Configuration
    MODES = [
        ("DEPTH: INFERNO", cv2.COLORMAP_INFERNO, "depth"),
        ("DEPTH: MAGMA", cv2.COLORMAP_MAGMA, "depth"),
        ("DEPTH: VIRIDIS", cv2.COLORMAP_VIRIDIS, "depth"),
        ("DEPTH: JET", cv2.COLORMAP_JET, "depth"),
        ("RAW: RGB", None, "raw")
    ]
    mode_index = 0

    print("--- DEPTH SYSTEM ENGAGED ---")
    print("KEYS: [Q] Terminate | [T] Cycle Color Map")

    while True:
        success, frame = cap.read()
        if not success:
            break

        start_time = time.perf_counter()
        
        current_name, current_cmap, current_type = MODES[mode_index]
        final_image = None

        if current_type == "depth":
            # --- Depth Path ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            depth_output = depth_pipe(pil_image)
            depth_np = np.array(depth_output["depth"])
            depth_np = cv2.resize(depth_np, (frame.shape[1], frame.shape[0]))
            
            depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            final_image = cv2.applyColorMap(depth_norm, current_cmap)

        else:
            # --- Raw Path ---
            final_image = frame

        # 5. HUD
        fps = 1 / (time.perf_counter() - start_time)
        
        cv2.putText(final_image, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_image, f"MODE: {current_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Depth Estimation Output", final_image)

        # 6. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            mode_index = (mode_index + 1) % len(MODES)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_system()