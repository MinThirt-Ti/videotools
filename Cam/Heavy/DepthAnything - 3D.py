import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import time
import open3d as o3d

# --- CONFIGURATION ---
SKIP = 3            # Density x2 (Previously 6)
FOV_H = 60.0        # Webcam FOV
NEAR_CLIP = 0.5     # Nearest physical distance (meters)
FAR_CLIP = 5.0      # Farthest physical distance (meters)

def create_reference_grid(size=5, step=1, y_level=-1.0):
    """Generates a floor grid. 1 Square = 1 Meter."""
    lines = []
    points = []
    # Z-parallel (Depth)
    for x in range(-size, size+1, step):
        points.append([x, y_level, 0])
        points.append([x, y_level, size])
        lines.append([len(points)-2, len(points)-1])
    # X-parallel (Width)
    for z in range(0, size+1, step):
        points.append([-size, y_level, z])
        points.append([size, y_level, z])
        lines.append([len(points)-2, len(points)-1])
    
    colors = [[0.3, 0.3, 0.3] for _ in range(len(lines))]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def run_vision_system():
    # 1. Hardware Initialization
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- HARDWARE ACCELERATION: {device.upper()} ---")

    # 2. Model Loading
    print("Loading DepthAnything...")
    depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    # 3. Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Pre-calculate projection grid
    W, H = 640, 480 
    # Focal Length (Pixels)
    fx = W / (2 * np.tan(np.deg2rad(FOV_H) / 2)) 
    fy = fx
    cx, cy = W / 2, H / 2
    
    # Sparse Grid Coordinates
    xv, yv = np.meshgrid(np.arange(0, W, SKIP), np.arange(0, H, SKIP))
    # Normalized Rays (z=1)
    x_ray = (xv - cx) / fx
    y_ray = (yv - cy) / fy

    # 4. Visualization State
    show_pcd = False
    vis = o3d.visualization.Visualizer()
    pcd_geometry = o3d.geometry.PointCloud()
    
    # Grid: 5x5 meters, 1m steps, placed 1m below camera
    # Floor is now at Y = -1.0 (Standard coordinates)
    ref_grid = create_reference_grid(size=5, step=1, y_level=-1.0) 
    ref_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    is_pcd_init = False

    MODES = [
        ("DEPTH: INFERNO", cv2.COLORMAP_INFERNO, "depth"),
        ("DEPTH: MAGMA", cv2.COLORMAP_MAGMA, "depth"),
        ("DEPTH: VIRIDIS", cv2.COLORMAP_VIRIDIS, "depth"),
        ("RAW: RGB", None, "raw")
    ]
    mode_index = 0

    print("--- DEPTH SYSTEM ENGAGED ---")
    print("KEYS (Focus OpenCV Window): [Q] Quit | [T] Cycle 2D Map | [P] Toggle 3D Cloud")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            start_time = time.perf_counter()
            
            # --- Inference ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Get Raw Tensor
            depth_output = depth_pipe(pil_image)
            raw_depth = np.array(depth_output["depth"]) # Shape: (H, W)
            raw_depth = cv2.resize(raw_depth, (W, H))

            # --- 3D Processing ---
            if show_pcd:
                # 1. Downsample
                d_sparse = raw_depth[::SKIP, ::SKIP]
                rgb_sparse = rgb_frame[::SKIP, ::SKIP] / 255.0 

                # 2. Pseudo-Calibration (Normalize & Clamp)
                # Normalize raw disparity to 0..1 per frame
                d_min, d_max = d_sparse.min(), d_sparse.max()
                d_norm = (d_sparse - d_min) / (d_max - d_min + 1e-5) # Avoid div/0
                
                # Metric Mapping: Disparity -> Depth (Meters)
                # Z = 1 / (m * d + c)
                # Logic: D=1 -> Z=NEAR_CLIP, D=0 -> Z=FAR_CLIP
                c_val = 1.0 / FAR_CLIP
                m_val = (1.0 / NEAR_CLIP) - c_val
                
                # Final Metric Depth (Meters)
                z_metric = 1.0 / (m_val * d_norm + c_val)
                
                # 3. Project (Meters)
                x_metric = x_ray * z_metric
                y_metric = y_ray * z_metric
                
                # Stack (N, 3). 
                # FIX: Invert Y to map "Pixel Down" to "World Down"
                # Camera is at (0,0,0). Y is Up.
                points = np.dstack((x_metric, -y_metric, z_metric)).reshape(-1, 3)
                colors = rgb_sparse.reshape(-1, 3)

                # 4. Open3D Update
                pcd_geometry.points = o3d.utility.Vector3dVector(points)
                pcd_geometry.colors = o3d.utility.Vector3dVector(colors)

                if not is_pcd_init:
                    vis.create_window(window_name="Metric 3D Stream", width=640, height=480)
                    vis.add_geometry(pcd_geometry)
                    vis.add_geometry(ref_grid)
                    vis.add_geometry(ref_axes)
                    
                    opt = vis.get_render_option()
                    opt.background_color = np.asarray([0, 0, 0])
                    opt.point_size = 2.0 # Slightly smaller points for higher density
                    
                    # Set View: Standard 3D (Y-Up)
                    vis.reset_view_point(True)
                    ctr = vis.get_view_control()
                    
                    # Look towards +Z (Where points are projected)
                    ctr.set_front([0, 0, 1]) 
                    
                    # FIX: Set Up vector to +Y (Standard World Up)
                    ctr.set_up([0, 1, 0])    
                    
                    # Position camera slightly back/up to see the scene
                    ctr.set_lookat([0, 0, 2]) 
                    ctr.set_zoom(0.6)
                    
                    is_pcd_init = True
                
                vis.update_geometry(pcd_geometry)
                if not vis.poll_events():
                    show_pcd = False
                    vis.destroy_window()
                    is_pcd_init = False
                vis.update_renderer()

            elif is_pcd_init:
                vis.destroy_window()
                is_pcd_init = False

            # --- HUD ---
            current_name, current_cmap, current_type = MODES[mode_index]
            final_image = None

            if current_type == "depth":
                depth_norm_vis = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                final_image = cv2.applyColorMap(depth_norm_vis, current_cmap)
            else:
                final_image = frame

            fps = 1 / (time.perf_counter() - start_time)
            
            cv2.putText(final_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(final_image, f"RANGE: {NEAR_CLIP}m - {FAR_CLIP}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if show_pcd:
                cv2.putText(final_image, "3D ACTIVE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Depth Estimation Output", final_image)

            # --- Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('t'): mode_index = (mode_index + 1) % len(MODES)
            elif key == ord('p'): show_pcd = not show_pcd

    except Exception as e:
        print(f"Critical Failure: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try: vis.destroy_window()
        except: pass

if __name__ == "__main__":
    run_vision_system()