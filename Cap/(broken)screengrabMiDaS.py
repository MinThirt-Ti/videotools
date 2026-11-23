import cv2
import numpy as np
from PIL import ImageGrab
import torch
import mss
import mss.tools
import win32gui

def list_window_names():
    """Lists all open window titles."""
    windows = []
    
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(winEnumHandler, None)
    
    for i, (hwnd, title) in enumerate(windows):
        print(f"{i+1}. {title}")
    
    while True:
        try:
            choice = int(input("Enter the number of the window to select: "))
            if 1 <= choice <= len(windows):
                hwnd, title = windows[choice - 1]
                return hwnd
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def MiDaS_screen():
    model_type = "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
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
    
    hwnd = list_window_names()
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
    
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    frame = np.fromstring(bmpstr, dtype='uint8')
    frame.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    
    left, top, right, bottom = 0, 0, 2560, 1440
    print(left, top, right, bottom)
    width = right - left
    height = bottom - top
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    while True:
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            screenshot = sct.grab(monitor)
        
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_BGRA2BGR)
        
        new_width = int(width / 4)
        new_height = int(height / 4)
        screenshot_cvr = cv2.resize(screenshot_cv, (0, 0), fx=0.25, fy=0.25)
        
        cv2.imshow("Screen Feed cv", screenshot_cv)  # Display the frame
        cv2.imshow("Screen Feed cvr", screenshot_cvr)  # Display the frame
        
        screenshot_rgb = cv2.cvtColor(screenshot_cvr, cv2.COLOR_BGR2RGB)
        
        input_batch = transform(screenshot_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(new_height,new_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        normalized_depth_map_color = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_BONE)
        cv2.imshow("Depth Feed", normalized_depth_map_color)
        
        key = cv2.waitKey(1)  # Wait for 1ms. Press 'q' to quit.
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()  # Close all windows

if __name__ == "__main__":
    # selected_window = select_window()
    
    
    MiDaS_screen()
