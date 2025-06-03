import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from detect_edges import detect_edges

def rp_heal(image, k=2):
    """
    Apply RP healing by first restoring brightness (adding k), then boosting contrast, color, and edges.
    """
    # 0. Restore brightness (moderate compensation)
    restored = cv2.add(image, k)
    # 1. CLAHE on restored image (moderate clip)
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge((L2, A, B))
    contrast_enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # 2. Moderate Unsharp mask
    blurred = cv2.GaussianBlur(contrast_enhanced, (5,5), 0)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.8, blurred, -0.8, 0)
    # 3. Edge overlay (with moderate thresholds)
    overlay = detect_edges(sharpened)
    # 4. Color boost (toned down)
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * 1.3, 0, 255)
    hsv[...,2] = np.clip(hsv[...,2] * 1.2, 0, 255)
    final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 5. Final contrast stretch (slightly increase contrast)
    final = cv2.convertScaleAbs(final, alpha=1.2, beta=0)
    return final


def rp_simulate(image):
    """
    Simulate RP vision by darkening linearly: subtract k from each channel.
    """
    # 1. Darken by subtracting k
    dark = cv2.subtract(image, 10)
    
    # 2. reduce sharpness and add noise
    # blur to reduce sharpness
    blurred = cv2.GaussianBlur(dark, (0, 0), sigmaX=3, sigmaY=3)
    # add gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    noisy = cv2.add(blurred.astype(np.int16), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # 3. add in slight vignette
    rows, cols = image.shape[:2]
    # create gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y @ kernel_x.T
    mask = kernel / np.max(kernel)
    vignette = np.empty_like(noisy, dtype=np.float32)
    for i in range(3):
        vignette[:, :, i] = noisy[:, :, i].astype(np.float32) * mask
    vignette = np.clip(vignette, 0, 255).astype(np.uint8)
    
    # 4. reduce contrast of things
    alpha = 0.7  # contrast factor <1
    beta = 0    # no brightness shift
    contrast = cv2.convertScaleAbs(vignette, alpha=alpha, beta=beta)
    
    return contrast


if __name__ == "__main__":
    frame = cv2.imread("input_images/dark.png")
    rp_view = rp_simulate(frame)
    healed = rp_heal(frame)

    titles = ["Original Image", "Simulated RP View", "Healed Image (RP Compensation)"]
    images = [frame, rp_view, healed]

    h, w = frame.shape[:2]
    pad = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)

    grid_height = h * 3 + pad * 4
    grid = np.zeros((grid_height, w, 3), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        y_offset = pad * (i + 1) + h * i
        grid[y_offset:y_offset + h] = img

        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = y_offset - 10
        cv2.putText(grid, title, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite("rp_output_vertical_grid.png", grid)