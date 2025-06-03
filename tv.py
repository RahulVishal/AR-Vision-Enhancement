import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def tv_heal(image):
    """
    Apply TV healing by compressing the full FOV into view with smooth, blended color extender background.
    """
    h, w = image.shape[:2]
    blurred_bg = cv2.GaussianBlur(image, (0, 0), sigmaX=50, sigmaY=50)

    scale_factor = 0.7
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    mask = np.zeros((h, w), dtype=np.float32)
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    mask[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = 1.0
    mask = cv2.GaussianBlur(mask, (151, 151), 0)
    mask = mask[..., np.newaxis]

    small_canvas = np.zeros_like(image, dtype=np.float32)
    small_canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = small.astype(np.float32)

    healed = small_canvas * mask + blurred_bg.astype(np.float32) * (1 - mask)
    return np.clip(healed, 0, 255).astype(np.uint8)

def tv_simulate(image):
    """
    Simulate TV vision with a reduced black tunnel zone plus a mild vignette ring.
    """
    h, w = image.shape[:2]
    dark = cv2.convertScaleAbs(image, alpha=0.5, beta=0)

    center_x, center_y = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    radius_clear = min(w, h) * 0.3
    radius_vignette = min(w, h) * 0.45  # increased to shrink black zone

    output = np.zeros_like(image, dtype=np.float32)

    center_mask = dist <= radius_clear
    output[center_mask] = image[center_mask].astype(np.float32)

    ring_mask = (dist > radius_clear) & (dist <= radius_vignette)
    blend_vals = np.clip((radius_vignette - dist[ring_mask]) / (radius_vignette - radius_clear), 0, 1)[..., np.newaxis]
    orig_ring = image[ring_mask].astype(np.float32)
    dark_ring = dark[ring_mask].astype(np.float32)
    output[ring_mask] = orig_ring * blend_vals + dark_ring * (1 - blend_vals)

    # outer region remains black but is now smaller
    return np.clip(output, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    frame = cv2.imread("input_images/dark.png")
    tv_view = tv_simulate(frame)
    healed = tv_heal(frame)

    titles = ["Original Image", "Simulated TV View", "Healed Image (TV Compensation)"]
    images = [frame, tv_view, healed]

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
        grid[y_offset : y_offset + h] = img

        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = y_offset - 10
        cv2.putText(grid, title, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite("tv_output_vertical_grid.png", grid)
