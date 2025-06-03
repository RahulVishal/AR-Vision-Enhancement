import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sam import sam

def cb_simulate(image):
    """
    Simulate red-green color blindness by merging R and G channels.
    """
    # convert to float for manipulation
    img = image.astype(np.float32)
    # compute average of red and green
    avg_rg = (img[..., 2] + img[..., 1]) / 2
    sim = img.copy()
    sim[..., 2] = avg_rg  # red channel
    sim[..., 1] = avg_rg  # green channel
    return np.clip(sim, 0, 255).astype(np.uint8)

def cb_heal(image):
    """
    Placeholder for CB healing: call sam(frame) when available.
    """
    image = sam(image, 'red')
    return cb_simulate(image)

if __name__ == "__main__":
    frame = cv2.imread("input_images/apple.png")
    cb_view = cb_simulate(frame)
    healed = cb_heal(frame)

    titles = ["Original Image", "Simulated CB View", "Healed Image (CB Compensation)"]
    images = [frame, cb_view, healed]

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

    cv2.imwrite("cb_output_vertical_grid.png", grid)