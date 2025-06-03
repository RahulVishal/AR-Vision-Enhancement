import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def _morphgrad_overlay(img, thresh=30, min_len=100, dk=3, di=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    uint8_grad = cv2.convertScaleAbs(grad)
    _, binary = cv2.threshold(uint8_grad, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.arcLength(cnt, closed=False) >= min_len:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    mask = cv2.dilate(mask, np.ones((dk, dk), np.uint8), iterations=di)
    overlay = img.copy()
    overlay[mask > 0] = (255, 255, 255)
    return overlay

def detect_edges(input_data):
    """
    Applies morphological-gradient-based edge detection on a single image or list.
    """
    if isinstance(input_data, (list, tuple)):
        results = [None] * len(input_data)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_morphgrad_overlay, img): idx for idx, img in enumerate(input_data)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results
    else:
        return _morphgrad_overlay(input_data)