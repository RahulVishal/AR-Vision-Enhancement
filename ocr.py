import cv2
import numpy as np
import easyocr
from concurrent.futures import ThreadPoolExecutor

reader = easyocr.Reader(['en'], gpu=True)

def ocr(image):
    h, w = image.shape[:2]
    segments = [
        image[0:h//2, 0:w//2],     # top-left
        image[0:h//2, w//2:],      # top-right
        image[h//2:, 0:w//2],      # bottom-left
        image[h//2:, w//2:]        # bottom-right
    ]
    offsets = [(0, 0), (0, w//2), (h//2, 0), (h//2, w//2)]

    def detect_text(seg_idx):
        seg = segments[seg_idx]
        return (seg_idx, reader.readtext(seg))

    with ThreadPoolExecutor() as executor:
        results = executor.map(detect_text, range(4))

    annotated_image = image.copy()
    for seg_idx, detections in results:
        y_off, x_off = offsets[seg_idx]
        for bbox, text, _ in detections:
            pts = np.array(bbox, dtype=np.int32) + [x_off, y_off]
            rect = cv2.boundingRect(pts)
            x, y, w_, h_ = rect
            cv2.rectangle(annotated_image, (x, y), (x + w_, y + h_), (0, 0, 0), -1)
            cv2.putText(annotated_image, text, (x, y + h_ - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated_image

if __name__ == "__main__":
    image = cv2.imread("input_images/oak-worth.png")
    out = ocr(image)
    cv2.imwrite("ocr.png", out)
