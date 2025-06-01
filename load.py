import os
import cv2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

lock = Lock()

processed_paths = []


def edge_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # smooth while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    # compute gradients
    grad_x = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad = cv2.convertScaleAbs(
        cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5,
                        cv2.convertScaleAbs(grad_y), 0.5, 0)
    )
    # simple threshold to get clean edges
    _, edges = cv2.threshold(abs_grad, 50, 255, cv2.THRESH_BINARY)
    # close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    # overlay
    edge_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0, edge_bgr, 1.0, 0)


def process_frame(frame, disease_name, apply_edge_detection=True):
    # 1. Normalize colors and dynamic range
    def normalize_chunk(chunk):
        hsv = cv2.cvtColor(chunk, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2]
        mean_v = v.mean()
        # adjust contrast based on brightness
        if mean_v > 180:
            contrast = 0.7
        elif mean_v < 60:
            contrast = 1.5
        else:
            contrast = 1.0
        # equalize histogram on V and S to spread values
        hsv[..., 2] = cv2.equalizeHist(hsv[..., 2].astype(np.uint8)).astype(np.float32)
        hsv[..., 1] = cv2.equalizeHist(hsv[..., 1].astype(np.uint8)).astype(np.float32)
        # apply contrast to V channel
        hsv[..., 2] = np.clip((hsv[..., 2] - 128) * contrast + 128, 0, 255)
        chunk_eq = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # slight global brightness shift to ensure no extremes
        return cv2.convertScaleAbs(chunk_eq, alpha=1.0, beta=5)

    splits = np.array_split(frame, 4, axis=0)
    with ThreadPoolExecutor() as executor:
        norm_chunks = list(executor.map(normalize_chunk, splits))
    frame = np.vstack(norm_chunks)

    # 2. Edge detection if requested
    if apply_edge_detection:
        frame = edge_detect(frame)

    # 3. Disease-specific brightness/contrast tweak
    if disease_name != 'rp':
        def adjust_pixel(chunk):
            return cv2.convertScaleAbs(chunk, alpha=0.9, beta=-25)
        splits2 = np.array_split(frame, 4, axis=0)
        with ThreadPoolExecutor() as executor:
            adjusted = list(executor.map(adjust_pixel, splits2))
        return np.vstack(adjusted)
    return frame

if __name__ == "__main__":
    videos_dir = 'videos copy'
    output_dir = 'outputs copy'
    disease_name = 'not_disease'
    apply_edge_detection = True

    os.makedirs(output_dir, exist_ok=True)
    print("🚀 Starting video processing with balanced color normalization...\n")

    video_files = [f for f in os.listdir(videos_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for filename in video_files:
        print(f"🎬 Processing: {filename}")
        input_path = os.path.join(videos_dir, filename)
        output_path = os.path.join(output_dir, filename)
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame, disease_name, apply_edge_detection)
            out.write(processed)
            count += 1
            if count % 10 == 0 or count == total_frames:
                bar_len = 30
                filled_len = bar_len * count // total_frames
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                percent = count / total_frames * 100
                with lock:
                    print(f"\r🎞️ {filename} ⏳ [{bar}] "
                          f"{count}/{total_frames} ({percent:.1f}%)", end='')

        cap.release()
        out.release()
        elapsed = time.time() - start_time
        sec_per_frame = elapsed / count if count else 0
        with lock:
            print(f"\n✅ {filename} done | ⏱️ {elapsed:.2f}s | "
                  f"⌛ {sec_per_frame:.4f} sec/frame\n")
            processed_paths.append(output_path)

    print("🏁 All videos processed.")