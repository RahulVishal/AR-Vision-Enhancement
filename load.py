import os
import cv2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def edge_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # find and filter contours by length
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > 100]

    # blank mask and draw thick black lines
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, long_contours, -1, 255, thickness=3)

    # overlay mask onto frame
    edge_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0, edge_bgr, 1.0, 0)


def enhance_frame_pixelwise(frame):
    def enhance_chunk(chunk):
        hsv = cv2.cvtColor(chunk, cv2.COLOR_BGR2HSV).astype(np.float32)
        # your hue/sat/bright tweaks
        hsv[..., 0] *= 0.8
        hsv[..., 1] *= 2.0
        hsv[..., 2] *= 0.8

        mean_b = hsv[..., 2].mean()
        if mean_b > 100:
            contrast = 0.8  # reduce contrast for bright scenes
        elif mean_b < 60:
            contrast = 1.8  # boost contrast for dark scenes
        else:
            contrast = 1.2  # slight boost in mid-range
        hsv[..., 2] = (hsv[..., 2] - 128) * contrast + 128

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        chunk = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.convertScaleAbs(chunk, alpha=1.1, beta=10)
    splits = np.array_split(frame, 4, axis=0)
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(enhance_chunk, splits))
    return np.vstack(processed_chunks)


def process_frame(frame, disease_name, apply_edge_detection=True):
    if disease_name == 'not_disease':
        frame = enhance_frame_pixelwise(frame)
        if apply_edge_detection:
            frame = edge_detect(frame)
        return frame
    elif disease_name != 'rp':
        def adjust_pixel(chunk):
            return cv2.convertScaleAbs(chunk, alpha=0.9, beta=-25)
        splits = np.array_split(frame, 4, axis=0)
        with ThreadPoolExecutor() as executor:
            processed_chunks = list(executor.map(adjust_pixel, splits))
        return np.vstack(processed_chunks)
    else:
        return frame

lock = Lock()
processed_paths = []

def process_video(input_path, output_path, disease_name, apply_edge_detection=True):
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
        processed = process_frame(frame, disease_name, True)
        out.write(processed)
        count += 1
        if count % 10 == 0 or count == total_frames:
            bar_len = 30
            filled_len = bar_len * count // total_frames
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            percent = count / total_frames * 100
            with lock:
                print(f"\r🎞️ {os.path.basename(input_path)} ⏳ [{bar}] "
                      f"{count}/{total_frames} ({percent:.1f}%)", end='')

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    sec_per_frame = elapsed / count if count else 0
    with lock:
        print(f"\n✅ {os.path.basename(input_path)} done | ⏱️ {elapsed:.2f}s | "
              f"⌛ {sec_per_frame:.4f} sec/frame\n")
        processed_paths.append(output_path)

if __name__ == "__main__":
    videos_dir = 'videos'
    output_dir = 'outputs'
    disease_name = 'not_disease'
    apply_edge_detection = False

    os.makedirs(output_dir, exist_ok=True)
    print("🚀 Starting video processing with per-frame pixel parallelism...\n")

    video_files = [f for f in os.listdir(videos_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for filename in video_files:
        print(f"🎬 Processing: {filename}")
        process_video(
            os.path.join(videos_dir, filename),
            os.path.join(output_dir, f"{filename}"),
            disease_name,
            apply_edge_detection
        )
    print("🏁 All videos processed.")
