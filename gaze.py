import cv2
import numpy as np
from collections import deque

def gay_ze(frame):
    """
    Maintain a queue of up to 30 edge-maps. Each call:
      1. Convert `frame` to a smaller grayscale + edge map.
      2. Insert it into a deque (maxlen=30), updating a running sum.
      3. Once the deque has 30 entries, compute the per-pixel average edge-map,
         compare the newest edge-map to that average, and return True if they
        ’re sufficiently similar (low mean abs. difference), else False.
    """
    # On first call, initialize static attributes
    if not hasattr(gay_ze, "buffer"):
        # Desired size for speed (you can adjust as needed)
        gay_ze.target_width = 160
        gay_ze.target_height = 120

        # Deque to hold up to 30 edge‐maps (uint8)
        gay_ze.buffer = deque(maxlen=30)

        # Running sum of the edge‐maps (float32), same shape as resized edge‐map
        gay_ze.running_sum = np.zeros((gay_ze.target_height, gay_ze.target_width), dtype=np.float32)

        # Flag to know when buffer is “full”
        gay_ze.full = False

    # Step 1: Convert input frame into a resized, grayscale edge-map
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (gay_ze.target_width, gay_ze.target_height), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # uint8 array, values 0 or up to 255

    # Step 2: Insert new edge‐map into deque, update running_sum accordingly
    if len(gay_ze.buffer) == 30:
        # Buffer is already full: oldest will be popped automatically
        oldest = gay_ze.buffer[0]
        gay_ze.running_sum -= oldest.astype(np.float32)
    else:
        # Still filling up; once we hit 30, mark as full
        if len(gay_ze.buffer) == 29:
            gay_ze.full = True

    gay_ze.buffer.append(edges)
    gay_ze.running_sum += edges.astype(np.float32)

    # If not yet full, we cannot compare properly—return False
    if not gay_ze.full:
        return False

    # Step 3: Compute the average edge‐map for the 30 frames
    avg_edge = gay_ze.running_sum / 30.0  # float32

    # Compare new edge map to this average
    # Compute mean absolute deviation:
    diff = np.abs(edges.astype(np.float32) - avg_edge)
    mean_diff = float(np.mean(diff))

    # Threshold chosen empirically—tune as needed.
    # If mean difference is low, all 30 frames are “the same.”
    THRESHOLD = 20.0

    return mean_diff < THRESHOLD


if __name__ == "__main__":
    # Example usage: read from a video file, call gay_ze on each frame.
    video_path = "input_images/input_video.mp4"  # Change this to your video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'.")
        exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        same_scene = gay_ze(frame)
        print(f"Frame {frame_idx:04d}: same_scene={same_scene}")
        frame_idx += 1

        # If you want to visualize edge‐maps or display, you could add:
        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) == 27:  # Esc to exit early
        #     break

    cap.release()
