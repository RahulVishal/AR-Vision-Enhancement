import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def sam(image, color_name):
    """
    Runs SAM on the BGR image, finds segments matching `color_name` under a balanced check:
    - For "red": at least 10% of mask pixels must have HSV hue in [0–15] or [165–180].
    - For others: mean BGR distance <120.
    Draws white outlines (thickness 4) on the original image and returns it.
    """
    # 1. Prepare boosted image for clarity (same as before)
    print("🎨 Boosting colors before segmentation...")
    hsv_orig = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    hsv_boost = hsv_orig.copy()
    hsv_boost[:, :, 1] = np.clip(hsv_boost[:, :, 1] * 2.0, 0, 255)
    hsv_boost[:, :, 2] = np.clip(hsv_boost[:, :, 2] * 2.0, 0, 255)
    boosted_bgr = cv2.cvtColor(hsv_boost.astype(np.uint8), cv2.COLOR_HSV2BGR)
    boosted_rgb = cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2RGB)

    # 2. Load SAM
    checkpoint_path = "sam_vit_b.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Missing checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Loading SAM model on {device}...")
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam_model.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # 3. Generate masks
    print("🔍 Generating masks with SAM (on boosted image)...")
    masks = mask_generator.generate(boosted_rgb)
    print(f"✅ {len(masks)} masks produced.")

    # 4. Filter masks
    print(f"🎯 Filtering masks for '{color_name}' (balanced check)...")
    kept_masks = []

    if color_name.lower() == "red":
        hsv_boost_full = cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2HSV)
        for mask in tqdm(masks, desc="Filtering (red)", leave=False):
            seg = mask["segmentation"]
            if not seg.any():
                continue
            hue_vals = hsv_boost_full[:, :, 0][seg]
            total = hue_vals.size
            if total == 0:
                continue
            count_red = np.count_nonzero((hue_vals <= 15) | (hue_vals >= 165))
            if (count_red / total) >= 0.10:  # at least 10% red
                kept_masks.append(seg)
        print(f"🗂️ {len(kept_masks)} masks deemed 'red' (balanced).")
    else:
        color_map = {
            "green":  np.array([  0, 255,   0]),
            "blue":   np.array([255,   0,   0]),
            "yellow": np.array([  0, 255, 255]),
            "orange": np.array([  0, 165, 255]),
            "purple": np.array([128,   0, 128]),
            "white":  np.array([255, 255, 255]),
            "black":  np.array([  0,   0,   0]),
        }
        cname = color_name.lower()
        if cname not in color_map:
            raise ValueError(f"Unsupported color: {color_name}")
        target_bgr = color_map[cname].astype(float)
        for mask in tqdm(masks, desc=f"Filtering ({color_name})", leave=False):
            seg = mask["segmentation"]
            pixels = boosted_bgr[seg]
            if pixels.size == 0:
                continue
            mean_bgr = pixels.mean(axis=0)
            if np.linalg.norm(mean_bgr - target_bgr) < 120:  # threshold 120
                kept_masks.append(seg)
        print(f"🗂️ {len(kept_masks)} masks match '{color_name}' (BGR <120).")

    # 5. Draw thicker white contours on original image
    print("🎨 Drawing contours (thickness 4) on matched masks...")
    outlined = image.copy()
    for seg in tqdm(kept_masks, desc="Outlining", leave=False):
        bin_mask = (seg.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(outlined, contours, -1, (255, 255, 255), 8)

    print("✅ Done. Returning outlined image.")
    return outlined

if __name__ == "__main__":
    img_path = "input_images/apple.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing input image: {img_path}")

    print("🖼️ Loading image for testing...")
    frame = cv2.imread(img_path)

    print("▶️ Running sam(...) with target color 'red' (balanced)...")
    result = sam(frame, "red")

    print("💾 Saving result to 'sam.png'...")
    cv2.imwrite("sam.png", result)
    print("✅ Result saved.")
