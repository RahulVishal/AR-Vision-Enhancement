import cv2
import numpy as np
import os
import shutil

# Setup output folder
out_dir = "outputs1"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

def save(name, im):
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), im)

# Load input image
input_path = "image copy.png"
img = cv2.imread(input_path)

# 1. Tunnel vision: dim + blur around edges
def tunnel_vision(im):
    h, w = im.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (w//2, h//2), min(w,h)//2, 1, -1)
    mask = cv2.GaussianBlur(mask, (101,101), 0)
    dark = cv2.convertScaleAbs(im, alpha=0.7, beta=-20)
    result = (dark * mask[...,None]).astype(np.uint8)
    return cv2.GaussianBlur(result, (9,9), 2)

# 2. Night blindness: dimmed brightness and color
def night_blindness(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] *= 0.4
    hsv[:,:,1] *= 0.5
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rp_dark(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.blur(l, (15,15))
    l = np.clip(l * 0.3, 0, 255).astype(np.uint8)
    a = cv2.blur(a, (15,15))
    b = cv2.blur(b, (15,15))
    a = np.clip((a - 128) * 0.3 + 128, 0, 255).astype(np.uint8)  # dull red-green
    b = np.clip((b - 128) * 0.3 + 128, 0, 255).astype(np.uint8)  # dull blue-yellow
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)



# 3b. Retinitis Pigmentosa (light): bright glare in highlights
def rp_light(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.where(l > 128, np.clip(l * 1.4, 0, 255), l).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# 4. Achromatopsia: full grayscale
def color_blindness(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 5. Macular degeneration: fuzzy central blur
def macular_degeneration(im):
    h, w = im.shape[:2]
    noise = np.random.normal(0, 15, (h, w)).astype(np.uint8)
    blob = np.zeros((h, w), np.uint8)
    center = (w//2 + np.random.randint(-30, 30), h//2 + np.random.randint(-30, 30))
    axes = (w//6 + np.random.randint(-20,20), h//6 + np.random.randint(-20,20))
    angle = np.random.randint(0, 360)
    cv2.ellipse(blob, center, axes, angle, 0, 360, 255, -1)
    blob = cv2.GaussianBlur(blob, (101,101), 0)
    blur = cv2.GaussianBlur(im, (51,51), 3)
    result = im.copy()
    result[blob > 40] = blur[blob > 40]
    return result

# 6. Blind spots: irregular fuzzy scotomas
def blind_spots(im, n=4):
    out = im.copy()
    h, w = out.shape[:2]
    for _ in range(n):
        spot = np.zeros((h, w), np.uint8)
        center = (np.random.randint(w), np.random.randint(h))
        axes = (np.random.randint(20, 60), np.random.randint(20, 60))
        angle = np.random.randint(0, 360)
        cv2.ellipse(spot, center, axes, angle, 0, 360, 255, -1)
        spot = cv2.GaussianBlur(spot, (51,51), 0)
        mask = spot > 30
        out[mask] = out[mask] * np.random.uniform(0.0, 0.2)
    return out.astype(np.uint8)

# 7. Cataracts: yellow haze + blur
def cataracts(im):
    haze = im.astype(float)
    haze[:,:,2] += 40  # add yellow
    haze = np.clip(haze,0,255).astype(np.uint8)
    return cv2.GaussianBlur(haze, (25,25), 5)

# 8. Diabetic retinopathy: tiny dark spots
def diabetic_retinopathy(im):
    out = im.copy()
    h,w = out.shape[:2]
    for _ in range(60):
        x,y = np.random.randint(0,w), np.random.randint(0,h)
        r = np.random.randint(2,5)
        color = tuple(np.random.randint(0,40) for _ in range(3))
        cv2.circle(out, (x,y), r, color, -1)
    return out

# Save all variations
save('tunnel_vision',          tunnel_vision(img))
save('night_blindness',        night_blindness(img))
save('rp_dark',                rp_dark(img))
save('rp_light',               rp_light(img))
save('color_blind',            color_blindness(img))
save('macular_degeneration',   macular_degeneration(img))
save('blind_spots',            blind_spots(img))
save('cataracts',              cataracts(img))
save('diabetic_retinopathy',   diabetic_retinopathy(img))
