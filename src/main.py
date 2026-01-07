import cv2
import mediapipe as mp
import numpy as np
import math

# ===================== MediaPipe =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===================== Camera =====================
cap = cv2.VideoCapture(0)

# ===================== Nail Landmark Pairs =====================
NAIL_PAIRS = {
    "thumb": (4, 3),
    "index": (8, 7),
    "middle": (12, 11),
    "ring": (16, 15),
    "pinky": (20, 19),
}

# ===================== EMA =====================
EMA_ALPHA = 0.3
smooth = {}

# ===================== Load Designs =====================
DESIGNS = {
    "short": cv2.imread("designs/french.png", cv2.IMREAD_UNCHANGED),
    "normal": cv2.imread("designs/classic.png", cv2.IMREAD_UNCHANGED),
    "long": cv2.imread("designs/glitter.png", cv2.IMREAD_UNCHANGED),
}

for k, v in DESIGNS.items():
    if v is None or v.shape[2] != 4:
        raise ValueError(f"{k} design must be RGBA PNG")

# ===================== Helpers =====================
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def ema(prev, curr):
    return EMA_ALPHA * curr + (1 - EMA_ALPHA) * prev

def classify_nail(ratio):
    if ratio < 1.6:
        return "short"
    elif ratio < 2.2:
        return "normal"
    else:
        return "long"

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def overlay_png(frame, png, center):
    x, y = center
    h, w = png.shape[:2]

    # Frame bounds
    fh, fw = frame.shape[:2]

    # Compute overlay region in frame
    x1 = max(0, x - w // 2)
    y1 = max(0, y - h // 2)
    x2 = min(fw, x + w // 2)
    y2 = min(fh, y + h // 2)

    # Compute corresponding region in PNG
    px1 = max(0, (w // 2) - x)
    py1 = max(0, (h // 2) - y)
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    # ✅ SAFETY CHECK (THIS FIXES YOUR CRASH)
    if x1 >= x2 or y1 >= y2 or px1 >= px2 or py1 >= py2:
        return  # skip invalid overlay

    roi = frame[y1:y2, x1:x2]
    png_crop = png[py1:py2, px1:px2]

    # Alpha blending
    alpha = png_crop[:, :, 3:4] / 255.0
    frame[y1:y2, x1:x2] = (
        roi * (1 - alpha) + png_crop[:, :, :3] * alpha
    ).astype(frame.dtype)


# ===================== Main Loop =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for finger, (tip_id, dip_id) in NAIL_PAIRS.items():
                tip = hand.landmark[tip_id]
                dip = hand.landmark[dip_id]

                tip_pt = (int(tip.x * w), int(tip.y * h))
                dip_pt = (int(dip.x * w), int(dip.y * h))

                dx, dy = tip_pt[0] - dip_pt[0], tip_pt[1] - dip_pt[1]
                length = math.hypot(dx, dy)
                if length == 0:
                    continue
                CUTICLE_SHIFT = 0.35  # 🔥 critical
                cx = int(tip_pt[0] * (1 - CUTICLE_SHIFT) + dip_pt[0] * CUTICLE_SHIFT)
                cy = int(tip_pt[1] * (1 - CUTICLE_SHIFT) + dip_pt[1] * CUTICLE_SHIFT)

                perp_dx, perp_dy = -dy, dx
                norm = math.hypot(perp_dx, perp_dy)
                perp_dx /= norm
                perp_dy /= norm

                WIDTH_SCALE = 0.35
                wx = int(perp_dx * length * WIDTH_SCALE)
                wy = int(perp_dy * length * WIDTH_SCALE)

                width = distance((cx - wx, cy - wy), (cx + wx, cy + wy))
                if width == 0:
                    continue

                ratio = length / width

                if finger not in smooth:
                    smooth[finger] = {"l": length, "w": width, "r": ratio}
                else:
                    smooth[finger]["l"] = ema(smooth[finger]["l"], length)
                    smooth[finger]["w"] = ema(smooth[finger]["w"], width)
                    smooth[finger]["r"] = ema(smooth[finger]["r"], ratio)

                s_len = smooth[finger]["l"]
                s_wid = smooth[finger]["w"]
                s_rat = smooth[finger]["r"]

                shape = classify_nail(s_rat)
                design = DESIGNS[shape]

                DESIGN_WIDTH_SCALE = 1.3
                DESIGN_HEIGHT_SCALE = 1.1

                design_resized = cv2.resize(
                    design,
                    (
                        int(s_wid * DESIGN_WIDTH_SCALE),
                        int(s_len * DESIGN_HEIGHT_SCALE)
                    ),
                    interpolation=cv2.INTER_AREA
                )


                angle = -math.degrees(math.atan2(dy, dx)) + 90
                if finger == "thumb":
                    angle -= 10  # thumb correction

                design_rot = rotate_image(design_resized, angle)

                overlay_png(frame, design_rot, (cx, cy))

    cv2.imshow("AI Nail Try-On", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
