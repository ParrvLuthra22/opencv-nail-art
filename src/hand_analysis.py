import cv2
import mediapipe as mp
import math

# ================= MediaPipe =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ================= Landmark Groups =================
PALM_POINTS = [0, 1, 5, 9, 13, 17]

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# ================= Utility =================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

# ================= Main Loop =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm = hand.landmark

            # --------- Convert landmarks to pixels ---------
            pts = [(int(l.x * w), int(l.y * h)) for l in lm]

            # ================= PALM ANALYSIS =================
            palm_pts = [pts[i] for i in PALM_POINTS]

            palm_cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
            palm_cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))

            palm_width = dist(pts[5], pts[17])
            palm_angle = angle(pts[0], pts[9])

            cv2.circle(frame, (palm_cx, palm_cy), 6, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"Palm width: {int(palm_width)}",
                (palm_cx - 40, palm_cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

            # ================= FINGER ANALYSIS =================
            for finger, idxs in FINGERS.items():
                base = pts[idxs[0]]
                tip = pts[idxs[-1]]

                finger_len = dist(base, tip)
                finger_angle = angle(base, tip)

                # Draw finger axis
                cv2.line(frame, base, tip, (0, 255, 0), 2)

                cx = (base[0] + tip[0]) // 2
                cy = (base[1] + tip[1]) // 2

                cv2.putText(
                    frame,
                    f"{finger}: {int(finger_len)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

            # ================= DRAW SKELETON =================
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Full Hand Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
