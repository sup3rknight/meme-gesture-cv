import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

# Load memes
dog = cv2.resize(cv2.imread("memes/dogmeme.jpg"), (300, 300))
meme = cv2.resize(cv2.imread("memes/meme.jpg"), (300, 300))
surprised = cv2.resize(cv2.imread("memes/suprised.jpg"), (300, 300))

idea = cv2.resize(cv2.imread("memes/idea.jpg"), (300, 300))
thinking = cv2.resize(cv2.imread("memes/thinking.jpg"), (300, 300))
stop = cv2.resize(cv2.imread("memes/stop.jpg"), (300, 300))
shaq = cv2.resize(cv2.imread("memes/ShaqMeme.jpg"), (300, 300))

counter = {
    "dog": 0, "meme": 0, "surprised": 0,
    "idea": 0, "thinking": 0, "stop": 0, "shaq": 0
}
FRAME_THRESHOLD = 10

def count_fingers(hand):
    tips = [8, 12, 16, 20]
    fingers = 0
    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers += 1
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    pose_result = pose.process(rgb)
    hand_result = hands.process(rgb)

    show_meme = None

    # ================= FACE =================
    if face_result.multi_face_landmarks:
        lm = face_result.multi_face_landmarks[0].landmark

        eyes_closed = (
            abs(lm[159].y - lm[145].y) < 0.018 and
            abs(lm[386].y - lm[374].y) < 0.018
        )

        eye_width = abs(lm[33].x - lm[263].x)
        inner_lip = abs(lm[13].y - lm[14].y)

        eye_mid_x = (lm[33].x + lm[263].x) / 2
        nose_shift = abs(lm[1].x - eye_mid_x) / eye_width

        if eyes_closed and nose_shift > 0.16:
            counter["dog"] += 1
        else:
            counter["dog"] = 0

        if eyes_closed and inner_lip < 0.012:
            counter["meme"] += 1
        else:
            counter["meme"] = 0

    # ================= POSE (SURPRISED) =================
    if pose_result.pose_landmarks:
        plm = pose_result.pose_landmarks.landmark
        if (
            plm[mp_pose.PoseLandmark.LEFT_WRIST].y <
            plm[mp_pose.PoseLandmark.LEFT_EAR].y and
            plm[mp_pose.PoseLandmark.RIGHT_WRIST].y <
            plm[mp_pose.PoseLandmark.RIGHT_EAR].y
        ):
            counter["surprised"] += 1
        else:
            counter["surprised"] = 0

    # ================= HANDS =================
    if hand_result.multi_hand_landmarks:
        hands_list = hand_result.multi_hand_landmarks

        # IDEA (ONE FINGER)
        if len(hands_list) == 1 and count_fingers(hands_list[0]) == 1:
            counter["idea"] += 1
        else:
            counter["idea"] = 0

        # THINKING (FINGER NEAR MOUTH)
        if (
            len(hands_list) == 1 and
            face_result.multi_face_landmarks
        ):
            hand = hands_list[0]
            finger_tip = hand.landmark[8]
            mouth = face_result.multi_face_landmarks[0].landmark[13]

            dist = abs(finger_tip.x - mouth.x) + abs(finger_tip.y - mouth.y)

            if dist < 0.08:
                counter["thinking"] += 1
            else:
                counter["thinking"] = 0
    else:
        for k in ["idea", "thinking", "stop", "shaq"]:
            counter[k] = 0

    # ================= PRIORITY =================
    if counter["surprised"] >= FRAME_THRESHOLD:
        show_meme = surprised
    elif counter["thinking"] >= FRAME_THRESHOLD:
        show_meme = thinking
    elif counter["idea"] >= FRAME_THRESHOLD:
        show_meme = idea
    elif counter["dog"] >= FRAME_THRESHOLD:
        show_meme = dog
    elif counter["meme"] >= FRAME_THRESHOLD:
        show_meme = meme

    if show_meme is not None:
        frame[30:330, 30:330] = show_meme

    cv2.imshow("Action Meme Generator", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
