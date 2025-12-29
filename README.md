## meme-gesture-cv
A fun real-time computer vision project that displays memes based on facial expressions and hand gestures using a webcam.

This project uses **MediaPipe** for face, hand, and pose landmark detection and **OpenCV** for real-time video processing.

---

## 🚀 Features

### 👤 Face-Based Gestures
- **Eyes closed + head tilt** 
- **Eyes closed + lips tucked** 

### 🙆 Pose-Based Gesture
- **Hands on head (wrists above ears)** 

### ✋ Hand-Based Gestures
- **One finger up** 
- **Finger near mouth** 

Gestures are stabilized using frame counters to reduce flickering.

---

## 🧠 Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

---

## 📁 Project Structure
```text
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── memes/
    ├── dogmeme.jpg
    ├── meme.jpg
    ├── suprised.jpg
    ├── idea.jpg
    └── thinking.jpg
```text

---

## ⚙️ Setup & Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
---

## 📌 Notes

- This project was created purely for **fun and learning purposes**.
- It is **not intended to offend, mock, or hurt anyone** in any way.
- Gesture detection is rule-based and may vary depending on lighting, camera quality, and user positioning.
- AI tools were used during development to assist with coding, debugging, and experimentation as part of the learning process.