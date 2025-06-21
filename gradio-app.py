import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, Tk, Label, Button, Frame
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Tải mô hình và cascade khuôn mặt
face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")
emotion_model = load_model("files/model_10epoch_val_acc_0.638.h5", compile=False)
EMOTIONS = ["Tức giận", "Kinh tởm", "Sợ hãi", "Hạnh phúc", "Buồn", "Bất ngờ", "Bình thường"]

def detect_emotion(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Không thể đọc ảnh."
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return "Không phát hiện khuôn mặt."

    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_model.predict(roi, verbose=0)[0]
    return f"Cảm xúc phát hiện: {EMOTIONS[np.argmax(preds)]}"

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        result = detect_emotion(file_path)
        result_label.config(text=result)

        # Hiển thị ảnh đã chọn
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Giữ ảnh trong bộ nhớ

# Tạo ứng dụng GUI
app = Tk()
app.title("🎭 Nhận diện cảm xúc khuôn mặt (Offline)")
app.geometry("520x450")
app.configure(bg="#ff69b4")

frame = Frame(app, bg="#ff69b4")
frame.pack(pady=10)

title_label = Label(frame, text="Chọn ảnh để nhận diện cảm xúc", font=("Segoe UI", 14, "bold"), bg="#ff69b4")
title_label.pack(pady=10)

browse_button = Button(frame, text="Tải ảnh", command=browse_file, font=("Segoe UI", 12), bg="#4CAF50", fg="white", padx=20, pady=5)
browse_button.pack(pady=5)

# Kết quả
result_label = Label(frame, text="", font=("Segoe UI", 14), bg="#ff69b4", fg="#FFF")
result_label.pack(pady=10)

# Hiển thị ảnh
image_label = Label(app, bg="#f2f2f2")
image_label.pack(pady=10)

app.mainloop()
