# Import thư viện cần thiết
import cv2  # Thư viện xử lý ảnh/video
import numpy as np  # Thư viện tính toán ma trận, xử lý ảnh
from tensorflow.keras.preprocessing.image import img_to_array  # Chuyển ảnh sang mảng đầu vào cho mô hình
from keras.models import load_model  # Dùng để tải mô hình đã huấn luyện

# Tải bộ phân loại khuôn mặt Haar cascade
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')

# Tải mô hình phân loại cảm xúc đã được huấn luyện
emotion_classifier = load_model('files/model_10epoch_val_acc_0.638.h5', compile=False)

# Danh sách các nhãn cảm xúc tương ứng với mô hình
EMOTIONS = ["Tuc gian", "Kinh tom", "So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]

# Mở webcam để bắt đầu quay video (0 là webcam mặc định)
camera = cv2.VideoCapture(0)

# Vòng lặp chính để xử lý từng khung hình
while True:
    ret, frame = camera.read()  # Đọc một khung hình từ webcam
    frame = cv2.flip(frame, 1)  # Lật ảnh theo chiều ngang (giống như soi gương)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh màu sang ảnh xám để xử lý khuôn mặt

    # Phát hiện khuôn mặt trong ảnh xám
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))

    # Nếu phát hiện ít nhất một khuôn mặt
    if len(faces) > 0:
        # Chọn khuôn mặt lớn nhất (gần camera nhất)
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face  # Tọa độ và kích thước khuôn mặt

        # Cắt vùng ảnh chứa khuôn mặt
        roi = gray[fY:fY + fH, fX:fX + fW]

        # Resize ảnh về kích thước 48x48 (phù hợp với đầu vào của mô hình)
        roi = cv2.resize(roi, (48, 48))

        # Chuẩn hóa ảnh (về khoảng [0,1]) và chuyển thành dạng mảng numpy
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)

        # Thêm một chiều để tạo batch (1 ảnh)
        roi = np.expand_dims(roi, axis=0)

        # Dự đoán cảm xúc bằng mô hình đã huấn luyện
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]  # Lấy nhãn cảm xúc có xác suất cao nhất

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        # Vẽ khung màu xanh bên dưới khuôn mặt để hiển thị tên cảm xúc
        cv2.rectangle(frame, (fX, fY + fH), (fX + fW, fY + fH + 25), (0,255,0), -1)

        # Ghi tên cảm xúc lên ảnh
        cv2.putText(frame, label, (fX + 6 , fY + fH + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị ảnh đã xử lý lên cửa sổ
    cv2.imshow('Emotion Recognition', frame)

    # Nếu người dùng nhấn phím 'q' thì thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng các cửa sổ
camera.release()
cv2.destroyAllWindows()
