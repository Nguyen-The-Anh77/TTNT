import cv2
import numpy as np 
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array

from keras.models import load_model

face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/model_10epoch_val_acc_0.638.h5', compile=False)
EMOTIONS = ["Tuc gian" ,"Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()   
    frame = cv2.flip(frame, 1) 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    
    # Chỉ thực hiện nhận biết cảm xúc khi phát hiện được có khuôn mặt trong hình
    if len(faces) > 0:
        # Chỉ thực hiện với khuôn mặt chính trong hình (khuôn mặt có diện tích lớn nhất)
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48, vì mạng mình train có đầu vào là 48x48
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        cv2.rectangle(frame, (fX, fY + fH), (fX + fW, fY + fH + 25), (0,255,0), -1)
        cv2.putText(frame, label, (fX + 6 , fY + fH + 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()