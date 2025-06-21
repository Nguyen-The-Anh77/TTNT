# Emotion Recognition

Ứng dụng sẽ dự đoán cảm xúc của bạn từ webcam và hiện thị trạng thái đồng thời lên khung hình.

<!-- <p align="center">
	<img src="https://media1.giphy.com/media/J11DaTiOI7a9BtkxG0/giphy.gif?cid=790b76110e10226fb5d26ed94c18522b30e1126c15ba0804&rid=giphy.gif&ct=g" />
</p> -->

# How it work

Những ứng dụng dự đoán cảm xúc, dự đoán tuổi, dự đoán giới tính... bây giờ đã không còn quá xa lạ với mọi người. Thậm chí thư viện `deepface` còn giúp chúng ta thực hiện những việc tưởng chừng khó khăn đó chỉ với duy nhất một dòng code:

```python
obj = DeepFace.analyze(img_path = "img4.jpg", actions = ['age', 'gender', 'race', 'emotion'])
```

<p align="center">
	<img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" />
</p>

Nhưng để tìm hiểu sâu hơn cách thức những dòng code này hoạt động thì trong bài viết này chúng ta sẽ đi từ huấn luyện mô hình đến xây dựng ứng dụng nhận diện cảm xúc theo thời gian thực.

## Huấn luyện mô hình

### Bước 1. Nạp các thư việc cần thiết cho quá trình huấn luyện

```python
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras import layers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import cv2
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
```

### Bước 2: Khai báo các tham số

```python
dataset_path = 'datasets/fer2013/fer2013.csv'
base_path = "models/" # Nơi lưu log và model sau khi train
image_size = (48,48)
batch_size = 32 # steps_per_epoch = len(train) / batch_size. batch_size càng nhỏ thì số ảnh trên mỗi vòng huấn luyện sẽ tăng lên, như thế thời gian huấn luyện sẽ kéo dài
num_epochs = 1000 # Số vòng huấn luyện
input_shape = (48, 48, 1)
validation_split = .2 # Tỉ lệ số lượng test/train
verbose = 1 # Status hiển thị khi train. 0 -> Không hiển thị gì
num_classes = 7 # ["Tuc gian" ,"Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]
patience = 50 # Hệ số trong hàm callbacks
```

### Bước 3: Hàm load datasets và chuẩn hoá hình ảnh

```python
def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = image_size

        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.array(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.array(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).values
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
```

```python
 faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
```

### Bước 5: Chia tệp huấn luyện và tệp kiểm thử theo tỷ lệ 0.2 mà ta đã thiết lập ở trên

```python
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

# suffle=True -> Xáo trộn tệp dữ liệu lên để nó có tính tổng quan hơn
```

### Bước 6: Khởi tạo model và thêm các lớp ẩn

```python
model = Sequential()
model.add(Conv2D(32,(3,3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(7))
model.add(Activation("softmax"))

model.summary()
```

Sau khi thêm các lớp, ta được tổng số params như hình:

<p align="center">
	<img src="https://i.imgur.com/kJjHg4C.png" />
</p>

### Bước 7: Khởi tạo Data Generator

```python
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
```

### Bước 8: Khởi tạo hàm callbacks

```python
# Nơi lưu file log và models
log_file_path = base_path + 'emotion_training.log'
model_names = base_path + 'emotion_model.accuracy_{val_acc:.2f}.hdf5'

# Khởi tạo Callback cho quá trình train
csv_logger = CSVLogger(log_file_path, append=False) # Lưu log
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
```

### Bước 9: Biên dịch mô hình

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Bước 10: Khởi động quá trình huấn luyện

```python
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=(xtest,ytest))
```

<p align="center">
	<img src="https://i.imgur.com/fvVrQZO.png" />
</p>

## Viết chương trình nhận diện khuôn mặt và dự đoán cảm xúc

Xong các bước trên là chúng ta đã có 1 mô hình dự đoán cảm xúc, công việc bây giờ là đưa nào vào trong ứng dụng của mình để dự đoán.

Các bước để thực hiện sẽ là:

-   Nhận diện khuôn mặt bằng Cascade Classifier
-   Cắt khu vực ảnh đó và xử lý đầu vào phù hợp như ảnh đầu vào của mô hình chúng ta huấn luyện ở trên (có kích thước 48x48)
-   Dùng model dự đoán cảm xúc của khuôn mặt và hiển thị nó ra

Chi tiết code tại [app.py]

# Installation

Để khởi cài đặt các thư viện cần thiết các bạn chỉ cần cài đặt `requirements.txt`:

```
pip install requirements.txt
```

# Usage

Để khởi chạy ứng dụng:

```
git clone https://github.com/KudoKhang/EmotionRecognition
cd EmotionRecognition
python app.py
```

Để sử dụng GUI của Gradio:

```
python gradio-app.py
```

<!-- <p align="center">
	<img src="https://i.imgur.com/VcX9nOs.png" />
</p> -->

Kết quả cuối cùng:

<!-- <p align="center">
	<img src="https://media1.giphy.com/media/J11DaTiOI7a9BtkxG0/giphy.gif?cid=790b76110e10226fb5d26ed94c18522b30e1126c15ba0804&rid=giphy.gif&ct=g" />
</p> -->

# Summary

Kể cả đối với chúng ta thì việc nhận biết cảm xúc của người khác cũng không phải lúc nào cũng chính xác 100% đặc biệt là những biểu cảm khó phân biệt nên chỉ nhìn vào 1 bức ảnh như KINH NGẠC hoặc SỢ HÃI... Model với độ chính xác chỉ 65% của chúng ta thì việc nhận diện nhầm những cảm xúc đó cũng khá dễ hiểu, thay vào đó nó dễ dàng nhận diện 2 cảm xúc có sự khác biệt rõ ràng nhất đó là HẠNH PHÚC và BÌNH THƯỜNG."# AI-nh-n-di-ncmxc"
