import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# پیش‌پردازش داده‌ها
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (96, 96)) # اندازه استاندارد
    normalized_image = resized_image / 255.0
    return np.reshape(normalized_image, (96, 96, 1))

# ساخت مدل CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D(2, 2),
    # ادامه لایه‌های مدل
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # برای تشخیص دو کلاس (چهره/غیر چهره)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
# توجه: باید داده‌های آموزشی و تست را فراهم کنید
train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_image)
train_data = train_data_generator.flow_from_directory('path_to_training_data', target_size=(96, 96))

model.fit(train_data, epochs=10) # تعداد epochs بسته به نیاز تنظیم می‌شود
