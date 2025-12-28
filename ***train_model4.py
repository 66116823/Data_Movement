import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

# ตั้งค่า Seed
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight  # <--- พระเอกของเราในรอบนี้

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from data_loader import load_data

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"
WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("\n🔄 Calling Data Loader...")
X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล!")
    exit()

# เก็บค่า y แบบตัวเลขไว้คำนวณ Weight ก่อนแปลงเป็น One-hot
y_integers = y.copy()

y_onehot = to_categorical(y, num_classes=3)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y_onehot, y_integers, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 🔥 3. COMPUTE CLASS WEIGHTS (จุดเปลี่ยนเกม)
# ==========================================
# คำนวณหาว่าคลาสไหนมีน้อย ให้เพิ่มน้ำหนักคลาสนั้นเยอะๆ
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weights_dict = dict(enumerate(class_weights))

print("\n⚖️ Class Weights (ถ่วงน้ำหนัก):")
for i, weight in class_weights_dict.items():
    print(f"   - {CLASSES[i]}: {weight:.4f}")
# ค่าที่ออกมา High Risk ควรจะได้ตัวเลขเยอะที่สุด (หรือพอๆ กับ Low)

# ==========================================
# 4. NORMALIZATION
# ==========================================
scaler = StandardScaler()
N_train, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_train, T, F)

N_test, T, F = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

# ==========================================
# 5. BUILD MODEL
# ==========================================
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, F)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

# ==========================================
# 6. TRAIN (ใส่ class_weight ลงไป)
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

print("\n🚀 START TRAINING (V4 - Weighted)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop],
    class_weight=class_weights_dict,  # 👈 ใส่ตัวถ่วงน้ำหนักตรงนี้!
    verbose=1
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================
# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report (V4 Weighted) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V4 Weighted)')
plt.savefig(os.path.join(OUTPUT_DIR, "result_v4_weighted_cm.png"))
plt.show()

model.save(os.path.join(OUTPUT_DIR, "gait_model_v4_weighted.h5"))
