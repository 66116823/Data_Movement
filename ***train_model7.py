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
from sklearn.utils.class_weight import compute_class_weight # <--- เพิ่มตัวช่วยคำนวณ

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

from data_loader import load_data

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"

WINDOW_SIZE = 256
STEP_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. LOAD DATA
# ==========================================
print(f"\n🔄 Loading Data (Window {WINDOW_SIZE})...")
X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล!")
    exit()

y_onehot = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. CLASS WEIGHTS (AUTO BALANCED)
# ==========================================
# เปลี่ยนจากกำหนดเอง เป็นให้สูตรคำนวณตามสัดส่วนข้อมูลจริง
# เพื่อไม่ให้โมเดลลำเอียงไปทาง High Risk มากเกินไปจนทิ้ง Low/Medium
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n⚖️ Auto Balanced Weights: {class_weight_dict}")
# ผลลัพธ์คาดการณ์: ค่าจะใกล้เคียงกัน เช่น {0: 1.x, 1: 0.x, 2: 0.x} ไม่โดดไป 5.0 แล้ว

# ==========================================
# 4. NORMALIZATION
# ==========================================
scaler = StandardScaler()
N_train, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_train, T, F)

N_test, T, F = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

# ==========================================
# 🔥 5. BUILD MODEL (V7 Adjusted)
# ==========================================
model = Sequential()

# CNN Part
# เพิ่ม Filter ชั้นแรกเป็น 128 เพื่อเก็บรายละเอียด Waveform ให้มากขึ้น
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, F)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Bi-LSTM Part
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.4))

# Dense Part
# ลด L2 ลงเหลือ 0.001 (เดิม 0.01) เพื่อลดแรงกดดัน ให้โมเดลเรียนรู้คลาสยากๆ ได้ดีขึ้น
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

model.summary()

# ==========================================
# 6. TRAIN
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

print("\n🚀 START TRAINING (V7 - Balanced Weights)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop],
    class_weight=class_weight_dict, # ใช้ค่าที่คำนวณอัตโนมัติ
    verbose=1
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================
print("\n📝 Generating Graphs...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy (V7 Balanced)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (V7 Balanced)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "result_v7_balanced_graph.png"))
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report (V7 Balanced) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V7 Balanced)')
plt.savefig(os.path.join(OUTPUT_DIR, "result_v7_balanced_cm.png"))
plt.show()

model.save(os.path.join(OUTPUT_DIR, "gait_model_v7_balanced.h5"))
print(f"\n💾 Saved V7 to: {OUTPUT_DIR}")
