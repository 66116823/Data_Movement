import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  # <--- (ใหม่) ตัวช่วยปรับสเกลข้อมูล
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau # <--- (ใหม่) ตัวช่วยลด LR อัตโนมัติ

# 🔥 เรียกใช้ฟังก์ชันจากไฟล์ data_loader.py (ใช้ไฟล์เดิมได้เลย)
from data_loader import load_data

# ==========================================
# 1. CONFIGURATION (ตั้งค่า)
# ==========================================
# Path ข้อมูล (แก้ให้ถูกต้อง ไม่มีช่องว่างนำหน้า)
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"

WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 60           # เพิ่มรอบหน่อย เพราะเราจะให้มันค่อยๆ เรียนรู้
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

# สร้างโฟลเดอร์ Output ถ้ายังไม่มี
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 สร้างโฟลเดอร์ใหม่: {OUTPUT_DIR}")

# ==========================================
# 2. DATA PREPARATION & NORMALIZATION
# ==========================================
print("🔄 Loading Data...")
X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล! กรุณาตรวจสอบ Path หรือชื่อไฟล์")
    exit()

# แปลง Label เป็น One-Hot
y_onehot = to_categorical(y, num_classes=3)

# แบ่งข้อมูล Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training Data (Raw): {X_train.shape}")

# 🔥🔥🔥 (ใหม่) ส่วนสำคัญ: NORMALIZATION 🔥🔥🔥
# ปรับสเกลข้อมูลให้เป็นมาตรฐานเดียวกัน (Mean=0, Std=1) 
# จะช่วยให้โมเดลไม่สับสนค่า BMI ที่สูงกว่าค่า Sensor มากๆ

scaler = StandardScaler()

# แปลง 3D (Samples, Time, Features) -> 2D (Samples*Time, Features) เพื่อให้ Scaler ทำงานได้
N_train, T, F = X_train.shape
X_train_reshaped = X_train.reshape(-1, F)
X_train_scaled = scaler.fit_transform(X_train_reshaped) # คำนวณสูตรจาก Train Set
X_train = X_train_scaled.reshape(N_train, T, F)         # แปลงกลับเป็น 3D

# ใช้สูตรเดิม (จาก Train) มาปรับ Test Set (ห้าม Fit ใหม่กับ Test)
N_test, T, F = X_test.shape
X_test_reshaped = X_test.reshape(-1, F)
X_test = scaler.transform(X_test_reshaped).reshape(N_test, T, F)

print("✅ Data Normalized (Scaled) เรียบร้อย! (พร้อมเทรนแล้ว)")

# ==========================================
# 3. BUILD MODEL
# ==========================================
model = Sequential()

# CNN Layers (ตาดู Pattern)
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 4)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))

# LSTM Layers (สมองจำลำดับ)
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.5))

# Output Layers
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

# 🔥 (ใหม่) เพิ่ม Callback: ถ้า Loss ไม่ลดลง 5 รอบ ให้ลด Learning Rate ลงครึ่งนึง
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print("\n🚀 START TRAINING...")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test), 
    callbacks=[reduce_lr],  # 👈 สั่งให้ใช้ตัวช่วยลด LR
    verbose=1
)

# ==========================================
# 5. EVALUATION & SAVING
# ==========================================
print("\n📝 SAVING RESULTS...")

# 5.1 กราฟ Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy (Normalized)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Normalized)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_result_v2.png")) # เซฟชื่อไฟล์ v2
plt.show()

# 5.2 Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_v2.png")) # เซฟชื่อไฟล์ v2
plt.show()

# 5.3 Save Model
model.save(os.path.join(OUTPUT_DIR, "gait_risk_model_v2.h5"))
print(f"\n💾 บันทึกโมเดลเวอร์ชัน V2 สำเร็จที่: {OUTPUT_DIR}")
