import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 🔥 เรียกใช้ฟังก์ชันจากไฟล์ data_loader.py
from data_loader import load_data

# ==========================================
# 1. CONFIGURATION (ตั้งค่า)
# ==========================================
ROOT_DIR = r"E:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"E:\Data Movement\CollectDATA\Model"  # 👈 ที่เก็บผลลัพธ์

WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

# ตรวจสอบและสร้างโฟลเดอร์ Output อัตโนมัติ
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 สร้างโฟลเดอร์ใหม่เรียบร้อย: {OUTPUT_DIR}")
else:
    print(f"📂 พบโฟลเดอร์ Output: {OUTPUT_DIR}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล! กรุณาตรวจสอบ Path หรือชื่อไฟล์")
    exit()

y_onehot = to_categorical(y, num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training Data: {X_train.shape}")
print(f"🧪 Testing Data:  {X_test.shape}")

# ==========================================
# 3. BUILD MODEL (CNN-LSTM)
# ==========================================
model = Sequential()
# CNN Layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 4)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
# LSTM Layers
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.5))
# Output Layers
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print("\n🚀 START TRAINING...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

# ==========================================
# 5. EVALUATION & SAVING
# ==========================================
print("\n📝 SAVING RESULTS...")

# 5.1 กราฟ Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
acc_path = os.path.join(OUTPUT_DIR, "training_result.png")
plt.savefig(acc_path)
print(f"📈 บันทึกกราฟเทรนแล้วที่: {acc_path}")
plt.show()

# 5.2 Confusion Matrix (Heatmap)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

# วาดและบันทึก Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"📉 บันทึก Confusion Matrix แล้วที่: {cm_path}")
plt.show()

# 5.3 Save Model
model_path = os.path.join(OUTPUT_DIR, "gait_risk_model.h5")
model.save(model_path)
print("-" * 50)
print(f"✅✅✅ SUCCESS! บันทึกโมเดลเสร็จสมบูรณ์ที่: {model_path}")
print("-" * 50)