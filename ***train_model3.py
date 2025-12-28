import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# 🔥 เพิ่ม EarlyStopping เข้ามา
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 

# เรียกใช้ data_loader ตัวใหม่ของคุณ (ที่มีการแปลง BMI)
from data_loader import load_data

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"

WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 32
# 🔥 ตั้ง Epoch ไว้สูงๆ ได้เลย เพราะเรามีตัวตัดจบแล้ว
EPOCHS = 100           
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("🔄 Loading Data (with BMI Categories 0,1,2)...")
X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล! กรุณาตรวจสอบ Path")
    exit()

# แปลง Label เป็น One-Hot
y_onehot = to_categorical(y, num_classes=3)

# แบ่งข้อมูล (Stratified เพื่อให้สัดส่วนแต่ละคลาสเท่าเดิม)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training Data Shape: {X_train.shape}")

# ==========================================
# 3. NORMALIZATION (ยังจำเป็นอยู่)
# ==========================================
# แม้ BMI จะเป็น 0,1,2 แล้ว แต่เราควรปรับให้เป็นมาตรฐานเดียวกับ Sensor (Mean=0, Std=1)
scaler = StandardScaler()

N_train, T, F = X_train.shape
X_train_reshaped = X_train.reshape(-1, F)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(N_train, T, F)

# ปรับ Test set ด้วย scaler ของ Train
N_test, T, F = X_test.shape
X_test_reshaped = X_test.reshape(-1, F)
X_test = scaler.transform(X_test_reshaped).reshape(N_test, T, F)

print("✅ Data Normalized เรียบร้อย")

# ==========================================
# 4. BUILD MODEL
# ==========================================
model = Sequential()

# CNN Layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, F)))
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
# 5. CALLBACKS (พระเอกของเรา)
# ==========================================
# 1. ลด Learning Rate ถ้าราบเรียบเกินไป
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# 2. 🔥 Early Stopping: ถ้าเทรนไป 15 รอบแล้ว Val Loss ไม่ดีขึ้น -> สั่งหยุด! และย้อนคืนค่าที่ดีที่สุด
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# ==========================================
# 6. TRAIN MODEL
# ==========================================
print("\n🚀 START TRAINING (with Early Stopping)...")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test), 
    callbacks=[reduce_lr, early_stop],  # ใส่ทั้งคู่เลย
    verbose=1
)

# ==========================================
# 7. EVALUATION
# ==========================================
print("\n📝 SAVING RESULTS (V3)...")

# กราฟ
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy (BMI Categorized)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (BMI Categorized)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_result_v3_bmi_cat.png"))
plt.show()

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V3 - BMI Categorized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_v3_bmi_cat.png"))
plt.show()

model.save(os.path.join(OUTPUT_DIR, "gait_risk_model_v3.h5"))
print(f"\n💾 เสร็จสมบูรณ์! บันทึกผลลัพธ์ไว้ที่: {OUTPUT_DIR}")
