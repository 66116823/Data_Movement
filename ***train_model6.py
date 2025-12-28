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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional # <--- เพิ่มตัวนี้
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2 # <--- เพิ่มตัวนี้ (Hyperparameter)

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
# 3. MANUAL CLASS WEIGHTS
# ==========================================
# ยังคงใช้สูตรเดิม เพราะมัน Safety First ดีแล้ว
manual_weights = {
    0: 1.0,
    1: 1.5,
    2: 5.0
}
print("\n⚖️ Weights:", manual_weights)

# ==========================================
# 4. NORMALIZATION
# ==========================================
scaler = StandardScaler()
N_train, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_train, T, F)

N_test, T, F = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

# ==========================================
# 🔥 5. BUILD MODEL (CNN + Bi-LSTM)
# ==========================================
model = Sequential()

# CNN Part (สกัด Feature)
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, F)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3)) # ปรับ Dropout ลดลงนิดนึง

# Bi-LSTM Part (พระเอกของเรา)
# ใช้ Bidirectional หุ้ม LSTM ไว้
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.4))

# Dense Part (Classifier)
# ใส่ L2 Regularizer (0.01) ช่วยคุมไม่ให้จำข้อสอบ
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

# ==========================================
# 6. TRAIN
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

print("\n🚀 START TRAINING (V6 - CNN + Bi-LSTM)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop],
    class_weight=manual_weights,
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
plt.title('Accuracy (CNN + Bi-LSTM)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (CNN + Bi-LSTM)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "result_v6_bilstm_graph.png"))
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report (V6 Bi-LSTM) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V6 Bi-LSTM)')
plt.savefig(os.path.join(OUTPUT_DIR, "result_v6_bilstm_cm.png"))
plt.show()

model.save(os.path.join(OUTPUT_DIR, "gait_model_v6_bilstm.h5"))
print(f"\n💾 Saved V6 to: {OUTPUT_DIR}")
