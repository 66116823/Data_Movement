import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================
# ⚙️ 1. CONFIGURATION (ตั้งค่าระบบ)
# ==========================================
INPUT_FILE = r"E:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_WithFile.csv"
OUTPUT_MODEL_PATH = r"E:\Data Movement\CollectDATA\Master_Data\Best_CNN_LSTM_Hybrid.h5"

# การตั้งค่า Window
TIME_STEPS = 120  # ต้องการ Input ยาว 120 เฟรม
STEP_SIZE = 10  # 🔥 Downsampling: ข้ามข้อมูลทีละ 10 แถว (ลดจาก 500Hz -> 50Hz)

# เลือก Feature ที่จะใช้ (ควรใช้ Z-score แล้ว)
# แนะนำให้ใช้ทั้งข้อมูลดิบและ Feature ที่คำนวณมา เพื่อให้ CNN จับ Pattern ได้ละเอียดสุด
FEATURES = [
    'ACC_Feature_Z',
    'GYRO_Feature_Z',
    # ถ้ามีคอลัมน์ Z-score ของแกนดิบ ให้ใส่เพิ่มตรงนี้จะแม่นขึ้นครับ เช่น:
    # 'ACC_X_Z', 'ACC_Y_Z', 'ACC_Z_Z'
]


# ==========================================
# 🛠️ 2. DATA PIPELINE (เตรียมข้อมูล)
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV กรุณาเช็ค Path")

    df = pd.read_csv(file_path)

    # 1. แก้ปัญหาข้อมูลซ้ำ (Duplicate Rows)
    print(f"🧹 Cleaning duplicates...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename', 'Time'])
    print(f"   - Reduced from {original_len} to {len(df)} rows")

    # 2. Group ตามไฟล์เพื่อเตรียมหั่นเป็นชิ้น
    grouped = df.groupby(['Subject_ID', 'Filename'])

    sequences = []
    labels = []
    subject_ids = []

    print(f"⏳ Processing & Downsampling (Step={step_size})...")

    for (sub_id, fname), group in grouped:
        # 3. Downsampling (ลดความละเอียดข้อมูล)
        # เลือกมา 1 แถว ทุกๆ 10 แถว เพื่อขยายมุมมองเวลาให้กว้างขึ้น
        group_ds = group.iloc[::step_size, :]

        series = group_ds[feature_cols].values

        # 4. Padding / Truncating (ปรับขนาดให้เท่ากับ 120)
        if len(series) == 0: continue

        if len(series) >= time_steps:
            seq = series[:time_steps]
        else:
            pad_len = time_steps - len(series)
            seq = np.pad(series, ((0, pad_len), (0, 0)), mode='constant')

        sequences.append(seq)
        labels.append(group['Label'].iloc[0])
        subject_ids.append(sub_id)

    print(f"✅ Preprocessing Done! Got {len(sequences)} sequences.")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# --- เรียกใช้ฟังก์ชัน ---
X, y, subjects = load_and_process_data(INPUT_FILE, TIME_STEPS, FEATURES, STEP_SIZE)

# ==========================================
# ✂️ 3. SUBJECT-BASED SPLITTING (แบ่งข้อมูลตามคน)
# ==========================================
print("\n🔄 Splitting Data (Subject-based)...")

unique_subjects = np.unique(subjects)
np.random.seed(42)
np.random.shuffle(unique_subjects)

# สูตรคำนวณ: Test 20%, Val 10%, Train Rest
n_total = len(unique_subjects)
n_test = 5  # ตามที่เราคุยกัน (ประมาณ 20% ของ 24)
n_val = 2  # (ประมาณ 10% ของที่เหลือ)
n_train = n_total - n_test - n_val

test_subs = unique_subjects[:n_test]
val_subs = unique_subjects[n_test: n_test + n_val]
train_subs = unique_subjects[n_test + n_val:]

print(f"   - Train Subjects ({len(train_subs)}): {train_subs}")
print(f"   - Val Subjects   ({len(val_subs)}): {val_subs}")
print(f"   - Test Subjects  ({len(test_subs)}): {test_subs}")

# สร้าง Mask และแบ่งข้อมูล
train_mask = np.isin(subjects, train_subs)
val_mask = np.isin(subjects, val_subs)
test_mask = np.isin(subjects, test_subs)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# One-Hot Encoding Labels
num_classes = len(np.unique(y))
y_train_hot = to_categorical(y_train, num_classes)
y_val_hot = to_categorical(y_val, num_classes)
y_test_hot = to_categorical(y_test, num_classes)

# ==========================================
# 🧠 4. BUILD CNN-LSTM HYBRID MODEL
# ==========================================
print("\n🏗️ Building CNN-LSTM Architecture...")

model = Sequential()

# --- Part 1: CNN (Feature Extractor) ---
# ทำหน้าที่สแกนหากราฟที่ผิดปกติและลด Noise
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIME_STEPS, len(FEATURES))))
model.add(BatchNormalization())  # ช่วยปรับค่าให้สมดุล เรียนรู้ไวขึ้น
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

# MaxPooling: ย่อข้อมูลลงครึ่งหนึ่ง (จาก 120 เหลือ 60)
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# --- Part 2: LSTM (Sequence Analyzer) ---
# รับข้อมูลที่ถูกย่อแล้ว มาดูความต่อเนื่องของเวลา
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.4))

# --- Part 3: Classifier ---
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ==========================================
# 🔥 5. TRAINING
# ==========================================
callbacks = [
    ModelCheckpoint(OUTPUT_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
]

print("\n🚀 Starting Training...")
history = model.fit(
    X_train, y_train_hot,
    epochs=60,  # ให้เวลาเรียนรู้นานหน่อย
    batch_size=64,
    validation_data=(X_val, y_val_hot),
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 🏆 6. FINAL EVALUATION
# ==========================================
print("\n" + "=" * 50)
print("🧐 Evaluating on UNSEEN Test Set...")
loss, accuracy = model.evaluate(X_test, y_test_hot, verbose=0)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)
