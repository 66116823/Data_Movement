import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
INPUT_FILE = r"E:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_WithFile.csv"
OUTPUT_DIR = r"E:\Data Movement\CollectDATA\Master_Data\XGBoost_LSTM_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ค่าคงที่
TIME_STEPS = 120  # จำนวนข้อมูลที่มองย้อนหลัง (Sequence Length)

# ใส่ทั้งข้อมูลดิบ (เพื่อจับ Dunk) และข้อมูล Normalized
FEATURES = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z',
    'ACC_Feature_Z', 'GYRO_Feature_Z'
]

# การตั้งค่า Optuna
N_TRIALS = 30
EPOCHS_PER_TRIAL = 25


# ==========================================
# 🛠️ 2. DATA LOADING & PROCESSING
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    """โหลดและประมวลผลข้อมูล"""
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV")

    df = pd.read_csv(file_path)

    print(f"🧹 Cleaning duplicates...")
    original_len = len(df)

    # 🔥 [สิ่งที่แก้ไข 2] เปลี่ยนวิธีการลบข้อมูลซ้ำให้ปลอดภัยขึ้น
    # ไม่สนใจคอลัมน์ Time ในการเช็คซ้ำ (เผื่อ Excel ยังมีปัญหา)
    # แต่เช็คว่าถ้า "ค่าเซนเซอร์ทั้งหมด" เหมือนกันเป๊ะๆ ค่อยลบ
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename'] + feature_cols)

    print(f"   - Reduced from {original_len} to {len(df)} rows")
    print(f"   - Note: Keeping rows even if timestamps look duplicate, provided sensor data is different.")

    grouped = df.groupby(['Subject_ID', 'Filename'])
    sequences, labels, subject_ids = [], [], []

    for (sub_id, fname), group in grouped:
        # Downsampling (ถ้าต้องการลดความถี่ข้อมูล ให้ปรับ step_size > 1)
        group_ds = group.iloc[::step_size, :]

        series = group_ds[feature_cols].values

        if len(series) == 0:
            continue

        # Padding / Truncating ให้ความยาวเท่ากับ TIME_STEPS
        if len(series) >= time_steps:
            seq = series[:time_steps]
        else:
            pad_len = time_steps - len(series)
            seq = np.pad(series, ((0, pad_len), (0, 0)), mode='constant')

        sequences.append(seq)

        # ใช้ Label ตามที่มีในไฟล์ (0, 1, 2)
        try:
            labels.append(group['Label'].iloc[0])
        except KeyError:
            # กรณีหา Label ไม่เจอ ลองหา Scenario แล้ว Map เอา (Optional)
            labels.append(0)

        subject_ids.append(sub_id)

    print(f"✅ Got {len(sequences)} sequences using 6-axis features.")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# ==========================================
# 🔧 3. FEATURE ENGINEERING FOR XGBOOST
# ==========================================
def extract_statistical_features(sequences):
    """แปลงข้อมูล Time Series (3D) เป็นตารางสถิติ (2D) เพื่อให้ XGBoost อ่านรู้เรื่อง"""
    print("🔧 Extracting statistical features...")
    n_samples = sequences.shape[0]
    n_features = sequences.shape[2]

    feature_list = []

    for i in range(n_samples):
        features = []
        for f in range(n_features):  # วนลูปทีละแกน (ACC_X, ..., GYRO_Z)
            series = sequences[i, :, f]

            # ค่าสถิติต่างๆ ที่สะท้อนลักษณะการเดิน
            features.extend([
                np.mean(series),  # ค่าเฉลี่ย
                np.std(series),  # ส่วนเบี่ยงเบนมาตรฐาน (สำคัญสำหรับ Dunk)
                np.min(series),  # ค่าต่ำสุด
                np.max(series),  # ค่าสูงสุด
                np.max(series) - np.min(series),  # Range (ความกว้างของค่า)
                np.var(series),  # ความแปรปรวน
            ])

            # การเปลี่ยนแปลง (Jerk/Change)
            diff = np.diff(series)
            features.extend([
                np.mean(np.abs(diff)),  # การเปลี่ยนแปลงเฉลี่ย
                np.max(np.abs(diff)),  # การกระชากสูงสุด
            ])

        feature_list.append(features)

    X_stat = np.array(feature_list)
    return X_stat


def select_important_features_xgboost(X_stat, y, top_k=50):
    """ใช้ XGBoost หาว่าฟีเจอร์ไหนสำคัญที่สุด"""
    print(f"\n🎯 XGBoost Feature Selection (Top-{top_k})...")

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    model.fit(X_stat, y)

    importances = model.feature_importances_
    # เลือก Index ของฟีเจอร์ที่มีคะแนนความสำคัญสูงสุด
    top_indices = np.argsort(importances)[-top_k:][::-1]

    print(f"✅ Selected top {top_k} features.")
    return top_indices, model


def reconstruct_sequences_from_features(sequences, selected_indices, n_original_features):
    """เลือกเฉพาะแกนข้อมูล (Channel) ที่ XGBoost บอกว่าสำคัญ มาสร้างเป็น Sequence ใหม่"""
    # คำนวณจำนวน Stats ต่อ 1 Channel (ตามฟังก์ชัน extract_statistical_features)
    # เราใส่ไป 6 stats + 2 diff stats = 8 features ต่อแกน
    features_per_channel = 8

    selected_channels = set()
    for idx in selected_indices:
        channel = idx // features_per_channel
        if channel < n_original_features:
            selected_channels.add(channel)

    selected_channels = sorted(list(selected_channels))

    # กันเหนียว: ถ้าไม่เหลือ Channel เลย ให้ใช้ทั้งหมด
    if not selected_channels:
        selected_channels = range(n_original_features)

    print(f"   -> Retained Channels Indices: {selected_channels}")

    # ตัดข้อมูลเอาเฉพาะแกนที่สำคัญ
    sequences_reduced = sequences[:, :, selected_channels]
    return sequences_reduced


# ==========================================
# 🧠 4. LSTM MODEL BUILDER
# ==========================================
def create_xgboost_lstm_model(trial, input_shape, num_classes):
    """สร้างโครงสร้าง LSTM โดยให้ Optuna ช่วยกำหนดค่า"""

    model = Sequential()

    # LSTM Layers
    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [64, 128, 200])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    model.add(LSTM(lstm_units_1, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Dense Layers
    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output Layer

    optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# ==========================================
# 🎯 5. OPTUNA OBJECTIVE
# ==========================================
def objective(trial, X_seq_train, y_train, X_seq_val, y_val, num_classes):
    # 1. XGBoost Selection Phase
    top_k = trial.suggest_int('top_k_features', 20, 48, step=4)  # สูงสุด 6 แกน * 8 stats = 48

    X_stat_train = extract_statistical_features(X_seq_train)

    # แปลง One-hot กลับเป็น Class ปกติชั่วคราวเพื่อเทรน XGBoost
    y_train_lbl = np.argmax(y_train, axis=1)

    selected_indices, _ = select_important_features_xgboost(X_stat_train, y_train_lbl, top_k=top_k)

    # 2. Reconstruct Data for LSTM
    n_features_orig = len(FEATURES)
    X_train_reduced = reconstruct_sequences_from_features(X_seq_train, selected_indices, n_features_orig)
    X_val_reduced = reconstruct_sequences_from_features(X_seq_val, selected_indices, n_features_orig)

    # 3. Train LSTM
    input_shape = (X_train_reduced.shape[1], X_train_reduced.shape[2])
    model = create_xgboost_lstm_model(trial, input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(
        X_train_reduced, y_train,
        validation_data=(X_val_reduced, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=trial.suggest_categorical('batch_size', [32, 64]),
        callbacks=[early_stopping, optuna.integration.TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )

    return max(history.history['val_accuracy'])


# ==========================================
# 🚀 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 XGBoost-LSTM Hybrid Model (Real-Time Optimized)")
    print("=" * 60)

    # 1. Load Data
    # step_size=1 คือเอาทุก millisecond ที่มี (ละเอียดสุด) ถ้าข้อมูลเยอะเกินค่อยปรับเป็น 2-5
    X, y, subjects = load_and_process_data(INPUT_FILE, TIME_STEPS, FEATURES, step_size=1)

    num_classes = len(np.unique(y))
    print(f"\n📊 Classes found: {np.unique(y)} (Total: {num_classes})")

    # 2. Split Data (Subject-Independent)
    unique_subjects = np.unique(subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_test = 5
    n_val = 2

    test_subs = unique_subjects[:n_test]
    val_subs = unique_subjects[n_test:n_test + n_val]
    train_subs = unique_subjects[n_test + n_val:]

    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    y_train_hot = to_categorical(y_train, num_classes)
    y_val_hot = to_categorical(y_val, num_classes)
    y_test_hot = to_categorical(y_test, num_classes)

    # 3. Optuna Tuning
    print(f"\n🔍 Starting Optimization ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, y_train_hot, X_val, y_val_hot, num_classes),
                   n_trials=N_TRIALS)

    print(f"\n🏆 Best Accuracy: {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")

    # 4. Train Final Model
    print("\n🔨 Training Final Model...")
    best_params = study.best_params

    # ทำซ้ำกระบวนการคัดเลือก Feature ด้วย Best Params
    X_stat_train = extract_statistical_features(X_train)
    selected_indices, xgb_model = select_important_features_xgboost(
        X_stat_train, np.argmax(y_train_hot, axis=1), top_k=best_params['top_k_features']
    )

    X_train_final = reconstruct_sequences_from_features(X_train, selected_indices, len(FEATURES))
    X_val_final = reconstruct_sequences_from_features(X_val, selected_indices, len(FEATURES))
    X_test_final = reconstruct_sequences_from_features(X_test, selected_indices, len(FEATURES))


    # สร้างโมเดลสุดท้าย
    class FixedTrial:
        def suggest_categorical(self, n, c): return best_params[n]

        def suggest_float(self, n, l, h, log=False): return best_params[n]


    final_model = create_xgboost_lstm_model(FixedTrial(), (X_train_final.shape[1], X_train_final.shape[2]), num_classes)

    final_model.fit(
        X_train_final, y_train_hot,
        validation_data=(X_val_final, y_val_hot),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    final_model.save(os.path.join(OUTPUT_DIR, "Final_XGB_LSTM_Model.h5"))

    # 5. Evaluate
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    y_pred = np.argmax(final_model.predict(X_test_final), axis=1)
    y_true = np.argmax(y_test_hot, axis=1)

    print(classification_report(y_true, y_pred))

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print("✅ Done!")