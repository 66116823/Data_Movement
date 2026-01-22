import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ 1. CONFIGURATION (ปรับปรุงใหม่)
# ==========================================
# 📂 ระบุไฟล์ใหม่ของคุณ
INPUT_FILE = r"D:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_1Sec.csv"

BASE_OUTPUT_PATH = r"D:\Data Movement\CollectDATA\Result_Model"
BASE_FOLDER_NAME = "XGBoost_LSTM_1Sec_Results"


# ---------------------------------------------------------
def get_next_output_dir(base_path, folder_name):
    i = 1
    while True:
        new_dir_name = f"{folder_name}_{i:02d}"
        full_path = os.path.join(base_path, new_dir_name)
        if not os.path.exists(full_path):
            return full_path
        i += 1


OUTPUT_DIR = get_next_output_dir(BASE_OUTPUT_PATH, BASE_FOLDER_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ New Output Directory Created: {OUTPUT_DIR}")
# ---------------------------------------------------------

# 🔥 [CRITICAL UPDATE] ค่าคงที่สำหรับข้อมูลรายวินาที
# 1 Row = 1 Second
# เราต้องการดูเหตุการณ์ย้อนหลังประมาณ 6 วินาที เพื่อตัดสินใจ
TIME_STEPS = 6
STEP_SIZE = 1  # ขยับทีละ 1 วินาที (ห้ามกระโดดข้าม)
N_FOLDS = 10

FEATURES = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z'
]

# ลดจำนวน Trials ลงนิดหน่อยเพราะข้อมูลน้อยลงแต่เน้นคุณภาพ
N_TRIALS = 30
EPOCHS_PER_TRIAL = 50


# ==========================================
# 🛠️ 2. DATA LOADING & PROCESSING
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV")

    df = pd.read_csv(file_path)

    # กรองข้อมูลซ้ำ
    print(f"🧹 Cleaning duplicates...")
    original_len = len(df)
    # ตัดคอลัมน์ที่ไม่จำเป็นในการเช็คซ้ำออก (เช่น Time อาจจะต่างกันแต่วินาทีเดียวกัน)
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename', 'Time'] + feature_cols)
    print(f"   - Reduced from {original_len} to {len(df)} rows")

    # 🔥 Normalization
    # เช็คว่าคอลัมน์ไหนยังไม่ได้ทำ Z-score (จากชื่อไฟล์คาดว่าทำมาแล้วบางส่วน แต่กันเหนียว)
    # สมมติว่า Feature_Z คือทำมาแล้ว, แต่ Raw ACC/GYRO อาจจะยัง
    raw_cols = [c for c in feature_cols if 'Feature_Z' not in c]
    if raw_cols:
        print(f"⚖️ Normalizing raw columns: {raw_cols}")
        scaler = StandardScaler()
        df[raw_cols] = scaler.fit_transform(df[raw_cols])

    grouped = df.groupby(['Subject_ID', 'Filename'])
    sequences, labels, subject_ids = [], [], []

    print("✂️ Slicing data into windows...")
    for (sub_id, fname), group in grouped:
        # Sort ตามเวลาเพื่อให้ลำดับถูกต้อง (สำคัญมากสำหรับข้อมูลรายวิ)
        group = group.sort_values('Time')

        # ใช้ข้อมูลทุกบรรทัด (เพราะ Step_size = 1)
        series = group[feature_cols].values
        label_series = group['Label'].values  # เก็บ Label ของแต่ละวิไว้ด้วย

        if len(series) < time_steps:
            # ถ้าไฟล์สั้นกว่า 6 วินาที ให้ Pad (เติม 0 ข้างหน้า)
            pad_len = time_steps - len(series)
            series = np.pad(series, ((pad_len, 0), (0, 0)), mode='constant')
            # ใช้ Label ตัวสุดท้ายของไฟล์
            target_label = label_series[-1]

            sequences.append(series)
            labels.append(target_label)
            subject_ids.append(sub_id)
        else:
            # Sliding Window
            for i in range(0, len(series) - time_steps + 1, step_size):
                seq = series[i:i + time_steps]

                # Label: ใช้ Label ของวินาทีสุดท้ายใน Window (เพราะเราทำนาย ณ ปัจจุบัน)
                # หรือใช้ Mode (ค่าที่พบบ่อยสุด) ในช่วง 6 วิ
                # ในที่นี้ขอใช้ "วินาทีสุดท้าย" เพื่อความ Real-time
                target_label = label_series[i + time_steps - 1]

                sequences.append(seq)
                labels.append(target_label)
                subject_ids.append(sub_id)

    # Handle NaNs in labels if any
    labels = np.nan_to_num(labels, nan=0).astype(int)

    print(f"✅ Got {len(sequences)} sequences using {len(feature_cols)} features.")
    print(f"   (Window Size: {time_steps} sec, Step: {step_size} sec)")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# ==========================================
# 🔧 3. FEATURE ENGINEERING FOR XGBOOST
# ==========================================
def extract_statistical_features(sequences):
    """
    แปลงข้อมูล Time Series เป็นตารางสถิติ
    สำหรับข้อมูล 6 วินาที: ค่า Diff และ Slope จะสำคัญมาก
    """
    # print("🔧 Extracting statistical features...") # ปิด print เพื่อลดรก
    n_samples = sequences.shape[0]
    n_features = sequences.shape[2]
    feature_list = []

    for i in range(n_samples):
        features = []
        for f in range(n_features):
            series = sequences[i, :, f]

            # Basic Stats
            mean_val = np.mean(series)
            std_val = np.std(series)
            min_val = np.min(series)
            max_val = np.max(series)
            range_val = max_val - min_val

            # Dynamic Stats (สำคัญสำหรับการเปลี่ยนแปลงรายวิ)
            diff = np.diff(series)
            mean_change = np.mean(diff) if len(diff) > 0 else 0
            max_change = np.max(np.abs(diff)) if len(diff) > 0 else 0

            features.extend([mean_val, std_val, min_val, max_val, range_val, mean_change, max_change])

        feature_list.append(features)

    return np.array(feature_list)


def select_important_features_xgboost(X_stat, y, top_k=20):
    """ใช้ XGBoost หาฟีเจอร์สำคัญ"""
    print(f"\n🎯 XGBoost Feature Selection (Top-{top_k})...")

    # ปรับ XGBoost ให้เหมาะกับข้อมูลน้อยลง (ลด Depth กัน Overfit)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0,
        tree_method='hist'  # เร็วขึ้น
    )
    model.fit(X_stat, y)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:][::-1]
    print(f"✅ Selected top {top_k} features.")
    return top_indices, model


def reconstruct_sequences_from_features(sequences, selected_indices, n_original_features):
    """Map index กลับไปที่ Channel เดิม"""
    # เรา extract 7 features ต่อ 1 channel (mean, std, min, max, range, mean_change, max_change)
    features_per_channel = 7
    selected_channels = set()
    for idx in selected_indices:
        channel = idx // features_per_channel
        if channel < n_original_features:
            selected_channels.add(channel)

    selected_channels = sorted(list(selected_channels))

    # กันเหนียว: ถ้าไม่เลือกเลย ให้เอาทั้งหมด
    if not selected_channels:
        selected_channels = list(range(n_original_features))

    # print(f"   -> XGBoost Selected Channels: {selected_channels}")
    return sequences[:, :, selected_channels]


# ==========================================
# 🧠 4. LSTM MODEL BUILDER
# ==========================================
def create_xgboost_lstm_model(trial, input_shape, num_classes):
    model = Sequential()

    # ปรับ LSTM Units ให้เล็กลงหน่อยเพราะ Time step น้อยลง (6 steps)
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    model.add(LSTM(lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    dense_units = trial.suggest_categorical('dense_units', [32, 64])
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# ==========================================
# 🎯 5. OPTUNA OBJECTIVE
# ==========================================
def objective(trial, X_seq_train, y_train, X_seq_val, y_val, num_classes):
    # Parameter สำหรับ XGBoost Selection
    top_k = trial.suggest_int('top_k_features', 10, 40, step=5)  # ลดจำนวน feature ลงเพราะ input น้อย

    X_stat_train = extract_statistical_features(X_seq_train)
    y_train_lbl = np.argmax(y_train, axis=1)

    selected_indices, _ = select_important_features_xgboost(X_stat_train, y_train_lbl, top_k=top_k)

    n_features_orig = len(FEATURES)
    X_train_reduced = reconstruct_sequences_from_features(X_seq_train, selected_indices, n_features_orig)
    X_val_reduced = reconstruct_sequences_from_features(X_seq_val, selected_indices, n_features_orig)

    input_shape = (X_train_reduced.shape[1], X_train_reduced.shape[2])
    model = create_xgboost_lstm_model(trial, input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_reduced, y_train,
        validation_data=(X_val_reduced, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=trial.suggest_categorical('batch_size', [16, 32]),  # Batch เล็กลง
        callbacks=[early_stopping, optuna.integration.TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )
    return max(history.history['val_accuracy'])


# ==========================================
# 🧠 6. TRAIN FINAL MODEL
# ==========================================
def train_final_model(model, X_train, y_train, X_val, y_val, params):
    print("\n🔨 Training Final Model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    y_integers = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights_dict = dict(enumerate(weights))
    print(f"⚖️ Class Weights: {class_weights_dict}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    return history


# ==========================================
# 🚀 7. MAIN EXECUTION WITH K-FOLD CV (UPDATED: Tune Every Fold)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 XGBoost-LSTM (1-Sec Resolution) with Full K-Fold Optimization")
    print("=" * 60)

    # 1. Load Data
    # หมายเหตุ: ใช้ค่า TIME_STEPS = 6, STEP_SIZE = 1 ตามที่คุยกัน
    X, y, subjects = load_and_process_data(INPUT_FILE, TIME_STEPS, FEATURES, step_size=STEP_SIZE)

    num_classes = len(np.unique(y))
    print(f"\n📊 Classes found: {np.unique(y)} (Total: {num_classes})")

    unique_subjects = np.unique(subjects)
    # np.random.seed(42) # เปิดบรรทัดนี้ถ้าอยากให้ผลการแบ่ง Fold เหมือนเดิมทุกครั้ง
    np.random.shuffle(unique_subjects)

    print(f"\n👥 Total Subjects Found: {len(unique_subjects)}")

    kfold = KFold(n_splits=N_FOLDS, shuffle=False)
    fold_results = []
    fold_predictions = []
    fold_true_labels = []

    # Loop Folds
    for fold_idx, (train_val_indices, test_indices) in enumerate(kfold.split(unique_subjects)):
        print("\n" + "=" * 80)
        print(f"📌 FOLD {fold_idx + 1}/{N_FOLDS}")
        print("=" * 80)

        train_val_subs = unique_subjects[train_val_indices]
        test_subs = unique_subjects[test_indices]

        # Validation Split
        n_val = max(1, int(len(train_val_subs) * 0.15))
        val_subs = train_val_subs[:n_val]
        train_subs = train_val_subs[n_val:]

        train_mask = np.isin(subjects, train_subs)
        val_mask = np.isin(subjects, val_subs)
        test_mask = np.isin(subjects, test_subs)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        y_train_hot = to_categorical(y_train, num_classes)
        y_val_hot = to_categorical(y_val, num_classes)
        y_test_hot = to_categorical(y_test, num_classes)

        # ----------------------------------------------------------------
        # 🔥 UPDATED: จูน Optuna ใหม่ "ทุก Fold" (ไม่ต้องมี if fold_idx == 0 แล้ว)
        # ----------------------------------------------------------------
        print(f"\n🔍 [Fold {fold_idx + 1}] Starting Optuna Optimization ({N_TRIALS} trials)...")

        # สร้าง Study ใหม่ทุกรอบ เพื่อไม่ให้จำค่าจาก Fold เก่า
        study = optuna.create_study(direction='maximize', pruner=MedianPruner())
        study.optimize(lambda trial: objective(trial, X_train, y_train_hot, X_val, y_val_hot, num_classes),
                       n_trials=N_TRIALS)

        best_params = study.best_params
        print(f"🏆 [Fold {fold_idx + 1}] Best Params Found: {best_params}")
        print(f"   Best Val Accuracy: {study.best_value:.4f}")

        # ----------------------------------------------------------------
        # Feature Selection & Training using THIS FOLD's best params
        # ----------------------------------------------------------------
        X_stat_train = extract_statistical_features(X_train)
        selected_indices, _ = select_important_features_xgboost(
            X_stat_train, np.argmax(y_train_hot, axis=1), top_k=best_params['top_k_features']
        )

        X_train_final = reconstruct_sequences_from_features(X_train, selected_indices, len(FEATURES))
        X_val_final = reconstruct_sequences_from_features(X_val, selected_indices, len(FEATURES))
        X_test_final = reconstruct_sequences_from_features(X_test, selected_indices, len(FEATURES))


        # Helper class for fixed params
        class FixedTrial:
            def suggest_categorical(self, n, c): return best_params[n]

            def suggest_float(self, n, l, h, log=False): return best_params[n]


        fold_model = create_xgboost_lstm_model(FixedTrial(), (X_train_final.shape[1], X_train_final.shape[2]),
                                               num_classes)
        train_final_model(fold_model, X_train_final, y_train_hot, X_val_final, y_val_hot, best_params)

        # Evaluate
        print(f"\n📊 Evaluating Fold {fold_idx + 1}...")
        y_pred = np.argmax(fold_model.predict(X_test_final), axis=1)
        y_true = np.argmax(y_test_hot, axis=1)

        fold_predictions.extend(y_pred)
        fold_true_labels.extend(y_true)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        fold_results.append(report)
        print(f"✅ Fold {fold_idx + 1} Test Acc: {report['accuracy']:.4f}")

        # Save Fold Model
        fold_model.save(os.path.join(OUTPUT_DIR, f"Fold_{fold_idx + 1}_Model.h5"))

        # Save Confusion Matrix per Fold
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Fold {fold_idx + 1}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_fold_{fold_idx + 1}.png"))
        plt.close()

    # ==========================================
    # Summary of All Folds
    # ==========================================
    overall_acc = np.mean([f['accuracy'] for f in fold_results])
    print("\n" + "=" * 80)
    print(f"🎯 OVERALL ACCURACY ({N_FOLDS}-Fold CV): {overall_acc:.4f}")
    print("=" * 80)

    # Save Results
    results_df = pd.DataFrame(classification_report(fold_true_labels, fold_predictions, output_dict=True)).transpose()
    results_df.to_csv(os.path.join(OUTPUT_DIR, "overall_results.csv"))
    print("\n--- Overall Report ---")
    print(results_df)

    cm = confusion_matrix(fold_true_labels, fold_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Overall Confusion Matrix ({N_FOLDS}-Fold CV)")
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_confusion_matrix.png"))
    print(f"\n✅ All Done! Results saved in: {OUTPUT_DIR}")