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
# ⚙️ 1. CONFIGURATION
# ==========================================
INPUT_FILE = r"D:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_WithFiles.csv"

# 📂 กำหนดโฟลเดอร์หลัก
BASE_OUTPUT_PATH = r"D:\Data Movement\CollectDATA\Result_Model"
BASE_FOLDER_NAME = "XGBoost_LSTM_KFold_Results"


# ---------------------------------------------------------
# ฟังก์ชันสร้างโฟลเดอร์รันถัดไปอัตโนมัติ (Auto-Increment)
# ---------------------------------------------------------
def get_next_output_dir(base_path, folder_name):
    """
    เช็คว่าโฟลเดอร์ Result_01 มีหรือยัง?
    ถ้ามีแล้ว -> สร้าง Result_02
    ถ้ามี 02 แล้ว -> สร้าง Result_03
    ไปเรื่อยๆ...
    """
    i = 1
    while True:
        new_dir_name = f"{folder_name}_{i:02d}"
        full_path = os.path.join(base_path, new_dir_name)
        if not os.path.exists(full_path):
            return full_path
        i += 1


# 🔥 เรียกใช้ฟังก์ชัน เพื่อกำหนด OUTPUT_DIR ใหม่
OUTPUT_DIR = get_next_output_dir(BASE_OUTPUT_PATH, BASE_FOLDER_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ New Output Directory Created: {OUTPUT_DIR}")
# ---------------------------------------------------------

# ค่าคงที่
TIME_STEPS = 40
STEP_SIZE = 20
N_FOLDS = 10  # จำนวน Folds สำหรับ Cross Validation

FEATURES = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z',
    'ACC_Feature_Z', 'GYRO_Feature_Z'
]

# การตั้งค่า Optuna
N_TRIALS = 30
EPOCHS_PER_TRIAL = 50


# ==========================================
# 🛠️ 2. DATA LOADING & PROCESSING
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    """โหลดและประมวลผลข้อมูล (พร้อมทำ Normalization ข้อมูลดิบ)"""
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV")

    df = pd.read_csv(file_path)

    print(f"🧹 Cleaning duplicates...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename'] + feature_cols)
    print(f"   - Reduced from {original_len} to {len(df)} rows")

    # 🔥 Normalization ข้อมูลดิบ
    raw_cols = [c for c in feature_cols if 'Feature_Z' not in c]
    if raw_cols:
        print(f"⚖️ Normalizing raw columns: {raw_cols}")
        scaler = StandardScaler()
        df[raw_cols] = scaler.fit_transform(df[raw_cols])

    grouped = df.groupby(['Subject_ID', 'Filename'])
    sequences, labels, subject_ids = [], [], []

    for (sub_id, fname), group in grouped:
        group_ds = group.iloc[::step_size, :]
        series = group_ds[feature_cols].values

        if len(series) == 0:
            continue

        if len(series) >= time_steps:
            seq = series[:time_steps]
        else:
            pad_len = time_steps - len(series)
            seq = np.pad(series, ((0, pad_len), (0, 0)), mode='constant')

        sequences.append(seq)
        try:
            labels.append(group['Label'].iloc[0])
        except KeyError:
            labels.append(0)
        subject_ids.append(sub_id)

    print(f"✅ Got {len(sequences)} sequences using {len(feature_cols)} features (Normalized).")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# ==========================================
# 🔧 3. FEATURE ENGINEERING FOR XGBOOST
# ==========================================
def extract_statistical_features(sequences):
    """แปลงข้อมูล Time Series เป็นตารางสถิติ"""
    print("🔧 Extracting statistical features...")
    n_samples = sequences.shape[0]
    n_features = sequences.shape[2]
    feature_list = []

    for i in range(n_samples):
        features = []
        for f in range(n_features):
            series = sequences[i, :, f]
            features.extend([
                np.mean(series), np.std(series), np.min(series), np.max(series),
                np.max(series) - np.min(series), np.var(series)
            ])
            diff = np.diff(series)
            features.extend([np.mean(np.abs(diff)), np.max(np.abs(diff))])
        feature_list.append(features)

    return np.array(feature_list)


def select_important_features_xgboost(X_stat, y, top_k=50):
    """ใช้ XGBoost หาฟีเจอร์สำคัญ"""
    print(f"\n🎯 XGBoost Feature Selection (Top-{top_k})...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    model.fit(X_stat, y)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:][::-1]
    print(f"✅ Selected top {top_k} features.")
    return top_indices, model


def reconstruct_sequences_from_features(sequences, selected_indices, n_original_features):
    """เลือกเฉพาะแกนที่สำคัญ"""
    features_per_channel = 8
    selected_channels = set()
    for idx in selected_indices:
        channel = idx // features_per_channel
        if channel < n_original_features:
            selected_channels.add(channel)

    selected_channels = sorted(list(selected_channels))
    if not selected_channels:
        selected_channels = range(n_original_features)

    print(f"   -> Retained Channels Indices: {selected_channels}")
    return sequences[:, :, selected_channels]


# ==========================================
# 🧠 4. LSTM MODEL BUILDER
# ==========================================
def create_xgboost_lstm_model(trial, input_shape, num_classes):
    model = Sequential()
    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [64, 128, 200])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    model.add(LSTM(lstm_units_1, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# ==========================================
# 🎯 5. OPTUNA OBJECTIVE
# ==========================================
def objective(trial, X_seq_train, y_train, X_seq_val, y_val, num_classes):
    print("\n" + "═" * 50)
    print(f"🚀 RUNNING TRIAL: {trial.number + 1} / {N_TRIALS}")
    print("═" * 50)

    top_k = trial.suggest_int('top_k_features', 20, 48, step=4)
    X_stat_train = extract_statistical_features(X_seq_train)
    y_train_lbl = np.argmax(y_train, axis=1)

    selected_indices, _ = select_important_features_xgboost(X_stat_train, y_train_lbl, top_k=top_k)

    n_features_orig = len(FEATURES)
    X_train_reduced = reconstruct_sequences_from_features(X_seq_train, selected_indices, n_features_orig)
    X_val_reduced = reconstruct_sequences_from_features(X_seq_val, selected_indices, n_features_orig)

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
# 🧠 6. TRAIN FINAL MODEL
# ==========================================
def train_final_model(model, X_train, y_train, X_val, y_val, params):
    """เทรนโมเดลรอบสุดท้ายแบบมี Class Weights"""
    print("\n🔨 Training Final Model...")

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    # 🔥 คำนวณน้ำหนัก Class
    y_integers = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
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
# 🚀 7. MAIN EXECUTION WITH K-FOLD CV
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 XGBoost-LSTM Hybrid Model with K-Fold CV")
    print("=" * 60)

    # 1. Load Data
    X, y, subjects = load_and_process_data(INPUT_FILE, TIME_STEPS, FEATURES, step_size=STEP_SIZE)

    num_classes = len(np.unique(y))
    print(f"\n📊 Classes found: {np.unique(y)} (Total: {num_classes})")

    # ==========================================
    # 2. K-FOLD CROSS VALIDATION SETUP
    # ==========================================
    unique_subjects = np.unique(subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    total_subjects = len(unique_subjects)
    print(f"\n👥 Total Subjects Found: {total_subjects}")
    print(f"🔄 Using {N_FOLDS}-Fold Cross Validation\n")

    # เก็บผลลัพธ์แต่ละ Fold
    fold_results = []
    fold_predictions = []
    fold_true_labels = []

    # สร้าง KFold object
    kfold = KFold(n_splits=N_FOLDS, shuffle=False)

    # ==========================================
    # 3. LOOP THROUGH EACH FOLD
    # ==========================================
    for fold_idx, (train_val_indices, test_indices) in enumerate(kfold.split(unique_subjects)):
        print("\n" + "=" * 80)
        print(f"📌 FOLD {fold_idx + 1}/{N_FOLDS}")
        print("=" * 80)

        # แบ่ง subjects ตาม fold
        train_val_subs = unique_subjects[train_val_indices]
        test_subs = unique_subjects[test_indices]

        # แบ่ง train_val เป็น train:val = 90:10
        n_val = int(len(train_val_subs) * 0.10)
        if n_val == 0:
            n_val = 1

        val_subs = train_val_subs[:n_val]
        train_subs = train_val_subs[n_val:]

        print(f"   🔹 Train Subjects: {len(train_subs)} persons")
        print(f"   🔹 Val Subjects:   {len(val_subs)} persons")
        print(f"   🔹 Test Subjects:  {len(test_subs)} persons")

        # Map กลับไปที่ข้อมูลจริง
        train_mask = np.isin(subjects, train_subs)
        val_mask = np.isin(subjects, val_subs)
        test_mask = np.isin(subjects, test_subs)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        y_train_hot = to_categorical(y_train, num_classes)
        y_val_hot = to_categorical(y_val, num_classes)
        y_test_hot = to_categorical(y_test, num_classes)

        # ==========================================
        # 4. OPTUNA OPTIMIZATION (ทำเฉพาะ Fold แรก)
        # ==========================================
        if fold_idx == 0:
            print(f"\n🔍 Starting Optimization ({N_TRIALS} trials) - Fold 1 only...")
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(lambda trial: objective(trial, X_train, y_train_hot, X_val, y_val_hot, num_classes),
                           n_trials=N_TRIALS)

            print(f"\n🏆 Best Accuracy: {study.best_value:.4f}")
            print(f"   Best Params: {study.best_params}")
            best_params = study.best_params
        else:
            print(f"\n♻️ Using best params from Fold 1")

        # ==========================================
        # 5. FEATURE SELECTION & MODEL TRAINING
        # ==========================================
        X_stat_train = extract_statistical_features(X_train)
        selected_indices, xgb_model = select_important_features_xgboost(
            X_stat_train, np.argmax(y_train_hot, axis=1), top_k=best_params['top_k_features']
        )

        X_train_final = reconstruct_sequences_from_features(X_train, selected_indices, len(FEATURES))
        X_val_final = reconstruct_sequences_from_features(X_val, selected_indices, len(FEATURES))
        X_test_final = reconstruct_sequences_from_features(X_test, selected_indices, len(FEATURES))


        # สร้างโมเดล
        class FixedTrial:
            def suggest_categorical(self, n, c): return best_params[n]

            def suggest_float(self, n, l, h, log=False): return best_params[n]


        fold_model = create_xgboost_lstm_model(
            FixedTrial(),
            (X_train_final.shape[1], X_train_final.shape[2]),
            num_classes
        )

        # เทรนโมเดล
        train_final_model(fold_model, X_train_final, y_train_hot, X_val_final, y_val_hot, best_params)

        # ==========================================
        # 6. EVALUATE FOLD
        # ==========================================
        print(f"\n📊 Evaluating Fold {fold_idx + 1}...")
        y_pred = np.argmax(fold_model.predict(X_test_final), axis=1)
        y_true = np.argmax(y_test_hot, axis=1)

        # เก็บผลลัพธ์
        fold_predictions.extend(y_pred)
        fold_true_labels.extend(y_true)

        # สร้าง classification report
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        fold_results.append(report_dict)

        # แสดงผลแต่ละ Fold
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.index.name = "Risk_Class"
        print(f"\n--- Fold {fold_idx + 1} Results ---")
        print(report_df.to_string(float_format="{:.2f}".format))

        # บันทึกโมเดลแต่ละ Fold
        fold_model.save(os.path.join(OUTPUT_DIR, f"Fold_{fold_idx + 1}_Model.h5"))

        # สร้าง Confusion Matrix สำหรับแต่ละ Fold
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Fold {fold_idx + 1}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_fold_{fold_idx + 1}.png"))
        plt.close()

    # ==========================================
    # 7. AGGREGATE RESULTS FROM ALL FOLDS
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 OVERALL K-FOLD CROSS VALIDATION RESULTS")
    print("=" * 80)

    # คำนวณค่าเฉลี่ยของแต่ละ metric จากทุก fold
    avg_metrics = {}
    metrics_to_average = ['precision', 'recall', 'f1-score']

    # หา classes ทั้งหมดที่มีในทุก fold
    all_classes = set()
    for report in fold_results:
        all_classes.update([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])

    # คำนวณค่าเฉลี่ย
    for class_name in sorted(all_classes):
        avg_metrics[class_name] = {}
        for metric in metrics_to_average + ['support']:
            values = [fold[class_name][metric] for fold in fold_results if class_name in fold]
            avg_metrics[class_name][metric] = np.mean(values) if values else 0

    # คำนวณค่าเฉลี่ยสำหรับ macro avg และ weighted avg
    for avg_type in ['macro avg', 'weighted avg']:
        avg_metrics[avg_type] = {}
        for metric in metrics_to_average:
            values = [fold[avg_type][metric] for fold in fold_results if avg_type in fold]
            avg_metrics[avg_type][metric] = np.mean(values) if values else 0

    # คำนวณ overall accuracy
    overall_accuracy = np.mean([fold['accuracy'] for fold in fold_results])
    avg_metrics['accuracy'] = {'precision': overall_accuracy, 'recall': overall_accuracy,
                               'f1-score': overall_accuracy, 'support': len(fold_true_labels)}

    # แสดงผลค่าเฉลี่ย
    avg_report_df = pd.DataFrame(avg_metrics).transpose()
    avg_report_df.index.name = "Risk_Class"
    print("\n--- Average Results Across All Folds ---")
    print(avg_report_df.to_string(float_format="{:.2f}".format))
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f}")

    # สร้าง Overall Confusion Matrix
    overall_cm = confusion_matrix(fold_true_labels, fold_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Overall Confusion Matrix ({N_FOLDS}-Fold CV)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_confusion_matrix.png"))
    plt.close()

    # บันทึก detailed results
    results_summary = {
        'best_params': best_params,
        'overall_accuracy': overall_accuracy,
        'fold_accuracies': [fold['accuracy'] for fold in fold_results],
        'average_metrics': avg_metrics
    }

    # บันทึกเป็น CSV
    avg_report_df.to_csv(os.path.join(OUTPUT_DIR, "kfold_average_results.csv"))

    print(f"\n✅ All Done! Results saved in: {OUTPUT_DIR}")
    print(f"📁 Saved {N_FOLDS} fold models and 1 overall confusion matrix")
