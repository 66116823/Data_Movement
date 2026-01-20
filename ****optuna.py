import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ 1. CONFIGURATION (ตั้งค่าระบบ & กฎการแบ่งกลุ่ม)
# ==========================================
INPUT_FILE = r"E:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_WithFile.csv"
OUTPUT_MODEL_PATH = r"E:\Data Movement\CollectDATA\Master_Data\Best_CNN_LSTM_RiskClass.h5"
OPTUNA_DB_PATH = r"E:\Data Movement\CollectDATA\Master_Data\optuna_study.db"
RESULTS_DIR = r"E:\Data Movement\CollectDATA\Master_Data\Results_RiskClass"

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed Settings
TIME_STEPS = 120
FEATURES = [
    'ACC_Feature_Z',
    'GYRO_Feature_Z',
]

# 🔥 [แก้ไขส่วนที่ 1] กำหนดกฎการแปลง Scenario -> Risk Class
# เช็คชื่อในไฟล์ CSV คอลัมน์ "Scenario" ให้ตรงตัวสะกดเป๊ะๆ (Case Sensitive)
RISK_MAPPING = {
    # === Low Risk (0) ===
    'Normal': 0,
    'Slow': 0,

    # === Medium Risk (1) ===
    'Left': 1,
    'Right': 1,

    # === High Risk (2) ===
    'Dunk': 2,
    'Stun': 2
}

# ชื่อ Class ที่จะโชว์ในกราฟและ Report (เรียงตามเลข 0, 1, 2)
RISK_CLASS_NAMES = ['Low', 'Medium', 'High']

# Optuna Settings
N_TRIALS = 50  # จำนวนครั้งที่จะทดลอง
EPOCHS_PER_TRIAL = 30  # epochs ต่อการทดลอง 1 ครั้ง


# ==========================================
# 🛠️ 2. DATA PIPELINE (เตรียมข้อมูล & จัดกลุ่ม Risk)
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    """โหลดและประมวลผลข้อมูล พร้อมแปลงเป็น 3 ระดับความเสี่ยง"""
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV กรุณาเช็ค Path")

    df = pd.read_csv(file_path)

    # ลบข้อมูลซ้ำ
    print(f"🧹 Cleaning duplicates...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename', 'Time'])
    print(f"   - Reduced from {original_len} to {len(df)} rows")

    # Group ตามไฟล์
    grouped = df.groupby(['Subject_ID', 'Filename'])

    sequences = []
    labels = []
    subject_ids = []

    print(f"⏳ Processing & Mapping Risk Levels (Step={step_size})...")

    for (sub_id, fname), group in grouped:

        # 🔥 [แก้ไขส่วนที่ 2] อ่านค่า Scenario และแปลงเป็น Risk Class
        # หมายเหตุ: ในไฟล์ CSV ต้องมีคอลัมน์ชื่อ 'Scenario' ที่เก็บคำว่า Normal, Slow, Dunk...
        # ถ้าในไฟล์ของคุณคอลัมน์นี้ชื่อ 'Label' หรือชื่ออื่น ให้แก้ตรง ['Scenario'] เป็นชื่อนั้น
        try:
            scenario_name = str(group['Scenario'].iloc[0]).strip()
        except KeyError:
            # Fallback: ถ้าหา Scenario ไม่เจอ ลองหา Label แทน
            try:
                scenario_name = str(group['Label'].iloc[0]).strip()
            except KeyError:
                print("❌ Error: ไม่พบคอลัมน์ 'Scenario' หรือ 'Label' ในไฟล์ CSV")
                return None, None, None

        # ตรวจสอบว่าชื่อท่าอยู่ในกฎ Risk Mapping ของเราไหม
        if scenario_name in RISK_MAPPING:
            risk_label = RISK_MAPPING[scenario_name]
        else:
            # ถ้าเจอชื่อท่าแปลกๆ ที่ไม่อยู่ในกฎ ให้ข้ามไป (ไม่เอามา Train)
            continue

            # --- ส่วนเดิม: ตัดต่อข้อมูล (Downsampling & Padding) ---
        group_ds = group.iloc[::step_size, :]
        series = group_ds[feature_cols].values

        if len(series) == 0: continue

        # Padding / Truncating
        if len(series) >= time_steps:
            seq = series[:time_steps]
        else:
            pad_len = time_steps - len(series)
            seq = np.pad(series, ((0, pad_len), (0, 0)), mode='constant')

        sequences.append(seq)
        labels.append(risk_label)  # เก็บ Label ความเสี่ยงใหม่ (0, 1, 2)
        subject_ids.append(sub_id)

    print(f"✅ Preprocessing Done! Got {len(sequences)} sequences mapped to {RISK_CLASS_NAMES}.")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# ==========================================
# 🏗️ 3. MODEL BUILDER (สร้างโมเดล - ไม่แก้ไขส่วนนี้)
# ==========================================
def create_model(trial, input_shape, num_classes):
    """สร้างโมเดลตาม Hyperparameters ที่ Optuna แนะนำ"""

    # Hyperparameters
    conv_filters_1 = trial.suggest_categorical('conv_filters_1', [32, 64, 128])
    conv_filters_2 = trial.suggest_categorical('conv_filters_2', [32, 64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
    pool_size = trial.suggest_categorical('pool_size', [2, 3])
    lstm_units = trial.suggest_categorical('lstm_units', [50, 100, 150, 200])
    dropout_rate_1 = trial.suggest_float('dropout_rate_1', 0.2, 0.5)
    dropout_rate_2 = trial.suggest_float('dropout_rate_2', 0.3, 0.6)
    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = Sequential()

    # CNN Layers
    model.add(Conv1D(filters=conv_filters_1, kernel_size=kernel_size,
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=conv_filters_2, kernel_size=kernel_size,
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(dropout_rate_1))

    # LSTM Layer
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate_2))

    # Dense Layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


# ==========================================
# 🎯 4. OPTUNA OBJECTIVE (ห้องทดลอง - ไม่แก้ไขส่วนนี้)
# ==========================================
def objective(trial, X_train, y_train, X_val, y_val, num_classes):
    # Suggest STEP_SIZE logic
    step_size = trial.suggest_categorical('step_size', [5, 10, 15, 20])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Reload data with dynamic step_size
    X_temp, y_temp, subjects_temp = load_and_process_data(
        INPUT_FILE, TIME_STEPS, FEATURES, step_size
    )

    # Splitting Logic (Keep consistency)
    unique_subjects = np.unique(subjects_temp)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_test = 5
    n_val = 2

    val_subs = unique_subjects[n_test: n_test + n_val]
    train_subs = unique_subjects[n_test + n_val:]

    train_mask = np.isin(subjects_temp, train_subs)
    val_mask = np.isin(subjects_temp, val_subs)

    X_train_trial = X_temp[train_mask]
    y_train_trial = y_temp[train_mask]
    X_val_trial = X_temp[val_mask]
    y_val_trial = y_temp[val_mask]

    y_train_hot = to_categorical(y_train_trial, num_classes)
    y_val_hot = to_categorical(y_val_trial, num_classes)

    input_shape = (TIME_STEPS, len(FEATURES))
    model = create_model(trial, input_shape, num_classes)

    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_accuracy')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train_trial, y_train_hot,
        epochs=EPOCHS_PER_TRIAL,
        batch_size=batch_size,
        validation_data=(X_val_trial, y_val_hot),
        callbacks=[pruning_callback, early_stopping],
        verbose=0
    )

    return max(history.history['val_accuracy'])


# ==========================================
# 📊 EVALUATION FUNCTIONS (เครื่องมือวัดผล - ไม่แก้ไขส่วนนี้)
# ==========================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    return {
        'overall': {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        },
        'per_class': {
            'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=None, zero_division=0)
        }
    }


def print_detailed_metrics(metrics, class_names):
    print("\n" + "=" * 60)
    print("📊 OVERALL METRICS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Macro Avg':<15} {'Weighted Avg':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {metrics['overall']['accuracy']:.4f}")
    print(
        f"{'Precision':<25} {metrics['overall']['precision_macro']:.4f} {' ' * 7} {metrics['overall']['precision_weighted']:.4f}")
    print(
        f"{'Recall':<25} {metrics['overall']['recall_macro']:.4f} {' ' * 7} {metrics['overall']['recall_weighted']:.4f}")
    print(f"{'F1-Score':<25} {metrics['overall']['f1_macro']:.4f} {' ' * 7} {metrics['overall']['f1_weighted']:.4f}")

    print("\n" + "=" * 60)
    print("📋 PER-CLASS METRICS")
    print("=" * 60)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} "
              f"{metrics['per_class']['precision'][i]:.4f} {' ' * 7} "
              f"{metrics['per_class']['recall'][i]:.4f} {' ' * 7} "
              f"{metrics['per_class']['f1'][i]:.4f}")
    print("=" * 60)


def save_metrics_to_csv(metrics, class_names, save_path):
    pd.DataFrame([metrics['overall']]).to_csv(save_path.replace('.csv', '_overall.csv'), index=False)
    pd.DataFrame({
        'Class': class_names,
        'Precision': metrics['per_class']['precision'],
        'Recall': metrics['per_class']['recall'],
        'F1-Score': metrics['per_class']['f1']
    }).to_csv(save_path.replace('.csv', '_per_class.csv'), index=False)


def evaluate_model(model, X_test, y_test, class_names, save_dir):
    print("\n" + "=" * 60)
    print("🔍 MAKING PREDICTIONS...")
    print("=" * 60)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    metrics = calculate_metrics(y_true, y_pred, class_names)
    print_detailed_metrics(metrics, class_names)

    print("\n" + "=" * 60)
    print("📄 SKLEARN CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(save_dir, 'confusion_matrix.png'))
    save_metrics_to_csv(metrics, class_names, os.path.join(save_dir, 'metrics.csv'))

    return metrics, y_pred, y_true


# ==========================================
# 🚀 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 OPTUNA HYPERPARAMETER TUNING - CNN-LSTM (Risk Class Mode)")
    print("=" * 60)

    # Initial Load
    X_init, y_init, subjects_init = load_and_process_data(
        INPUT_FILE, TIME_STEPS, FEATURES, step_size=10
    )

    # ตรวจสอบว่ามีครบ 3 class หรือไม่ (0, 1, 2)
    num_classes = len(RISK_CLASS_NAMES)
    print(f"\n📊 Dataset Info:")
    print(f"   - Total Samples: {len(X_init)}")
    print(f"   - Number of Classes: {num_classes} {RISK_CLASS_NAMES}")

    if X_init is None or len(X_init) == 0:
        print("❌ Error: No data loaded. Check RISK_MAPPING or CSV file.")
        exit()

    # Optuna Setup
    study = optuna.create_study(
        direction='maximize',
        study_name='CNN_LSTM_RiskClass',
        storage=f'sqlite:///{OPTUNA_DB_PATH}',
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    print(f"\n🎯 Starting Optimization...")
    study.optimize(
        lambda trial: objective(trial, None, None, None, None, num_classes),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION COMPLETED!")
    print("=" * 60)
    print(f"✅ Best Validation Accuracy: {study.best_value:.4f}")

    # Train Final Model
    best_params = study.best_params
    print("\n🔨 Training Final Model with Best Params...")

    X_final, y_final, subjects_final = load_and_process_data(
        INPUT_FILE, TIME_STEPS, FEATURES, best_params['step_size']
    )

    unique_subjects = np.unique(subjects_final)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_test = 5
    n_val = 2

    test_subs = unique_subjects[:n_test]
    val_subs = unique_subjects[n_test: n_test + n_val]
    train_subs = unique_subjects[n_test + n_val:]

    train_mask = np.isin(subjects_final, train_subs)
    val_mask = np.isin(subjects_final, val_subs)
    test_mask = np.isin(subjects_final, test_subs)

    X_train, y_train = X_final[train_mask], y_final[train_mask]
    X_val, y_val = X_final[val_mask], y_final[val_mask]
    X_test, y_test = X_final[test_mask], y_final[test_mask]

    y_train_hot = to_categorical(y_train, num_classes)
    y_val_hot = to_categorical(y_val, num_classes)
    y_test_hot = to_categorical(y_test, num_classes)


    # Mock Trial for Final Build
    class BestTrial:
        def __init__(self, params): self.params = params

        def suggest_categorical(self, name, choices): return self.params[name]

        def suggest_float(self, name, low, high, log=False): return self.params[name]


    best_trial = BestTrial(best_params)
    final_model = create_model(best_trial, (TIME_STEPS, len(FEATURES)), num_classes)

    final_history = final_model.fit(
        X_train, y_train_hot,
        epochs=60,
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val_hot),
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)],
        verbose=1
    )

    final_model.save(OUTPUT_MODEL_PATH)

    # 🔥 [แก้ไขส่วนที่ 3] เรียกใช้ Evaluate ด้วยชื่อ Class ใหม่
    evaluate_model(final_model, X_test, y_test_hot, RISK_CLASS_NAMES, RESULTS_DIR)

    print("\n✨ All Process Completed Successfully!")
