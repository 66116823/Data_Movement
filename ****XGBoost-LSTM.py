import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = r"E:\Data Movement\CollectDATA\Master_Data\Feature_Z-score_NormBase_WithFile.csv"
OUTPUT_DIR = r"E:\Data Movement\CollectDATA\Master_Data\XGBoost_LSTM_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_STEPS = 120
FEATURES = ['ACC_Feature_Z', 'GYRO_Feature_Z']
N_TRIALS = 30
EPOCHS_PER_TRIAL = 25


# ==========================================
# 🛠️ DATA LOADING (ใช้ฟังก์ชันเดิม)
# ==========================================
def load_and_process_data(file_path, time_steps, feature_cols, step_size):
    """โหลดและประมวลผลข้อมูล"""
    print(f"📂 Loading Data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ ไม่พบไฟล์ CSV")

    df = pd.read_csv(file_path)
    print(f"🧹 Cleaning duplicates...")
    df = df.drop_duplicates(subset=['Subject_ID', 'Filename', 'Time'])

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
        labels.append(group['Label'].iloc[0])
        subject_ids.append(sub_id)

    print(f"✅ Got {len(sequences)} sequences")
    return np.array(sequences), np.array(labels), np.array(subject_ids)


# ==========================================
# 🔧 FEATURE ENGINEERING FOR XGBOOST
# ==========================================
def extract_statistical_features(sequences):
    """
    แปลง 3D sequences → 2D statistical features
    Input: (n_samples, time_steps, n_features)
    Output: (n_samples, n_stat_features)
    """
    print("🔧 Extracting statistical features...")
    n_samples = sequences.shape[0]
    n_features = sequences.shape[2]

    feature_list = []

    for i in range(n_samples):
        features = []
        for f in range(n_features):
            series = sequences[i, :, f]

            # Statistical Features
            features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
                np.percentile(series, 25),
                np.percentile(series, 75),
                np.var(series),
            ])

            # Difference Features
            diff = np.diff(series)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.max(np.abs(diff)),
            ])

            # Energy & Zero Crossings
            features.append(np.sum(series ** 2))
            features.append(np.sum(np.diff(np.sign(series)) != 0))

        feature_list.append(features)

    X_stat = np.array(feature_list)
    print(f"✅ Statistical features shape: {X_stat.shape}")
    return X_stat


def select_important_features_xgboost(X_stat, y, top_k=50):
    """
    ใช้ XGBoost เลือก Top-K Features ที่สำคัญ
    """
    print(f"\n🎯 XGBoost Feature Selection (Top-{top_k})...")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_stat, y)

    # Get Feature Importance
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:][::-1]

    print(f"✅ Selected {top_k} most important features")
    print(f"   Top 5 importance scores: {importances[top_indices[:5]]}")

    return top_indices, model


def reconstruct_sequences_from_features(sequences, selected_indices, n_original_features):
    """
    Reconstruct time-series จาก selected features
    แนวคิด: แต่ละ feature คือ statistic ของ (time_step, feature_idx)
    เราจะเลือกเฉพาะ features ที่สำคัญมาใช้
    """
    # คำนวณว่า feature แต่ละตัวมาจาก (timestep, feature_idx) ไหน
    features_per_channel = 13  # จำนวน stats ต่อ channel (8+3+1+1)

    # หา channel ที่ถูกเลือก
    selected_channels = set()
    for idx in selected_indices:
        channel = idx // features_per_channel
        selected_channels.add(channel)

    selected_channels = sorted(list(selected_channels))

    print(f"🔍 Selected {len(selected_channels)}/{n_original_features} feature channels")

    # เลือกเฉพาะ channels ที่สำคัญ
    sequences_reduced = sequences[:, :, selected_channels]

    return sequences_reduced


# ==========================================
# 🧠 XGBOOST-LSTM HYBRID MODEL
# ==========================================
def create_xgboost_lstm_model(trial, input_shape, num_classes):
    """สร้าง LSTM Model หลัง Feature Selection"""

    # Hyperparameters
    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [64, 100, 128, 150])
    lstm_units_2 = trial.suggest_categorical('lstm_units_2', [32, 64, 100])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    use_two_lstm = trial.suggest_categorical('use_two_lstm', [True, False])

    # Build Model
    model = Sequential()

    # First LSTM
    if use_two_lstm:
        model.add(LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units_2, return_sequences=False))
    else:
        model.add(LSTM(lstm_units_1, return_sequences=False, input_shape=input_shape))

    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Dense Layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate * 0.7))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# ==========================================
# 🎯 OPTUNA OBJECTIVE
# ==========================================
def objective(trial, X_seq_train, y_train, X_seq_val, y_val, num_classes):
    """Optuna objective สำหรับ XGBoost-LSTM"""

    # XGBoost Feature Selection
    top_k = trial.suggest_int('top_k_features', 20, 100, step=10)
    step_size = trial.suggest_categorical('step_size', [5, 10, 15])

    # Extract statistical features
    X_stat_train = extract_statistical_features(X_seq_train)

    # Feature Selection
    selected_indices, xgb_model = select_important_features_xgboost(
        X_stat_train,
        np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train,
        top_k=top_k
    )

    # Reconstruct sequences
    n_original_features = X_seq_train.shape[2]
    X_reduced_train = reconstruct_sequences_from_features(
        X_seq_train, selected_indices, n_original_features
    )
    X_reduced_val = reconstruct_sequences_from_features(
        X_seq_val, selected_indices, n_original_features
    )

    # Build LSTM Model
    input_shape = (X_reduced_train.shape[1], X_reduced_train.shape[2])
    model = create_xgboost_lstm_model(trial, input_shape, num_classes)

    # Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_accuracy')

    history = model.fit(
        X_reduced_train, y_train,
        validation_data=(X_reduced_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=trial.suggest_categorical('batch_size', [32, 64]),
        callbacks=[early_stopping, pruning_callback],
        verbose=0
    )

    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc


# ==========================================
# 📊 EVALUATION FUNCTIONS
# ==========================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost-LSTM Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Confusion Matrix saved: {save_path}")


def evaluate_model(model, X_test, y_test, class_names, save_dir):
    print("\n" + "=" * 60)
    print("🔍 FINAL EVALUATION")
    print("=" * 60)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    return accuracy


# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 XGBoost-LSTM HYBRID MODEL")
    print("=" * 60)

    # Load Data
    X, y, subjects = load_and_process_data(INPUT_FILE, TIME_STEPS, FEATURES, step_size=10)
    num_classes = len(np.unique(y))

    # Split Data
    unique_subjects = np.unique(subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_test, n_val = 5, 2
    test_subs = unique_subjects[:n_test]
    val_subs = unique_subjects[n_test:n_test + n_val]
    train_subs = unique_subjects[n_test + n_val:]

    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # One-Hot Encoding
    y_train_hot = to_categorical(y_train, num_classes)
    y_val_hot = to_categorical(y_val, num_classes)
    y_test_hot = to_categorical(y_test, num_classes)

    print(f"\n📊 Dataset: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Optuna Optimization
    print("\n🔬 Starting Optuna Optimization...")
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train_hot, X_val, y_val_hot, num_classes),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    print("\n🏆 Best Trial Results:")
    print(f"   Validation Accuracy: {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")

    # Train Final Model
    print("\n🔨 Training Final Model with Best Params...")
    best_params = study.best_params

    # Apply best feature selection
    X_stat_train = extract_statistical_features(X_train)
    selected_indices, xgb_model = select_important_features_xgboost(
        X_stat_train, y_train, top_k=best_params['top_k_features']
    )

    X_train_reduced = reconstruct_sequences_from_features(X_train, selected_indices, len(FEATURES))
    X_val_reduced = reconstruct_sequences_from_features(X_val, selected_indices, len(FEATURES))
    X_test_reduced = reconstruct_sequences_from_features(X_test, selected_indices, len(FEATURES))


    # Build final model
    class BestTrial:
        def __init__(self, params):
            self.params = params

        def suggest_categorical(self, name, choices):
            return self.params[name]

        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

        def suggest_int(self, name, low, high, step=1):
            return self.params[name]


    best_trial = BestTrial(best_params)
    input_shape = (X_train_reduced.shape[1], X_train_reduced.shape[2])
    final_model = create_xgboost_lstm_model(best_trial, input_shape, num_classes)

    print("\n🏗️ Final Model Architecture:")
    final_model.summary()

    # Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    final_model.fit(
        X_train_reduced, y_train_hot,
        validation_data=(X_val_reduced, y_val_hot),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )

    # Save
    model_path = os.path.join(OUTPUT_DIR, 'xgboost_lstm_model.h5')
    final_model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Evaluate
    class_names = [f"Class_{i}" for i in range(num_classes)]
    test_acc = evaluate_model(final_model, X_test_reduced, y_test_hot, class_names, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"✅ FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print("=" * 60)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
