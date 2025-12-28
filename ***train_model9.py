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
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from data_loader import load_data

# ==========================================
# 1. CONFIGURATION (V9 - Revised)
# ==========================================
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"

WINDOW_SIZE = 128   # Safe Zone
STEP_SIZE = 64      

# 🔥 จุดปรับปรุงของ V9 นี้:
BATCH_SIZE = 16       # ลดลงเพื่อให้เรียนละเอียด (เดิม 32)
EPOCHS = 100
LEARNING_RATE = 0.001 # ใช้ค่ามาตรฐาน (ไม่ลดเป็น 0.0001 แล้ว)
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. CUSTOM ATTENTION LAYER
# ==========================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# ==========================================
# 3. LOAD & PREP DATA
# ==========================================
print(f"\n🔄 Loading Data for V9 (Window {WINDOW_SIZE})...")
try:
    X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

if len(X) == 0:
    print("❌ X is empty!")
    exit()

y_onehot = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

# Class Weights
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️ Weights: {class_weight_dict}")

# Normalization
scaler = StandardScaler()
N_train, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_train, T, F)
N_test, T, F = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

# ==========================================
# 🔥 4. BUILD MODEL (V9 - Wider Vision)
# ==========================================
inputs = Input(shape=(WINDOW_SIZE, F))

# --- CNN Layers (Kernel Size = 5) ---
# ขยายมุมมองให้กว้างขึ้น เพื่อจับ Pattern การเดินได้ดีขึ้น
x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

# --- Bi-LSTM Layer ---
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.4)(x)

# --- Attention Layer ---
x = Attention()(x)

# --- Dense Layers ---
# ลด L2 ลงเหลือ 0.0005 (จากเดิม 0.001) เพื่อให้โมเดลยืดหยุ่นขึ้น
x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=LEARNING_RATE), 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. TRAIN
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("\n🚀 START TRAINING (V9 - Wider Vision)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE, # 16
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# ==========================================
# 6. EVALUATE
# ==========================================
print("\n📝 Generating Graphs...")
# Plot Accuracy/Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy (V9)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (V9)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "result_v9_graph.png"))
plt.show()

# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Report
print("\n--- Classification Report (V9) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V9)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, "result_v9_cm.png"))
plt.show()

# Save Model
model.save(os.path.join(OUTPUT_DIR, "gait_model_v9_wider.h5"))
print(f"\n💾 Saved V9 to: {os.path.join(OUTPUT_DIR, 'gait_model_v9_wider.h5')}")
