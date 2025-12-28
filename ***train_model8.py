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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional, Input, Flatten, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

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
LEARNING_RATE = 0.001 # เริ่มต้นเท่าเดิม แต่มีตัวลด LR ช่วย
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
        # สร้าง Weight เพื่อเรียนรู้ว่า Time step ไหนสำคัญ
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
        # x shape: (batch_size, time_steps, features)
        # คำนวณ Score ความสำคัญ
        e = K.tanh(K.dot(x, self.W) + self.b)
        # แปลงเป็น Probability (ผลรวมเป็น 1)
        a = K.softmax(e, axis=1)
        # เอา Probability ไปคูณกับค่าจริง (ถ่วงน้ำหนัก)
        output = x * a
        # รวมผลลัพธ์ทั้งหมด (Sum over time steps)
        return K.sum(output, axis=1)

# ==========================================
# 3. LOAD & PREP DATA
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

# Class Weights (Balanced)
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
# 🔥 4. BUILD MODEL (V8 - Functional API)
# ==========================================
# ต้องใช้ Functional API เพื่อให้ต่อง่ายขึ้น
inputs = Input(shape=(WINDOW_SIZE, F))

# 1. CNN Layers (Feature Extraction)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(filters=128, kernel_size=3, activation='relu')(x) # เพิ่ม Filter
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

# 2. Bi-LSTM Layer
# *สำคัญ* return_sequences=True เพื่อส่ง Time steps ทั้งหมดให้ Attention
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.4)(x)

# 3. Attention Layer
x = Attention()(x)

# 4. Dense Layers (Classification)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
model.summary()

# ==========================================
# 5. TRAIN
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1) # เพิ่ม patience เป็น 20 ให้โอกาสนานขึ้น

print("\n🚀 START TRAINING (V8 - Attention)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# ==========================================
# 6. EVALUATE
# ==========================================
print("\n📝 Generating Graphs...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy (V8 Attention)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (V8 Attention)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "result_v8_attention_graph.png"))
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Classification Report (V8 Attention) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V8 Attention)')
plt.savefig(os.path.join(OUTPUT_DIR, "result_v8_attention_cm.png"))
plt.show()

model.save(os.path.join(OUTPUT_DIR, "gait_model_v8_attention.h5"))
print(f"\n💾 Saved V8 to: {OUTPUT_DIR}")
