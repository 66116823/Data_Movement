import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

# Seed
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
# เปลี่ยนจาก LSTM เป็น GRU
from tensorflow.keras.layers import Dense, Dropout, GRU, Conv1D, MaxPooling1D, Bidirectional, Input, Layer, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from data_loader import load_data

# ==========================================
# 1. CONFIGURATION (V14 - GRU Switch)
# ==========================================
ROOT_DIR = r"D:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"D:\Data Movement\CollectDATA\Model"

WINDOW_SIZE = 128
STEP_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. CUSTOM ATTENTION
# ==========================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# ==========================================
# 3. LOAD DATA
# ==========================================
print(f"\n🔄 Loading Data V14...")
try:
    X, y = load_data(ROOT_DIR, WINDOW_SIZE, STEP_SIZE)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

if len(X) == 0:
    print("❌ Empty Data")
    exit()

y_onehot = to_categorical(y, num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y)

# --- 🔥 Class Weight (สูตร V12 ที่ดีที่สุด) ---
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weight_dict = dict(enumerate(class_weights))

# Boost High Risk & Medium Risk
class_weight_dict[2] = class_weight_dict[2] * 1.5
class_weight_dict[1] = class_weight_dict[1] * 1.2
print(f"🚀 Weights: {class_weight_dict}")

# Scaler
scaler = StandardScaler()
N_train, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_train, T, F)
N_test, T, F = X_test.shape
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

# ==========================================
# 🔥 4. BUILD MODEL (V14 - GRU Edition)
# ==========================================
inputs = Input(shape=(WINDOW_SIZE, F))

# --- CNN Block 1 (เหมือน V12) ---
x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

# --- CNN Block 2 (เหมือน V12) ---
x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

# --- 🔥 GRU Block (เปลี่ยนจาก LSTM) ---
# GRU ทำงานได้ดีกับข้อมูลที่มีความผันผวน และใช้ Parameter น้อยกว่า
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# --- Attention ---
x = Attention()(x)

# --- Dense ---
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(3, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
model.summary()

# ==========================================
# 5. TRAIN
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

print("\n🚀 START TRAINING (V14 - GRU)...")
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
# 6. REPORT
# ==========================================
print("\n📝 Generating Results...")
# Graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title('Acc V14'); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title('Loss V14'); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "result_v14_graph.png")); plt.show()

# Matrix & Report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n--- Classification Report (V14) ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (V14)'); plt.savefig(os.path.join(OUTPUT_DIR, "result_v14_cm.png")); plt.show()
model.save(os.path.join(OUTPUT_DIR, "gait_model_v14_gru.h5"))
