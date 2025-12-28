import pandas as pd
import numpy as np
import os
import glob

# ==========================================
# ⚙️ SETTINGS (ตั้งค่า)
# ==========================================
# ชื่อคอลัมน์ที่จะใช้ (ต้องมีในไฟล์ CSV ของคุณ)
FEATURE_COLS = ['Gender', 'BMI', 'Z_ACC_SVM_Jerk', 'Z_GYRO_SVM_Jerk']

# การแปลงโฟลเดอร์เป็น Label (0, 1, 2)
LABEL_MAPPING = {
    'Normal': 0, 'Slow': 0,  # 🟢 Low Risk
    'Left': 1, 'Right': 1,  # 🟡 Medium Risk
    'Dunk': 2, 'Stun': 2  # 🔴 High Risk
}


def load_data(root_dir, window_size=128, step_size=64):
    """
    อ่านไฟล์ CSV -> ทำ Sliding Window -> ส่งกลับเป็น X, y
    """
    print("=" * 60)
    print(f"🔄 START: Loading Data & Sliding Window (Size={window_size})")
    print("=" * 60)

    X_data = []
    y_data = []

    # โฟลเดอร์ที่เราเก็บไฟล์ที่ทำ preprocess เสร็จแล้ว
    input_subfolder = "File_Trainmodel"

    # หา Subject ทั้งหมดที่ขึ้นต้นด้วย S_
    subjects = [d for d in os.listdir(root_dir) if d.startswith('S_')]

    if len(subjects) == 0:
        print(f"❌ Error: ไม่พบโฟลเดอร์ Subject ใน {root_dir}")
        return np.array([]), np.array([])

    count_files = 0

    for sub in subjects:
        # วนลูปตาม 6 กรณี (Normal, Dunk, ...)
        for activity, label_code in LABEL_MAPPING.items():

            # Path: E:\...\S_ALIF\File_Trainmodel\Dunk
            folder_path = os.path.join(root_dir, sub, input_subfolder, activity)

            if not os.path.exists(folder_path):
                continue

            # หาไฟล์ _to_Model.csv
            files = glob.glob(os.path.join(folder_path, "*_to_Model.csv"))

            for f in files:
                try:
                    df = pd.read_csv(f)
                    count_files += 1

                    # ดึงค่า 4 คอลัมน์ออกมาเป็น Array
                    # Shape: (จำนวนบรรทัดในไฟล์, 4)
                    data_values = df[FEATURE_COLS].values

                    # --- Sliding Window Logic ---
                    # ตัดข้อมูลทีละ window_size ขยับทีละ step_size
                    num_readings = len(data_values)

                    for i in range(0, num_readings - window_size, step_size):
                        # ตัดช่วง i ถึง i+128
                        window = data_values[i: i + window_size]

                        # เช็คว่าตัดมาครบ 128 บรรทัดไหม (ถ้าไม่ครบไม่เอา)
                        if window.shape[0] == window_size:
                            X_data.append(window)
                            y_data.append(label_code)

                except Exception as e:
                    print(f"⚠️ Error reading {os.path.basename(f)}: {e}")

    # แปลง List เป็น Numpy Array ก้อนใหญ่
    X_final = np.array(X_data)
    y_final = np.array(y_data)

    print("-" * 50)
    print(f"✅ PROCESS COMPLETE")
    print(f"📂 Total Files Processed: {count_files}")
    print(f"📦 X Shape (Features): {X_final.shape}")  # (Samples, 128, 4)
    print(f"🎯 y Shape (Labels):   {y_final.shape}")  # (Samples,)
    print("-" * 50)


    return X_final, y_final
