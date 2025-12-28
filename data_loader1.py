import pandas as pd
import numpy as np
import os
import glob

# ==========================================
# ⚙️ SETTINGS
# ==========================================
# เรายังคงใช้ชื่อคอลัมน์เดิม แต่ค่าข้างในจะถูกเปลี่ยนเป็น 0, 1, 2
FEATURE_COLS = ['Gender', 'BMI', 'Z_ACC_SVM_Jerk', 'Z_GYRO_SVM_Jerk']

LABEL_MAPPING = {
    'Normal': 0, 'Slow': 0,   # Low Risk
    'Left': 1, 'Right': 1,    # Medium Risk
    'Dunk': 2, 'Stun': 2      # High Risk
}

# 🟢 ฟังก์ชันแปลง BMI เป็นกลุ่ม (0, 1, 2)
def convert_bmi_to_category(bmi_value):
    if bmi_value < 18.5:
        return 0  # ผอม (Low Weight)
    elif 18.5 <= bmi_value < 25.0:
        return 1  # ปกติ (Normal)
    else:
        return 2  # สูงกว่าเกณฑ์ (Overweight/Obese)

def load_data(root_dir, window_size=128, step_size=64):
    print("="*60)
    print(f"🔄 START: Loading Data & Converting BMI (0,1,2)")
    print("="*60)
    
    X_data = []
    y_data = []
    
    input_subfolder = "File_Trainmodel"
    subjects = [d for d in os.listdir(root_dir) if d.startswith('S_')]
    
    if len(subjects) == 0:
        print(f"❌ Error: ไม่พบโฟลเดอร์ Subject ใน {root_dir}")
        return np.array([]), np.array([])

    count_files = 0
    
    for sub in subjects:
        for activity, label_code in LABEL_MAPPING.items():
            folder_path = os.path.join(root_dir, sub, input_subfolder, activity)
            
            if not os.path.exists(folder_path):
                continue
            
            files = glob.glob(os.path.join(folder_path, "*_to_Model.csv"))
            
            for f in files:
                try:
                    df = pd.read_csv(f)
                    count_files += 1

                    # 🔥🔥🔥 เพิ่มจุดแปลง BMI ตรงนี้ครับ 🔥🔥🔥
                    # ใช้ .apply เพื่อแปลงทุกแถวในคอลัมน์ BMI
                    df['BMI'] = df['BMI'].apply(convert_bmi_to_category)

                    # ดึงค่าตามปกติ (ตอนนี้ BMI เป็น 0,1,2 แล้ว)
                    data_values = df[FEATURE_COLS].values
                    
                    # Sliding Window Logic
                    for i in range(0, len(data_values) - window_size, step_size):
                        window = data_values[i : i + window_size]
                        if window.shape[0] == window_size:
                            X_data.append(window)
                            y_data.append(label_code)
                            
                except Exception as e:
                    print(f"⚠️ Error reading {os.path.basename(f)}: {e}")

    X_final = np.array(X_data)
    y_final = np.array(y_data)
    
    print("-" * 50)
    print(f"✅ PROCESS COMPLETE")
    print(f"📂 Total Files Processed: {count_files}")
    print(f"📦 X Shape: {X_final.shape}")
    print(f"🎯 y Shape: {y_final.shape}")
    print("-" * 50)
    
    return X_final, y_final
