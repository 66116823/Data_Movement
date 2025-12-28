import pandas as pd
import os
import glob

# ==========================================
# ⚙️ CONFIGURATION (ตั้งค่า Path)
# ==========================================
ROOT_DIR = r"E:\Data Movement\CollectDATA\Subject"
INPUT_SUBFOLDER = r"Complete_Preprocess\Data"
OUTPUT_SUBFOLDER = r"File_Trainmodel"

# รายชื่อ 6 กรณี
SCENARIOS = ['Dunk', 'Left', 'Normal', 'Right', 'Slow', 'Stun']

# ชื่อคอลัมน์ที่จะดึง (ต้องตรงกับในไฟล์ CSV เป๊ะๆ)
COL_RAW_DATE = 'Date'
COL_RAW_TIME = 'Time'
COL_Z_ACC = 'Z_ACC_SVM_Jerk'
COL_Z_GYRO = 'Z_GYRO_SVM_Jerk'


# ==========================================
# 🛠️ FUNCTIONS
# ==========================================

def get_user_input(subject_name):
    """ฟังก์ชันขอข้อมูล Gender และ BMI จากผู้ใช้"""
    print("=" * 60)
    print(f"👤 กำลังจัดการข้อมูลของ: {subject_name}")
    print("=" * 60)

    while True:
        try:
            g_input = input(f"   👉 กรุณาระบุเพศสำหรับ {subject_name} (0=ชาย, 1=หญิง): ")
            if g_input in ['0', '1']:
                gender = int(g_input)
                break
            print("   ❌ ผิดพลาด! ใส่ได้แค่ 0 หรือ 1 เท่านั้น")
        except ValueError:
            pass

    while True:
        try:
            b_input = input(f"   👉 กรุณาระบุ BMI สำหรับ {subject_name} (ทศนิยม): ")
            bmi = float(b_input)
            break
        except ValueError:
            print("   ❌ ผิดพลาด! กรุณาใส่เป็นตัวเลข (เช่น 22.5)")

    return gender, bmi


def process_data():
    # 1. ค้นหา Subject ทั้งหมด
    subjects = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.startswith("S_")]

    if not subjects:
        print("❌ ไม่พบโฟลเดอร์ Subject ใน Path ที่กำหนด")
        return

    # 2. วนลูปทีละคน
    for sub in subjects:
        # --- ถาม User Input (Gender, BMI) ---
        gender, bmi = get_user_input(sub)

        # 3. วนลูปทีละกรณี (Dunk, Normal, ...)
        for case in SCENARIOS:
            # Path ขาเข้า
            input_path = os.path.join(ROOT_DIR, sub, INPUT_SUBFOLDER, case)

            # Path ขาออก (สร้างรอไว้เลย)
            output_path = os.path.join(ROOT_DIR, sub, OUTPUT_SUBFOLDER, case)
            os.makedirs(output_path, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"   ⚠️ ไม่พบโฟลเดอร์: {input_path} (ข้าม)")
                continue

            # หาไฟล์ _RawFeatures ทั้งหมดเพื่อใช้เป็นตัวตั้งต้น
            raw_files = glob.glob(os.path.join(input_path, "*_RawFeatures.csv"))

            print(f"   📂 Processing {case}: พบ {len(raw_files)} ไฟล์")

            for raw_f in raw_files:
                try:
                    # สร้างชื่อไฟล์ ZScores คู่กัน
                    # เช่น Dunk_Walk_ALIF_01_RawFeatures.csv -> Dunk_Walk_ALIF_01_ZScores.csv
                    base_name = os.path.basename(raw_f).replace("_RawFeatures.csv", "")
                    zscore_f = os.path.join(input_path, f"{base_name}_ZScores.csv")

                    if not os.path.exists(zscore_f):
                        print(f"      ❌ ไม่พบไฟล์คู่ ZScores ของ: {base_name}")
                        continue

                    # อ่านไฟล์ CSV
                    df_raw = pd.read_csv(raw_f)
                    df_z = pd.read_csv(zscore_f)

                    # ตรวจสอบจำนวนแถว (ควรเท่ากัน) แต่ถ้าไม่เท่าจะยึดตาม Raw เป็นหลัก
                    min_len = min(len(df_raw), len(df_z))
                    df_raw = df_raw.iloc[:min_len]
                    df_z = df_z.iloc[:min_len]

                    # --- สร้าง DataFrame ใหม่ ---
                    df_new = pd.DataFrame()

                    # Col A: Date
                    df_new['Date'] = df_raw[COL_RAW_DATE]

                    # Col B: Time
                    df_new['Time'] = df_raw[COL_RAW_TIME]

                    # Col C: Gender (ใส่ค่าเดียวซ้ำทุกบรรทัด)
                    df_new['Gender'] = gender

                    # Col D: BMI (ใส่ค่าเดียวซ้ำทุกบรรทัด)
                    df_new['BMI'] = bmi

                    # Col E: Z_ACC_SVM_Jerk
                    df_new['Z_ACC_SVM_Jerk'] = df_z[COL_Z_ACC]

                    # Col F: Z_GYRO_SVM_Jerk
                    df_new['Z_GYRO_SVM_Jerk'] = df_z[COL_Z_GYRO]

                    # บันทึกไฟล์ใหม่ (*_to_Model.csv)
                    output_filename = f"{base_name}_to_Model.csv"
                    save_file = os.path.join(output_path, output_filename)

                    df_new.to_csv(save_file, index=False)
                    # print(f"      ✅ Saved: {output_filename}") # เปิดบรรทัดนี้ถ้าอยากเห็นทุกไฟล์

                except Exception as e:
                    print(f"      💀 Error processing {os.path.basename(raw_f)}: {e}")

    print("\n🎉 เสร็จสิ้นกระบวนการทั้งหมดเรียบร้อยแล้วครับ!")


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    process_data()