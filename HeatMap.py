import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import filedialog
import re  # นำเข้า module สำหรับจัดการข้อความ (Regular Expression)


# ฟังก์ชันสำหรับดึงชื่อกลุ่มข้อมูลจากชื่อไฟล์
def extract_group_name(file_paths):
    if not file_paths:
        return "Unknown_Group"

    # ดึงเฉพาะชื่อไฟล์ (ไม่เอา Path ยาวๆ)
    base_names = [os.path.basename(f) for f in file_paths]

    # หา "คำนำหน้า" ที่เหมือนกันของทุกไฟล์ (Common Prefix)
    # เช่น ถ้าเลือก Normal_Walk_NECK_01 และ Normal_Walk_NECK_02 -> จะได้ Normal_Walk_NECK_0
    common_prefix = os.path.commonprefix(base_names)

    # ลบนามสกุลไฟล์ (.csv) ออก
    name = common_prefix.replace(".csv", "")

    # ลบตัวเลข, ขีดล่าง, หรือคำว่า test ที่อยู่ท้ายสุดออก เพื่อให้เหลือแต่ชื่อกลุ่มหลักๆ
    # เช่น "Normal_Walk_NECK_0" -> "Normal_Walk_NECK"
    clean_name = re.sub(r'[_0-9]+$', '', name)  # ลบตัวเลขและ _ ท้ายคำ
    clean_name = re.sub(r'_test$', '', clean_name, flags=re.IGNORECASE)  # ลบ _test ท้ายคำ

    # กรณีเลือกไฟล์เดียว หรือชื่อไม่เหมือนกันเลย ให้ใช้ชื่อไฟล์แรกแบบตัดส่วนท้าย
    if len(clean_name) < 3:
        clean_name = os.path.splitext(base_names[0])[0]
        clean_name = re.sub(r'[_0-9]+$', '', clean_name)

    return clean_name


# --- 1. ส่วนเลือกไฟล์ (GUI) ---
print("กรุณาเลือกไฟล์ CSV ที่ต้องการวิเคราะห์...")

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames(
    title="เลือกไฟล์ CSV (เช่น กลุ่ม Normal_Walk_NECK)",
    filetypes=[("CSV Files", "*.csv")]
)

if file_paths:
    selected_files = list(file_paths)

    # --- ดึงชื่อกลุ่มมาตั้งเป็นตัวแปร ---
    group_label = extract_group_name(selected_files)
    print(f"\nกลุ่มข้อมูลที่ตรวจพบ: {group_label}")
    print(f"จำนวนไฟล์ที่เลือก: {len(selected_files)} ไฟล์")

    base_folder = os.path.dirname(selected_files[0])
    data_list = []

    # --- 2. อ่านและเตรียมข้อมูล ---
    for file in selected_files:
        df_temp = pd.read_csv(file)
        if len(df_temp) > 350:
            df_temp = df_temp.iloc[350:]
        data_list.append(df_temp)

    if data_list:
        df_all = pd.concat(data_list, ignore_index=True)
        cols_to_analyze = [c for c in df_all.columns if any(x in c for x in ['ACC', 'GYRO', 'MAG'])]
        df_analysis = df_all[cols_to_analyze]

        # --- 3. สร้าง Correlation Matrix ---
        corr_matrix = df_analysis.corr()

        # --- 4. สร้าง Heatmap ---
        plt.figure(figsize=(12, 10))

        # ตั้งชื่อ Title ตามชื่อไฟล์ที่ดึงมา
        plot_title = f'Correlation Heatmap - {group_label}'

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(plot_title, fontsize=16)  # ใส่ Title ที่มีชื่อกลุ่ม
        plt.tight_layout()

        # สร้างโฟลเดอร์ Analysis
        analysis_folder = os.path.join(base_folder, 'Analysis')
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)

        # บันทึกไฟล์โดยใช้ชื่อกลุ่มต่อท้าย
        filename = f'Heatmap_{group_label}.png'
        save_path = os.path.join(analysis_folder, filename)

        plt.savefig(save_path)
        print(f"บันทึกกราฟชื่อ: {filename}")
        plt.show()

    else:
        print("ไฟล์ที่เลือกไม่มีข้อมูล")
else:
    print("ยกเลิกการเลือกไฟล์")
