# หมายเหตุ ควรติดตั้ง pip install pandas openpyxl scipy ใน terminal ก่อน

import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from openpyxl.styles import Font  # ใช้สำหรับทำตัวหนาใน Excel

def get_experiment_info():
    """ฟังก์ชันสำหรับแสดงเมนูเลือกสถานการณ์และรับชื่อผู้ทดลอง"""
    scenarios = [
        "กรณีเดินปกติ",
        "กรณีเดินปกติช้า",
        "กรณีเดินเอียงขวา",
        "กรณีเดินเอียงซ้าย",
        "กรณีเดินเอียงซ้ายขวา",
        "กรณีเดินหยุดชะงัก"
    ]

    print("\n" + "=" * 40)
    print("   กรุณาเลือกสถานการณ์การทดลอง (Scenario)")
    print("=" * 40)
    for index, scenario in enumerate(scenarios, 1):
        print(f" [{index}] {scenario}")
    print("-" * 40)

    while True:
        try:
            choice = input("เลือกหมายเลข (1-6): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 6:
                selected_scenario = scenarios[int(choice) - 1]
                break
            else:
                print("❌ ข้อมูลไม่ถูกต้อง กรุณาเลือกตัวเลข 1-6 เท่านั้น")
        except Exception:
            print("❌ ข้อมูลไม่ถูกต้อง")

    print("-" * 40)
    subject_name = input("กรุณาระบุชื่อผู้เข้าทดลอง: ").strip()

    # รวมข้อความตามรูปแบบที่ต้องการ -> กรณี... : ชื่อ...
    full_title = f"{selected_scenario} : {subject_name}"
    return full_title


def process_sensor_file(file_path, header_title):
    # 1. ตรวจสอบไฟล์
    if not os.path.exists(file_path):
        print(f"Error: ไม่พบไฟล์ที่ระบุ: {file_path}")
        return

    print(f"\nกำลังประมวลผลไฟล์: {file_path} ...")

    try:
        # 2. อ่านไฟล์ CSV
        df = pd.read_csv(file_path)

        # รายชื่อคอลัมน์ที่ต้องการคำนวณ
        target_columns = [
            'ACC_X', 'ACC_Y', 'ACC_Z',
            'GYRO_X', 'GYRO_Y', 'GYRO_Z',
            'MAG_X', 'MAG_Y', 'MAG_Z'
        ]

        valid_columns = [col for col in target_columns if col in df.columns]

        if not valid_columns:
            print("Error: ไม่พบคอลัมน์ข้อมูลที่ต้องการคำนวณเลย")
            return

        # 3. คำนวณค่าทางสถิติ
        stats_data = {}
        for col in valid_columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                stats_data[col] = {
                    'Mean': series.mean(),
                    'Median': series.median(),
                    'Skewness': series.skew(),
                    'Kurtosis': series.kurt(),
                    'Max': series.max(),
                    'Min': series.min()
                }

        # สร้าง DataFrame (ตัวแปรอยู่ด้านบน, สถิติอยู่ด้านซ้าย)
        summary_df = pd.DataFrame(stats_data)

        # 4. บันทึกผลลัพธ์ลงไฟล์ Excel (.xlsx)
        output_filename = os.path.splitext(file_path)[0] + '_summary.xlsx'

        print("กำลังบันทึกไฟล์ Excel...")
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Sheet 1: ข้อมูลดิบ
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Sheet 2: ตารางสรุป
            # startrow=2 คือเริ่มเขียนตารางที่บรรทัดที่ 3 (เว้นที่ไว้ให้หัวข้อ)
            summary_df.to_excel(writer, sheet_name='Summary Stats', startrow=2, index=True)

            # --- ส่วนการเขียนหัวข้อ (Header Title) ---
            workbook = writer.book
            worksheet = writer.sheets['Summary Stats']

            # เขียนข้อความที่เซลล์ A1
            worksheet['A1'] = "การทดลอง:"
            worksheet['B1'] = header_title

            # จัดรูปแบบตัวหนา (Bold) ให้สวยงาม
            bold_font = Font(bold=True, size=12)
            worksheet['A1'].font = bold_font
            worksheet['B1'].font = bold_font

        print("\n" + "=" * 40)
        print("✅ เสร็จสมบูรณ์!")
        print(f"📄 หัวข้อตาราง: {header_title}")
        print(f"📁 ไฟล์ผลลัพธ์: {output_filename}")
        print("=" * 40)

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")


if __name__ == "__main__":
    # 1. รับค่า Scenario และ ชื่อผู้ทดลอง ผ่าน Terminal ก่อน
    experiment_title = get_experiment_info()

    # 2. เปิดหน้าต่างเลือกไฟล์
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("\n⏳ กรุณาเลือกไฟล์ CSV จากหน้าต่างที่ปรากฏขึ้นมา...")

    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์ CSV ข้อมูลเซนเซอร์",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if file_path:
        process_sensor_file(file_path, experiment_title)
    else:

        print("❌ ไม่ได้เลือกไฟล์ โปรแกรมจบการทำงาน")
