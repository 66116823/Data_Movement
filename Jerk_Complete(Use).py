import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


# ==========================================
# ตั้งค่าฟอนต์ภาษาไทย
# ==========================================
def set_thai_font():
    if os.name == 'nt':
        plt.rcParams['font.family'] = 'Tahoma'
    else:
        plt.rcParams['font.family'] = 'Ayuthaya'
    plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ส่วนที่ 1: Helper Functions (คำนวณ)
# ==========================================

def calculate_svm_if_needed(df, sensor):
    col_name = f'{sensor}_SVM'
    if col_name in df.columns: return col_name
    cols = [f'{sensor}_X', f'{sensor}_Y', f'{sensor}_Z']
    if all(c in df.columns for c in cols):
        df[col_name] = np.sqrt(df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2)
        return col_name
    return None


def parse_time_column(df):
    if 'Time' not in df.columns: return df.index.to_numpy()
    try:
        def to_sec(t):
            parts = str(t).strip().split(':')
            if len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
            return float(t)

        t_vals = df['Time'].apply(to_sec).to_numpy()
        return t_vals - t_vals[0] if len(t_vals) > 0 else df.index.to_numpy()
    except:
        return df.index.to_numpy()


def normalize_series(data):
    data_abs = np.abs(data)
    min_val = np.min(data_abs)
    max_val = np.max(data_abs)
    if max_val - min_val == 0: return np.zeros_like(data)
    return (data_abs - min_val) / (max_val - min_val)


def apply_filters_and_jerk(signal, time_sec, fs):
    dt = 1.0 / fs
    # 1. MA
    w_ma = int(0.2 * fs)
    if w_ma < 3: w_ma = 3
    sig_ma = pd.Series(signal).rolling(window=w_ma, center=True).mean().bfill().ffill().values
    jerk_ma = np.gradient(sig_ma, dt)
    # 2. Butter
    cutoff = 5.0
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1: normal_cutoff = 0.99
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    sig_butter = filtfilt(b, a, signal)
    jerk_butter = np.gradient(sig_butter, dt)
    # 3. SavGol
    w_sg = int(0.25 * fs)
    if w_sg % 2 == 0: w_sg += 1
    if w_sg < 5: w_sg = 5
    sig_savgol = savgol_filter(signal, window_length=w_sg, polyorder=3)
    jerk_savgol = np.gradient(sig_savgol, dt)

    return {'Jerk_MA': jerk_ma, 'Jerk_Butter': jerk_butter, 'Jerk_SavGol': jerk_savgol}


# ==========================================
# ส่วนที่ 2: ฟังก์ชันกราฟ (รับ Path แยก 3 ประเภท)
# ==========================================

def plot_separate_filters(time_sec, plot_data, title, paths_dict, file_basename):
    """
    สร้างกราฟแยก 3 ไฟล์ ลงใน path ที่กำหนดแยกกัน
    """
    set_thai_font()

    # Config จับคู่ Key กับการตั้งค่า
    filters_config = [
        {'id': 'graph_ma', 'name': 'Moving Average', 'data_key': 'Jerk_MA', 'suffix': '_plot_MA.png'},
        {'id': 'graph_butter', 'name': 'Butterworth', 'data_key': 'Jerk_Butter', 'suffix': '_plot_Butter.png'},
        {'id': 'graph_savgol', 'name': 'Savitzky-Golay', 'data_key': 'Jerk_SavGol', 'suffix': '_plot_SavGol.png'}
    ]

    sensor_colors = {'ACC': 'green', 'GYRO': '#D4AC0D', 'MAG': 'red'}

    for f_conf in filters_config:
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        fig.suptitle(f"Normalized Jerk Analysis ({f_conf['name']}): {title}", fontsize=14, fontweight='bold')

        sensors = ['ACC', 'GYRO', 'MAG']
        for i, sensor in enumerate(sensors):
            ax = axs[i]
            if sensor in plot_data:
                norm_jerk = normalize_series(plot_data[sensor][f_conf['data_key']])
                ax.plot(time_sec, norm_jerk, color=sensor_colors[sensor], label=f'{sensor}', linewidth=1.2)
                ax.set_title(f'{sensor} Normalized Jerk (0-1)')
                ax.set_ylabel('Norm. Mag')
                ax.set_xlabel('Time (s)')
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim([time_sec[0], time_sec[-1]])
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # ดึง Path ตามประเภทกราฟ
        target_dir = paths_dict[f_conf['id']]
        full_path = os.path.join(target_dir, file_basename + f_conf['suffix'])

        plt.savefig(full_path)
        plt.close()


# ==========================================
# ส่วนที่ 3: Logic หลัก (Popup 5 ครั้ง)
# ==========================================

def analyze_smoothness_mode():
    print("\n" + "=" * 70)
    print("   🌊 Mode: SVM -> Filter -> Jerk Analysis")
    print("   📂 Feature: Auto-Popup Folder Selection (เลือกทีละ 5 โฟลเดอร์)")
    print("=" * 70)

    # 1. ข้อมูลการทดลอง
    scenarios = ["กรณีเดินปกติ", "กรณีเดินปกติช้า", "กรณีเดินเอียงขวา", "กรณีเดินเอียงซ้าย", "กรณีเดินเซ",
                 "กรณีเดินหยุดชะงัก"]
    print("\n--- เลือกสถานการณ์ ---")
    for i, s in enumerate(scenarios, 1): print(f" [{i}] {s}")
    c = input("เลือก (1-6): ").strip()
    idx = int(c) - 1 if c.isdigit() and 1 <= int(c) <= 6 else 0
    header_title = f"{scenarios[idx]} : {input('ชื่อผู้ทดลอง: ').strip()}"

    # 2. เลือกไฟล์ Input
    root = tk.Tk();
    root.withdraw();
    root.attributes('-topmost', True)
    file_paths = filedialog.askopenfilenames(
        title="เลือกไฟล์ข้อมูล Input (Excel หรือ CSV)",
        filetypes=[("All Supported", "*.xlsx *.xls *.csv"), ("Excel", "*.xlsx"), ("CSV", "*.csv")]
    )
    if not file_paths: return

    # =================================================================
    # 3. Popup ถามโฟลเดอร์ 5 ครั้ง
    # =================================================================
    print("\n" + "-" * 60)
    print("   ⚙️  ระบบจะเปิดหน้าต่างให้เลือกโฟลเดอร์ 5 ครั้ง ตามลำดับ...")
    print("       (หากกด Cancel/ปิด = เก็บที่เดียวกับไฟล์ต้นฉบับ)")
    print("-" * 60)

    output_dirs = {}

    # รายการขั้นตอนการเก็บไฟล์
    steps = [
        ('timeseries', '1/5 เลือกโฟลเดอร์เก็บ: ข้อมูลดิบ (_timeseries.csv)'),
        ('summary', '2/5 เลือกโฟลเดอร์เก็บ: ตารางสรุป (_summary.csv)'),
        ('graph_ma', '3/5 เลือกโฟลเดอร์เก็บ: กราฟ Moving Average'),
        ('graph_butter', '4/5 เลือกโฟลเดอร์เก็บ: กราฟ Butterworth'),
        ('graph_savgol', '5/5 เลือกโฟลเดอร์เก็บ: กราฟ Savitzky-Golay')
    ]

    for key, title_text in steps:
        print(f"   📂 กำลังเปิดหน้าต่าง: {title_text} ...")

        # ทำให้หน้าต่างเด้งมาอยู่บนสุดเสมอ
        root.attributes('-topmost', True)
        path = filedialog.askdirectory(title=title_text)

        if path:
            print(f"      👉 เลือก: {path}")
            output_dirs[key] = path
        else:
            print(f"      👉 (User ยกเลิก) -> จะเก็บไว้ที่เดียวกับไฟล์ต้นฉบับ")
            output_dirs[key] = None  # เก็บเป็น None ไว้ก่อน เดี๋ยวไปแทนค่าทีหลัง

    print("\n" + "-" * 60)
    print("   ✅ ตั้งค่าโฟลเดอร์เสร็จสิ้น! เริ่มประมวลผล...")
    print("-" * 60)
    # =================================================================

    for file_idx, file_path in enumerate(file_paths, 1):
        try:
            filename = os.path.basename(file_path)
            basename_only = os.path.splitext(filename)[0]
            source_dir = os.path.dirname(file_path)

            print(f"\n[{file_idx}/{len(file_paths)}] 📄 Processing: {filename}")

            # ฟังก์ชันช่วยเลือก Path (ถ้า User กด Cancel ให้ใช้ Source Dir)
            def resolve_path(key_name):
                return output_dirs[key_name] if output_dirs[key_name] else source_dir

            # อ่านไฟล์
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')

            time_sec = parse_time_column(df)
            diffs = np.diff(time_sec)
            valid_diffs = diffs[diffs > 0]
            fs = 1.0 / np.mean(valid_diffs) if len(valid_diffs) > 0 else 50.0

            final_stats = []
            ts_data = {'Time': time_sec}
            plot_data = {}
            sensors = ['ACC', 'GYRO', 'MAG']

            for sensor in sensors:
                svm_col = calculate_svm_if_needed(df, sensor)
                if not svm_col: continue
                res = apply_filters_and_jerk(df[svm_col].values, time_sec, fs)
                plot_data[sensor] = res

                ts_data[f'{sensor}_Jerk_MA'] = res['Jerk_MA']
                ts_data[f'{sensor}_Jerk_Butter'] = res['Jerk_Butter']
                ts_data[f'{sensor}_Jerk_SavGol'] = res['Jerk_SavGol']

                for method in ['Moving Average', 'Butterworth', 'Savitzky-Golay']:
                    if 'Moving' in method:
                        key = 'Jerk_MA'
                    elif 'Butter' in method:
                        key = 'Jerk_Butter'
                    else:
                        key = 'Jerk_SavGol'
                    final_stats.append({
                        'Experiment': header_title,
                        'Sensor': sensor,
                        'Method': method,
                        'Max Jerk': np.max(np.abs(res[key])),
                        'Smoothness': np.mean(np.abs(res[key]))
                    })

            # --- Save Output ตาม Path ที่เลือกจาก Popup ---
            if plot_data:
                # 1. ส่ง Path กราฟทั้ง 3 ไปให้ฟังก์ชัน plot
                graph_paths = {
                    'graph_ma': resolve_path('graph_ma'),
                    'graph_butter': resolve_path('graph_butter'),
                    'graph_savgol': resolve_path('graph_savgol')
                }
                plot_separate_filters(time_sec, plot_data, header_title, graph_paths, basename_only)

                # 2. Save CSV - TimeSeries
                ts_path = os.path.join(resolve_path('timeseries'), basename_only + "_jerk_timeseries.csv")
                pd.DataFrame(ts_data).to_csv(ts_path, index=False, encoding='utf-8-sig')

                # 3. Save CSV - Summary
                sum_path = os.path.join(resolve_path('summary'), basename_only + "_jerk_summary.csv")
                pd.DataFrame(final_stats).to_csv(sum_path, index=False, encoding='utf-8-sig')

                print(f"      ✔ Saved 5 files successfully.")

        except Exception as e:
            print(f"      ❌ Error: {e}")

    print("\n✅ เสร็จสิ้นกระบวนการ!");
    input("กด Enter เพื่อออก...")


if __name__ == "__main__":
    analyze_smoothness_mode()
