import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


# ==========================================
# ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ
# ==========================================
def set_thai_font():
    if os.name == 'nt':  # Windows
        plt.rcParams['font.family'] = 'Tahoma'
    else:  # Mac/Linux
        plt.rcParams['font.family'] = 'Ayuthaya'
    plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ส่วนที่ 1: Helper Functions
# ==========================================

def calculate_svm_if_needed(df, sensor):
    """ตรวจสอบว่ามี SVM หรือยัง ถ้าไม่มีให้คำนวณใหม่"""
    col_name = f'{sensor}_SVM'
    if col_name in df.columns:
        return col_name
    cols = [f'{sensor}_X', f'{sensor}_Y', f'{sensor}_Z']
    if all(c in df.columns for c in cols):
        df[col_name] = np.sqrt(df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2)
        return col_name
    return None


def parse_time_column(df):
    """แปลงเวลาเป็นวินาที (เริ่มที่ 0)"""
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
    """แปลงข้อมูลให้อยู่ในช่วง 0 ถึง 1 (Min-Max Normalization)"""
    data_abs = np.abs(data)
    min_val = np.min(data_abs)
    max_val = np.max(data_abs)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data_abs - min_val) / (max_val - min_val)


def apply_filters_and_jerk(signal, time_sec, fs):
    """คำนวณ Jerk ด้วย 3 วิธี (ใช้ dt แก้ปัญหา Divide by Zero)"""
    dt = 1.0 / fs

    # 1. Moving Average
    w_ma = int(0.2 * fs)
    if w_ma < 3: w_ma = 3
    sig_ma = pd.Series(signal).rolling(window=w_ma, center=True).mean().bfill().ffill().values
    jerk_ma = np.gradient(sig_ma, dt)

    # 2. Butterworth Low-Pass
    cutoff = 5.0
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1: normal_cutoff = 0.99
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    sig_butter = filtfilt(b, a, signal)
    jerk_butter = np.gradient(sig_butter, dt)

    # 3. Savitzky-Golay
    w_sg = int(0.25 * fs)
    if w_sg % 2 == 0: w_sg += 1
    if w_sg < 5: w_sg = 5
    sig_savgol = savgol_filter(signal, window_length=w_sg, polyorder=3)
    jerk_savgol = np.gradient(sig_savgol, dt)

    return {
        'Jerk_MA': jerk_ma,
        'Jerk_Butter': jerk_butter,
        'Jerk_SavGol': jerk_savgol
    }


# ==========================================
# ส่วนที่ 2: ฟังก์ชันสร้างกราฟ (แยกไฟล์ + สีตามธีม + แกน X ทุกรูป)
# ==========================================

def plot_separate_filters(time_sec, plot_data, title, filename_base):
    """
    สร้างกราฟแยก 3 ไฟล์ตามชนิด Filter
    สี: ACC=Green, GYRO=Yellow, MAG=Red
    **แก้ไข: แสดงแกน X (Time) ในทุก Subplot**
    """
    set_thai_font()

    # Config ของแต่ละ Filter
    filters_config = [
        {'name': 'Moving Average', 'key': 'Jerk_MA', 'file_suffix': '_plot_MA.png'},
        {'name': 'Butterworth', 'key': 'Jerk_Butter', 'file_suffix': '_plot_Butter.png'},
        {'name': 'Savitzky-Golay', 'key': 'Jerk_SavGol', 'file_suffix': '_plot_SavGol.png'}
    ]

    # Theme สี
    sensor_colors = {
        'ACC': 'green',
        'GYRO': '#D4AC0D',
        'MAG': 'red'
    }

    saved_files = []

    for f_conf in filters_config:
        # **แก้ไข: sharex=False เพื่อให้แต่ละกราฟมีแกน X ของตัวเอง**
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        fig.suptitle(f"Normalized Jerk Analysis ({f_conf['name']}): {title}", fontsize=14, fontweight='bold')

        sensors = ['ACC', 'GYRO', 'MAG']

        for i, sensor in enumerate(sensors):
            ax = axs[i]
            if sensor in plot_data:
                raw_jerk = plot_data[sensor][f_conf['key']]
                norm_jerk = normalize_series(raw_jerk)

                ax.plot(time_sec, norm_jerk,
                        color=sensor_colors[sensor],
                        label=f'{sensor} ({f_conf["name"]})',
                        linewidth=1.2)

                ax.set_title(f'{sensor} Normalized Jerk (0-1)')
                ax.set_ylabel('Norm. Mag (0-1)')
                ax.set_ylim(-0.05, 1.05)

                # **แก้ไข: ใส่ Label แกน X ให้ทุกกราฟ**
                ax.set_xlabel('Time (seconds)')
                ax.set_xlim([time_sec[0], time_sec[-1]])  # ล็อกช่วงเวลาให้เท่ากันทุกกราฟ

                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = filename_base + f_conf['file_suffix']
        plt.savefig(save_path)
        plt.close()
        saved_files.append(os.path.basename(save_path))

    return saved_files


# ==========================================
# ส่วนที่ 3: Logic หลัก
# ==========================================

def analyze_smoothness_mode():
    print("\n" + "=" * 60)
    print("   🌊 Mode: SVM -> Filter -> Jerk Analysis")
    print("   📊 Graph: 3 Files (Theme: ACC🟢 GYRO🟡 MAG🔴)")
    print("   📥 Input: Excel (.xlsx) OR CSV (.csv)")
    print("=" * 60)

    # 1. รับข้อมูล
    scenarios = ["กรณีเดินปกติ", "กรณีเดินปกติช้า", "กรณีเดินเอียงขวา", "กรณีเดินเอียงซ้าย", "กรณีเดินเซ",
                 "กรณีเดินหยุดชะงัก"]
    print("\n--- เลือกสถานการณ์ ---")
    for i, s in enumerate(scenarios, 1): print(f" [{i}] {s}")
    c = input("เลือก (1-6): ").strip()
    idx = int(c) - 1 if c.isdigit() and 1 <= int(c) <= 6 else 0
    header_title = f"{scenarios[idx]} : {input('ชื่อผู้ทดลอง: ').strip()}"

    # 2. เลือกไฟล์
    root = tk.Tk();
    root.withdraw();
    root.attributes('-topmost', True)
    file_paths = filedialog.askopenfilenames(
        title="เลือกไฟล์ข้อมูล (Excel หรือ CSV)",
        filetypes=[("All Supported", "*.xlsx *.xls *.csv"), ("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv")]
    )

    if not file_paths: return

    for file_idx, file_path in enumerate(file_paths, 1):
        try:
            filename = os.path.basename(file_path)
            print(f"\n[{file_idx}/{len(file_paths)}] 📄 กำลังประมวลผล: {filename}")

            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')

            time_sec = parse_time_column(df)

            diffs = np.diff(time_sec)
            valid_diffs = diffs[diffs > 0]
            if len(valid_diffs) > 0:
                dt_avg = np.mean(valid_diffs)
                fs = 1.0 / dt_avg
            else:
                fs = 50.0

            print(f"   -> Sampling Rate detected: {fs:.2f} Hz")

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
                        'Smoothness (Mean Abs Jerk)': np.mean(np.abs(res[key]))
                    })

            base_name = os.path.splitext(file_path)[0]

            if plot_data:
                created_files = plot_separate_filters(time_sec, plot_data, header_title, base_name)
                for f in created_files:
                    print(f"   ✔ Created Graph: {f}")

            pd.DataFrame(final_stats).to_csv(base_name + "_jerk_summary.csv", index=False, encoding='utf-8-sig')
            pd.DataFrame(ts_data).to_csv(base_name + "_jerk_timeseries.csv", index=False, encoding='utf-8-sig')
            print(f"   ✔ Saved CSV Files Successfully")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    print("\n✅ เสร็จสิ้นทุกขั้นตอน!");
    input("กด Enter เพื่อออก...")


if __name__ == "__main__":
    analyze_smoothness_mode()