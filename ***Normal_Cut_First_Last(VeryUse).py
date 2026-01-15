import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================

# 📂 PATH หลัก (แก้ตรงนี้ถ้าเปลี่ยนเครื่องสำหรับโหมด Auto)
ROOT_DIR = r"E:\Data Movement\CollectDATA\Subject"

CONFIG_FILE = 'global_standard_config.json'
GLOBAL_STATS = None

# 🎯 เกณฑ์วัดการขยับตัว (Threshold) สำหรับ Gyro SVM (Degree/s)
MOVEMENT_THRESHOLD = 15.0


def set_thai_font():
    """ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ"""
    if os.name == 'nt':
        plt.rcParams['font.family'] = 'Tahoma'
    else:
        plt.rcParams['font.family'] = 'Ayuthaya'
    plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 🛠️ HELPER: TIME & SORTING & TRIMMING
# ==========================================
def parse_time_column(df):
    """แปลงเวลาเป็นวินาที (Seconds) และ Reset ให้เริ่มที่ 0"""
    if 'Time' not in df.columns: return df.index.to_numpy()
    try:
        time_str = df['Time'].astype(str)

        def t2s(t):
            p = t.strip().split(':')
            if len(p) == 2:
                return float(p[0]) * 60 + float(p[1])
            elif len(p) == 3:
                return float(p[0]) * 3600 + float(p[1]) * 60 + float(p[2])
            return 0.0

        sec = time_str.apply(t2s).to_numpy()
        return sec - sec[0] if len(sec) > 0 else df.index.to_numpy()
    except:
        return df.index.to_numpy()


def sort_data_by_time(df):
    """เรียงลำดับข้อมูลตามเวลา"""
    if 'Time' not in df.columns:
        return df
    try:
        temp_time = parse_time_column(df)
        df['_temp_sort_idx'] = temp_time
        df = df.sort_values(by='_temp_sort_idx')
        df = df.drop(columns=['_temp_sort_idx'])
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(f"⚠️ Sorting Warning: {e}")
        return df


def trim_initial_data(df, cut_seconds=5.0):
    """ตัดข้อมูล 5 วินาทีแรกทิ้ง (Fixed Cut)"""
    try:
        temp_time = parse_time_column(df)
        df_trimmed = df[temp_time >= cut_seconds].copy()
        df_trimmed.reset_index(drop=True, inplace=True)
        return df_trimmed
    except Exception as e:
        print(f"Error trimming initial data: {e}")
        return df


def trim_idle_start_end(df, threshold=MOVEMENT_THRESHOLD):
    """
    ✅ Adaptive Trimming: ตัดช่วงนิ่งหัว-ท้าย อัตโนมัติ
    """
    gyro_cols = ['GYRO_X', 'GYRO_Y', 'GYRO_Z']
    if not all(col in df.columns for col in gyro_cols):
        return df

    try:
        # 1. Gyro SVM
        temp_gyro_svm = np.sqrt(df['GYRO_X'] ** 2 + df['GYRO_Y'] ** 2 + df['GYRO_Z'] ** 2)

        # 2. Mask
        is_active = temp_gyro_svm > threshold

        if not is_active.any():
            print(f"      ⚠️ Warning: Data is mostly idle (Max Gyro < {threshold}).")
            return df

            # 3. Find Start/End indices
        start_idx = is_active.idxmax()
        end_idx = is_active[::-1].idxmax()

        # 4. Crop
        df_trimmed = df.loc[start_idx: end_idx].copy()
        df_trimmed.reset_index(drop=True, inplace=True)

        return df_trimmed

    except Exception as e:
        print(f"Error in adaptive trimming: {e}")
        return df


# ==========================================
# 🧮 2. PHYSICS CORE
# ==========================================
def calculate_features(df):
    t_sec = parse_time_column(df)
    features_df = pd.DataFrame()

    dt_series = pd.Series(t_sec).diff()
    avg_dt = dt_series[dt_series > 0].mean()
    if pd.isna(avg_dt) or avg_dt == 0:
        avg_dt = 0.02

    for s in ['ACC', 'GYRO']:
        cols = [f'{s}_X', f'{s}_Y', f'{s}_Z']

        if all(c in df.columns for c in cols):
            # SVM
            svm = (df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2) ** 0.5
            features_df[f'{s}_SVM'] = svm

            # Smooth SVM
            svm_smooth = svm.rolling(window=5, center=True).mean().bfill().ffill()

            # Jerk
            jerk = (svm_smooth.diff() / avg_dt).abs().fillna(0)
            features_df[f'{s}_SVM_Jerk'] = jerk

    return features_df


# ==========================================
# 🌎 3. GLOBAL STANDARD SCALING
# ==========================================
def build_global_standard():
    print("\n" + "=" * 60)
    print("   🏗️  BUILDING GLOBAL STANDARD (BASELINE: NORMAL WALK)")
    print(f"   📂 Root Source: {ROOT_DIR}")
    print("=" * 60)

    if not os.path.exists(ROOT_DIR):
        print(f"❌ Error: Path not found: {ROOT_DIR}")
        return

    stats_acc = {}
    file_count = 0

    subject_dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.startswith('S_')]
    print(f"🔎 Found {len(subject_dirs)} subjects.\n")

    for sub_name in subject_dirs:
        normal_path = os.path.join(ROOT_DIR, sub_name, 'test', 'Normal')
        if not os.path.exists(normal_path): continue

        csv_files = glob.glob(os.path.join(normal_path, "*.csv"))

        for fpath in csv_files:
            try:
                print(f"Reading: {sub_name} -> {os.path.basename(fpath)}", end='\r')
                df = pd.read_csv(fpath)

                # Process steps
                df = sort_data_by_time(df)
                df = trim_initial_data(df, cut_seconds=5.0)
                if df.empty: continue
                df = trim_idle_start_end(df, threshold=MOVEMENT_THRESHOLD)
                if df.empty: continue

                feat_df = calculate_features(df)

                for col in feat_df.columns:
                    data = feat_df[col].dropna()
                    if data.empty: continue

                    if col not in stats_acc:
                        stats_acc[col] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0, 'min': float('inf'), 'max': float('-inf')}

                    s = stats_acc[col]
                    s['n'] += len(data)
                    s['sum'] += data.sum()
                    s['sum_sq'] += (data ** 2).sum()
                    s['min'] = min(s['min'], data.min())
                    s['max'] = max(s['max'], data.max())

                file_count += 1
            except Exception as e:
                print(f"\nSkipping {os.path.basename(fpath)}: {e}")

    if file_count == 0:
        print("\n❌ ไม่พบข้อมูล Normal Walk เลย")
        return

    final_config = {}
    for col, s in stats_acc.items():
        if s['n'] > 0:
            mean = s['sum'] / s['n']
            variance = (s['sum_sq'] / s['n']) - (mean ** 2)
            std = variance ** 0.5 if variance > 0 else 1.0

            final_config[col] = {'mean': mean, 'std': std, 'min': s['min'], 'max': s['max']}

    with open(CONFIG_FILE, 'w') as f:
        json.dump(final_config, f, indent=4)

    print(f"\n\n✅ Processed {file_count} files.")
    print(f"✅ Created Baseline Ruler at: {CONFIG_FILE}")


def load_global_standard():
    global GLOBAL_STATS
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            GLOBAL_STATS = json.load(f)
        print(f"\nℹ️  Loaded Global Standard from {CONFIG_FILE}")
    else:
        GLOBAL_STATS = None
        print("\n⚠️  Local Scaling Mode")


def scale_zscore(series, col_name):
    if GLOBAL_STATS and col_name in GLOBAL_STATS:
        g_mean = GLOBAL_STATS[col_name]['mean']
        g_std = GLOBAL_STATS[col_name]['std']
        denom = g_std if g_std != 0 else 1.0
        return (series - g_mean) / denom
    else:
        std = series.std()
        return (series - series.mean()) / std if std != 0 else series.apply(lambda x: 0.0)


# ==========================================
# 📊 4. PLOTTING & EXPORT
# ==========================================
def plot_zscore_only(df, output_path, header_title, time_axis, metric_type):
    try:
        set_thai_font()
        suffix_map = {'SVM': '_SVM', 'Jerk': '_SVM_Jerk'}
        suffix = suffix_map.get(metric_type)
        if not suffix: return

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        mode_text = "Global Standard" if GLOBAL_STATS else "Local Scaling"
        fig.suptitle(f'{metric_type} Z-Score Analysis\n{mode_text} | (Trimmed)', fontsize=16, fontweight='bold')

        sensors = ['ACC', 'GYRO']
        colors = ['tab:blue', 'tab:orange']

        for i, sensor in enumerate(sensors):
            col_name = f'{sensor}{suffix}'
            if col_name in df.columns:
                z_data = scale_zscore(df[col_name], col_name)

                axs[i].plot(time_axis, z_data, color=colors[i], linewidth=1.2)
                axs[i].set_title(sensor, fontsize=14, fontweight='bold', loc='center')
                axs[i].set_ylabel('Z-Score', fontsize=10)
                axs[i].axhline(0, color='black', lw=1, linestyle='--')
                axs[i].grid(True, alpha=0.3)
                axs[i].set_xlabel("Time (s)", fontsize=12)
                axs[i].tick_params(labelbottom=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Plot Error: {e}")


def process_file_direct(input_path, output_data_dir, output_plot_dir, filename_no_ext, scenario_name):
    try:
        df = pd.read_csv(input_path)

        # 1. Sort Data
        df = sort_data_by_time(df)

        # 2. Fixed Trim (5s)
        df = trim_initial_data(df, cut_seconds=5.0)
        if df.empty: return False

        # 3. Adaptive Trim (>15.0 deg/s)
        df = trim_idle_start_end(df, threshold=MOVEMENT_THRESHOLD)
        if df.empty: return False

        # 4. Features
        feat_df = calculate_features(df)
        df_full = pd.concat([df, feat_df], axis=1)
        t_sec = parse_time_column(df_full)

        # 5. Scale & Export
        z_df = pd.DataFrame()
        for c in feat_df.columns:
            z_df[f"Z_{c}"] = scale_zscore(feat_df[c], c)

        raw_out = os.path.join(output_data_dir, f"{filename_no_ext}_RawFeatures.csv")
        df_full.to_csv(raw_out, index=False)

        z_out = os.path.join(output_data_dir, f"{filename_no_ext}_ZScores.csv")
        z_df.to_csv(z_out, index=False)

        for m in ['SVM', 'Jerk']:
            plot_path = os.path.join(output_plot_dir, f"{filename_no_ext}_{m}_ZScore.png")
            plot_zscore_only(df_full, plot_path, f"{scenario_name} : {filename_no_ext}", t_sec, m)

        return True
    except Exception as e:
        print(f" ❌ Error processing {os.path.basename(input_path)}: {e}")
        return False


# ==========================================
# 🖐️ 5. MANUAL SELECTION MODES
# ==========================================

# --- OPTION 3: Auto Output Path (Smart) ---
def process_manual_selection_auto_path():
    load_global_standard()
    print("\n" + "=" * 60)
    print("   🖐️  MODE 3: Select File -> Auto Output Path")
    print("=" * 60)

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()

    try:
        file_paths = filedialog.askopenfilenames(
            parent=root, title="เลือกไฟล์ CSV (บันทึกอัตโนมัติลง Folder โครงสร้าง)",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
    finally:
        root.destroy()

    if not file_paths: return

    print(f"\n📂 เลือก {len(file_paths)} ไฟล์ -> กำลังประมวลผล...\n")
    success_count = 0

    for i, fpath in enumerate(file_paths, 1):
        fname = os.path.basename(fpath)
        fname_no_ext = os.path.splitext(fname)[0]
        current_dir = os.path.dirname(fpath)
        scenario_name = os.path.basename(current_dir)
        parent_dir = os.path.dirname(current_dir)

        # Logic เดิม: เช็คโครงสร้างโฟลเดอร์เพื่อหาที่ลง
        if os.path.basename(parent_dir).lower() == 'test':
            subject_path = os.path.dirname(parent_dir)
            preprocess_root = os.path.join(subject_path, 'Complete_Preprocess')
            out_data = os.path.join(preprocess_root, 'Data', scenario_name)
            out_plot = os.path.join(preprocess_root, 'Plot_Graph', scenario_name)
            output_msg = f"Subject: {os.path.basename(subject_path)}"
        else:
            base_out = os.path.join(os.getcwd(), "Manual_Process_Output")
            out_data = os.path.join(base_out, "Data")
            out_plot = os.path.join(base_out, "Plots")
            output_msg = "Local Output"

        os.makedirs(out_data, exist_ok=True)
        os.makedirs(out_plot, exist_ok=True)

        print(f"   [{i}/{len(file_paths)}] {fname} -> {output_msg} ... ", end='', flush=True)

        if process_file_direct(fpath, out_data, out_plot, fname_no_ext, scenario_name):
            print("✅ OK")
            success_count += 1
        else:
            print("❌ Failed")

    print(f"\n🎉 Mode 3 Finished! ({success_count}/{len(file_paths)} files)")


# --- ✅ OPTION 4: Custom Output Path (New!) ---
def process_manual_save_custom():
    """
    ฟังก์ชันใหม่: เลือกไฟล์ -> เลือกโฟลเดอร์ปลายทางเอง -> ประมวลผล
    """
    load_global_standard()
    print("\n" + "=" * 60)
    print("   🖐️  MODE 4: Select File -> Custom Output Folder")
    print("=" * 60)

    # 1. Setup UI
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()

    try:
        # 2. เลือกไฟล์ Input
        file_paths = filedialog.askopenfilenames(
            parent=root,
            title="[ขั้นตอน 1/2] เลือกไฟล์ CSV ที่ต้องการประมวลผล",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_paths:
            print("❌ ไม่มีการเลือกไฟล์")
            return

        # 3. เลือกโฟลเดอร์ Output
        print("⏳ กรุณาเลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์...")
        target_dir = filedialog.askdirectory(
            parent=root,
            title="[ขั้นตอน 2/2] เลือกโฟลเดอร์ปลายทางเพื่อบันทึกไฟล์"
        )
        if not target_dir:
            print("❌ ไม่มีการเลือกโฟลเดอร์ปลายทาง")
            return

    finally:
        root.destroy()

    # 4. เตรียมโฟลเดอร์ย่อย Data และ Plots ในปลายทางที่เลือก
    out_data_root = os.path.join(target_dir, "Data")
    out_plot_root = os.path.join(target_dir, "Plots")
    os.makedirs(out_data_root, exist_ok=True)
    os.makedirs(out_plot_root, exist_ok=True)

    print(f"\n📂 Input: {len(file_paths)} files")
    print(f"📂 Output: {target_dir}")
    print("-" * 40)

    success_count = 0

    # 5. วนลูปประมวลผล
    for i, fpath in enumerate(file_paths, 1):
        fname = os.path.basename(fpath)
        fname_no_ext = os.path.splitext(fname)[0]

        # พยายามเดาชื่อท่าจากโฟลเดอร์แม่ของไฟล์ (เพื่อใช้แปะหัวกราฟ)
        # ถ้าไม่มีโครงสร้างก็ใช้ชื่อโฟลเดอร์ที่ไฟล์อยู่
        scenario_name = os.path.basename(os.path.dirname(fpath))

        print(f"   [{i}/{len(file_paths)}] {fname} ... ", end='', flush=True)

        # ส่ง Path ปลายทางที่เลือกไว้เข้าไปตรงๆ
        if process_file_direct(fpath, out_data_root, out_plot_root, fname_no_ext, scenario_name):
            print("✅ OK")
            success_count += 1
        else:
            print("❌ Failed")

    print(f"\n🎉 Mode 4 Finished! ({success_count}/{len(file_paths)} files)")
    print(f"👉 ตรวจสอบไฟล์ได้ที่: {target_dir}")


# ==========================================
# 🚀 6. MAIN AUTOMATION LOGIC
# ==========================================
def process_batch_directory():
    load_global_standard()
    if not os.path.exists(ROOT_DIR):
        print(f"❌ ไม่พบโฟลเดอร์หลัก: {ROOT_DIR}")
        return

    subject_dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.startswith('S_')]
    print(f"\n📂 พบ Subject ทั้งหมด: {len(subject_dirs)} คน\n")

    scenario_list = ['Dunk', 'Left', 'Normal', 'Right', 'Slow', 'Stun']

    for i, sub_name in enumerate(subject_dirs, 1):
        sub_path = os.path.join(ROOT_DIR, sub_name)
        test_path = os.path.join(sub_path, 'test')
        print(f"👤 [{i}/{len(subject_dirs)}] Processing: {sub_name} ... ", end='', flush=True)

        if not os.path.exists(test_path):
            print("❌ (No test folder)")
            continue

        preprocess_root = os.path.join(sub_path, 'Complete_Preprocess')
        data_root = os.path.join(preprocess_root, 'Data')
        plot_root = os.path.join(preprocess_root, 'Plot_Graph')
        total_files = 0
        success_files = 0
        print("")

        for scenario in scenario_list:
            input_scenario_path = os.path.join(test_path, scenario)
            if not os.path.exists(input_scenario_path): continue

            out_data_scen = os.path.join(data_root, scenario)
            out_plot_scen = os.path.join(plot_root, scenario)
            os.makedirs(out_data_scen, exist_ok=True)
            os.makedirs(out_plot_scen, exist_ok=True)

            csv_files = glob.glob(os.path.join(input_scenario_path, "*.csv"))
            if not csv_files: continue

            print(f"   🔹 {scenario}: พบ {len(csv_files)} ไฟล์...")

            for fpath in csv_files:
                fname = os.path.basename(fpath)
                fname_no_ext = os.path.splitext(fname)[0]
                if process_file_direct(fpath, out_data_scen, out_plot_scen, fname_no_ext, scenario):
                    success_files += 1
                total_files += 1

        print(f"   ✅ {sub_name}: สำเร็จ {success_files}/{total_files} ไฟล์")
        print("-" * 50)

    print("\n🎉 ALL DONE! 🎉")


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 60)
        print("   GAIT PRE-PROCESSOR (HYBRID + ADAPTIVE TRIM)")
        print(f"   Auto Target: {ROOT_DIR}")
        print("=" * 60)
        print(" [1] 🏗️  Build Global Standard (Only 'Normal' as Baseline)")
        print(" [2] 🚀  Process Data (Auto Batch from Root Dir)")
        print(" [3] 🖐️  Select File -> Auto Save (Standard Structure)")
        print(" [4] 💾  Select File -> Custom Save Location (Choose Folder)")
        print(" [0] Exit")

        c = input("Select: ").strip()
        if c == '1':
            build_global_standard()
        elif c == '2':
            process_batch_directory()
        elif c == '3':
            process_manual_selection_auto_path()
        elif c == '4':
            process_manual_save_custom()
        elif c == '0':
            break
