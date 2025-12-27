import pandas as pd
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import json
import numpy as np


# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
def set_thai_font():
    """ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ"""
    if os.name == 'nt':
        plt.rcParams['font.family'] = 'Tahoma'
    else:
        plt.rcParams['font.family'] = 'Ayuthaya'
    plt.rcParams['axes.unicode_minus'] = False


CONFIG_FILE = 'global_standard_config.json'
GLOBAL_STATS = None


# ==========================================
# 🛠️ HELPER: TRIM DATA
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


def trim_initial_data(df, cut_seconds=5.0):
    """ตัดข้อมูล 5 วินาทีแรกทิ้ง และ Reset Index"""
    try:
        temp_time = parse_time_column(df)
        df_trimmed = df[temp_time >= cut_seconds].copy()
        df_trimmed.reset_index(drop=True, inplace=True)

        if df_trimmed.empty:
            print(f"⚠️ Warning: Data is empty after trimming first {cut_seconds}s")
            return df

        return df_trimmed
    except Exception as e:
        print(f"Error trimming data: {e}")
        return df


# ==========================================
# 🧮 2. PHYSICS CORE (SVM + JERK)
# ==========================================
def calculate_features(df):
    t_sec = parse_time_column(df)
    features_df = pd.DataFrame()

    dt_series = pd.Series(t_sec).diff()
    avg_dt = dt_series.mean() if dt_series.mean() > 0 else 0.02

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
# 🌎 3. GLOBAL STANDARD SCALING (Z-SCORE)
# ==========================================
def build_global_standard():
    print("\n" + "=" * 60)
    print("   🏗️  BUILDING GLOBAL STANDARD (ACC & GYRO)")
    print("   ✂️  Excluding first 5 seconds")
    print("=" * 60)

    root = tk.Tk();
    root.withdraw();
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="เลือกโฟลเดอร์ข้อมูล Training Set")
    if not folder_path: return

    stats_acc = {}
    files = []

    for r, d, f in os.walk(folder_path):
        for file in f:
            if file.lower().endswith('.csv'): files.append(os.path.join(r, file))

    print(f"\nพบไฟล์ทั้งหมด: {len(files)} ไฟล์ ... กำลังคำนวณ")

    for idx, fpath in enumerate(files, 1):
        try:
            print(f"[{idx}/{len(files)}] Processing: {os.path.basename(fpath)}", end='\r')
            df = pd.read_csv(fpath)

            df = trim_initial_data(df, cut_seconds=5.0)
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
        except Exception as e:
            print(f"\nError file {fpath}: {e}")

    final_config = {}
    for col, s in stats_acc.items():
        if s['n'] > 0:
            mean = s['sum'] / s['n']
            variance = (s['sum_sq'] / s['n']) - (mean ** 2)
            std = variance ** 0.5 if variance > 0 else 1.0

            final_config[col] = {
                'mean': mean,
                'std': std,
                'min': s['min'],
                'max': s['max']
            }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(final_config, f, indent=4)
    print(f"\n\n✅ บันทึกมาตรฐานกลางเรียบร้อยที่: {CONFIG_FILE}")


def load_global_standard():
    global GLOBAL_STATS
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            GLOBAL_STATS = json.load(f)
        print(f"\nℹ️  Loaded Global Standard from {CONFIG_FILE}")
    else:
        GLOBAL_STATS = None
        print("\n⚠️  ไม่พบไฟล์มาตรฐานกลาง")


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
# 📊 4. PLOTTING & EXPORT (แก้ไข Title ตามที่ขอ)
# ==========================================
def plot_zscore_only(df, filename_base, header_title, time_axis, metric_type):
    try:
        set_thai_font()
        suffix_map = {'SVM': '_SVM', 'Jerk': '_SVM_Jerk'}
        suffix = suffix_map.get(metric_type)
        if not suffix: return None

        # สร้าง Canvas
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        mode_text = "Global Standard" if GLOBAL_STATS else "Local Scaling"
        # Main Title (หัวกระดาษใหญ่)
        fig.suptitle(f'{metric_type} Z-Score Analysis\n{mode_text} | (First 5s trimmed)', fontsize=16,
                     fontweight='bold')

        sensors = ['ACC', 'GYRO']
        colors = ['tab:blue', 'tab:orange']

        for i, sensor in enumerate(sensors):
            col_name = f'{sensor}{suffix}'
            if col_name in df.columns:
                z_data = scale_zscore(df[col_name], col_name)

                # Plot
                axs[i].plot(time_axis, z_data, color=colors[i], linewidth=1.2)

                # ✅ แก้ไขตรงนี้: ย้ายชื่อ Sensor ไปเป็น Title ด้านบนกราฟ
                axs[i].set_title(sensor, fontsize=14, fontweight='bold', loc='center')

                # ปรับแกน Y ให้เหลือแค่คำว่า Z-Score เฉยๆ (จะได้ไม่รก)
                axs[i].set_ylabel('Z-Score', fontsize=10)

                # เส้นแบ่ง 0
                axs[i].axhline(0, color='black', lw=1, linestyle='--')
                axs[i].grid(True, alpha=0.3)

                # แกน X
                axs[i].set_xlabel("Time (s)", fontsize=12)
                axs[i].tick_params(labelbottom=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        img_name = f"{filename_base}_{metric_type}_ZScore.png"
        plt.savefig(img_name)
        plt.close()
        return img_name
    except Exception as e:
        print(f"Plot Error: {e}")
        return None


def process_single_file(path, scenario, name, file_id):
    try:
        df = pd.read_csv(path)

        # 1. ตัดข้อมูล 5 วินาทีแรก
        df = trim_initial_data(df, cut_seconds=5.0)

        if df.empty:
            print(f"Skipping {os.path.basename(path)} (Too short)")
            return []

        # 2. คำนวณ Features
        feat_df = calculate_features(df)
        df_full = pd.concat([df, feat_df], axis=1)  # รวม Raw + Features
        t_sec = parse_time_column(df_full)

        # 3. เตรียมข้อมูล Z-Score
        z_df = pd.DataFrame()
        for c in feat_df.columns:
            z_df[f"Z_{c}"] = scale_zscore(feat_df[c], c)

        # 4. Save as CSV
        base_name = f"{scenario}_{name}_{file_id}"

        # Save Raw + Features
        csv_raw_name = f"{base_name}_RawFeatures.csv"
        df_full.to_csv(csv_raw_name, index=False)

        # Save Z-Scores Only (สำหรับเข้า AI)
        csv_z_name = f"{base_name}_ZScores.csv"
        z_df.to_csv(csv_z_name, index=False)

        generated_files = [csv_raw_name, csv_z_name]

        # 5. Plot Graph
        for m in ['SVM', 'Jerk']:
            img = plot_zscore_only(df_full, base_name, f"{scenario} : {name}", t_sec, m)
            if img: generated_files.append(img)

        return generated_files
    except Exception as e:
        print(f"Error {os.path.basename(path)}: {e}")
        return []


# ==========================================
# 🚀 5. MAIN EXECUTION FLOW
# ==========================================
def parse_filename_info(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) < 2: return "Unknown", "Unknown", base
    try:
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    except:
        return "Error", "Error", base


def analyze_auto_mode():
    load_global_standard()

    print("\n👉 เลือกไฟล์ CSV ข้อมูลใหม่ที่ต้องการ Process")
    root = tk.Tk();
    root.withdraw();
    root.attributes('-topmost', True)
    file_paths = filedialog.askopenfilenames(title="เลือกไฟล์ CSV", filetypes=[("CSV", "*.csv")])

    if not file_paths: return

    grouped_data = {}
    for path in file_paths:
        scenario, person_name, f_id = parse_filename_info(os.path.basename(path))
        if person_name not in grouped_data: grouped_data[person_name] = []
        grouped_data[person_name].append({'path': path, 'scenario': scenario, 'id': f_id})

    for person, data_list in grouped_data.items():
        print(f"\n👤 Processing: {person}")
        person_files = []
        for item in data_list:
            print(f"   Running: {item['scenario']} (ID: {item['id']})...")
            out = process_single_file(item['path'], item['scenario'], person, item['id'])
            person_files.extend(out)

        print(f"   👉 เลือกที่เก็บไฟล์ผลลัพธ์ (.csv & .png) ของ {person}...")
        dest_dir = filedialog.askdirectory(title=f"Save Output for {person}")
        if dest_dir:
            for f in person_files:
                try:
                    shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
                except:
                    pass
            print(f"✅ Saved to {dest_dir}")


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("   GAIT PRE-PROCESSOR (Header Fix)")
        print("   (Auto-Trim First 5s | SVM+Jerk)")
        print("=" * 50)
        print(" [1] 🏗️  Build Global Standard")
        print(" [2] 🚀  Process New Data")
        print(" [0] Exit")

        c = input("Select: ").strip()
        if c == '1':
            build_global_standard()
        elif c == '2':
            analyze_auto_mode()
        elif c == '0':
            break