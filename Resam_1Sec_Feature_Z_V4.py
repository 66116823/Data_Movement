import pandas as pd
import os
import glob
import numpy as np
import warnings
from scipy.stats import entropy

# ปิด Warning
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. ตั้งค่า (CONFIGURATION)
# ==========================================
ROOT_DIR = r"E:\Data Movement\CollectDATA\Subject"
OUTPUT_DIR = r"E:\Data Movement\CollectDATA\Master_Data"
# ✅ เปลี่ยนชื่อไฟล์ให้รู้ว่ามี MinMaxRange
OUTPUT_FILENAME = "Resample_1Sec_Full_Features_Plus_MinMaxRange.csv"

MOVEMENT_THRESHOLD = 15.0
BASELINE_SCENARIO = 'Normal'

# DATABASE (คงเดิม)
SUBJECT_DB = {
    'S_ALIF': {'Gender': 0, 'BMI': 22.13},
    'S_ARPO': {'Gender': 1, 'BMI': 34.37},
    'S_BUNCHA': {'Gender': 0, 'BMI': 19.03},
    'S_CHO': {'Gender': 0, 'BMI': 29.94},
    'S_DAVE': {'Gender': 0, 'BMI': 18.36},
    'S_FANG': {'Gender': 1, 'BMI': 20.05},
    'S_FEE': {'Gender': 0, 'BMI': 23.34},
    'S_GUMPUN': {'Gender': 0, 'BMI': 19.47},
    'S_GUTAR': {'Gender': 1, 'BMI': 18.60},
    'S_HANIS': {'Gender': 1, 'BMI': 18.52},
    'S_KEEN': {'Gender': 0, 'BMI': 30.37},
    'S_MIND': {'Gender': 1, 'BMI': 18.36},
    'S_MINNY': {'Gender': 1, 'BMI': 16.49},
    'S_MINT': {'Gender': 1, 'BMI': 23.49},
    'S_NECK': {'Gender': 0, 'BMI': 17.31},
    'S_OPOR': {'Gender': 1, 'BMI': 16.23},
    'S_PHET': {'Gender': 0, 'BMI': 16.00},
    'S_PON': {'Gender': 0, 'BMI': 31.61},
    'S_PUIFAII': {'Gender': 1, 'BMI': 25.97},
    'S_RISTA': {'Gender': 1, 'BMI': 15.81},
    'S_SODA': {'Gender': 1, 'BMI': 28.04},
    'S_SUN': {'Gender': 0, 'BMI': 29.05},
    'S_TEW': {'Gender': 0, 'BMI': 15.79},
    'S_WAN': {'Gender': 1, 'BMI': 24.17},
    'S_WOON': {'Gender': 1, 'BMI': 25.73}
}

LABEL_MAPPING = {
    'Normal': 0, 'Slow': 0,
    'Left': 1, 'Right': 1,
    'Dunk': 2, 'Stun': 2
}

# -----------------------------------------------------------
# 🔥 1. กำหนดลำดับคอลัมน์
# -----------------------------------------------------------
RAW_SENSORS = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
ENGINEERED_FEATURES = ['ACC_SVM', 'ACC_Jerk', 'ACC_Combined', 'GYRO_SVM', 'GYRO_Jerk', 'GYRO_Combined']
Z_FEATURES = [f"{feat}_Z" for feat in ENGINEERED_FEATURES]

# ✅ แก้ไขจุดที่ 1: เพิ่ม 'Min', 'Max', 'Range' ในรายชื่อ Basic Stats
BASIC_STATS = ['Mean', 'Median', 'SD', 'Min', 'Max', 'Range', 'Skew', 'Kurt', 'P75', 'P90', 'P95']

ADVANCED_STATS = [
    'FFT_DomFreq', 'FFT_Energy', 'FFT_Entropy',
    'Reg_A', 'Reg_B', 'Reg_C', 'Reg_Error'
]

ADVANCED_TARGET_COLS = RAW_SENSORS + ENGINEERED_FEATURES
ALL_PROCESS_COLS = RAW_SENSORS + ENGINEERED_FEATURES + Z_FEATURES

# สร้างชื่อคอลัมน์ (Loop นี้จะสร้างชื่อ _Min, _Max, _Range ให้เองอัตโนมัติจาก BASIC_STATS)
EXTENDED_STATS_COLS = []
for col in ALL_PROCESS_COLS:
    for stat in BASIC_STATS:
        EXTENDED_STATS_COLS.append(f"{col}_{stat}")  # เช่น ACC_X_Min, ACC_X_Range

    if col in ADVANCED_TARGET_COLS:
        for stat in ADVANCED_STATS:
            EXTENDED_STATS_COLS.append(f"{col}_{stat}")

FINAL_COLUMNS_ORDER = [
                          'Subject_ID', 'Date', 'Time', 'Name', 'Filename', 'Scenario', 'Label',
                          'Gender', 'BMI', 'BMI_Label'
                      ] + RAW_SENSORS + EXTENDED_STATS_COLS


# ==========================================
# 🛠️ ฟังก์ชัน Helper (คงเดิม)
# ==========================================
def calculate_bmi_label(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25.0:
        return 1
    else:
        return 2


def convert_time_string_to_seconds(t_str):
    try:
        t_str = str(t_str).strip()
        parts = t_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            return float(t_str)
    except:
        return np.nan


def remove_startup_noise(df, time_col='Time', cutoff_sec=5):
    if df.empty or time_col not in df.columns: return df
    temp_seconds = df[time_col].apply(convert_time_string_to_seconds)
    if temp_seconds.isna().all(): return df
    start_time = temp_seconds.min()
    mask = (temp_seconds - start_time) >= cutoff_sec
    filtered_df = df[mask].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def trim_idle_start_end(df, threshold=MOVEMENT_THRESHOLD):
    gyro_cols = ['GYRO_X', 'GYRO_Y', 'GYRO_Z']
    if not all(col in df.columns for col in gyro_cols): return df
    try:
        gx, gy, gz = df['GYRO_X'].values, df['GYRO_Y'].values, df['GYRO_Z'].values
        svm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        is_active = svm > threshold
        if not np.any(is_active): return df
        start_idx = np.argmax(is_active)
        end_idx = len(is_active) - np.argmax(is_active[::-1]) - 1
        return df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    except:
        return df


def calculate_raw_features(df):
    time_sec = df['Time'].apply(convert_time_string_to_seconds)
    delta_time = time_sec.diff().fillna(0.02)
    delta_time.replace(0, 0.02, inplace=True)

    df['ACC_SVM'] = np.sqrt(df['ACC_X'] ** 2 + df['ACC_Y'] ** 2 + df['ACC_Z'] ** 2)
    delta_acc_svm = df['ACC_SVM'].diff().fillna(0)
    acc_jerk = delta_acc_svm / delta_time
    acc_jerk.replace([np.inf, -np.inf], 0, inplace=True)
    df['ACC_Jerk'] = acc_jerk
    df['ACC_Combined'] = df['ACC_SVM'] * df['ACC_Jerk']

    df['GYRO_SVM'] = np.sqrt(df['GYRO_X'] ** 2 + df['GYRO_Y'] ** 2 + df['GYRO_Z'] ** 2)
    delta_gyro_svm = df['GYRO_SVM'].diff().fillna(0)
    gyro_jerk = delta_gyro_svm / delta_time
    gyro_jerk.replace([np.inf, -np.inf], 0, inplace=True)
    df['GYRO_Jerk'] = gyro_jerk
    df['GYRO_Combined'] = df['GYRO_SVM'] * df['GYRO_Jerk']
    return df


# --- Advanced Math Helpers (คงเดิม) ---
def get_fft_values(series):
    try:
        fft_vals = np.fft.rfft(series)
        fft_mag = np.abs(fft_vals)
        if len(fft_mag) > 1:
            dom_freq = np.argmax(fft_mag[1:]) + 1
        else:
            dom_freq = 0
        energy = np.sum(fft_mag ** 2) / len(series)
        psd = fft_mag ** 2
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            ent = entropy(psd_norm)
        else:
            ent = 0
        return dom_freq, energy, ent
    except:
        return 0, 0, 0


def get_poly_values(series):
    try:
        y = series.values
        x = np.arange(len(y))
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            p = np.poly1d(coeffs)
            y_pred = p(x)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            return a, b, c, rmse
        else:
            return 0, 0, 0, 0
    except:
        return 0, 0, 0, 0


# =========================================================================
# 🔥🔥🔥 MAIN PROCESSING FUNCTION (UPDATED) 🔥🔥🔥
# =========================================================================
def process_file_to_1sec_features(df, baseline_stats=None):
    try:
        # Step 0: Calc Features & Z-Score (คงเดิม)
        df = calculate_raw_features(df)
        if baseline_stats:
            for feature in ENGINEERED_FEATURES:
                z_col_name = f"{feature}_Z"
                mean_val = baseline_stats.get(f"{feature}_Mean", 0)
                std_val = baseline_stats.get(f"{feature}_SD", 1)
                if std_val == 0: std_val = 1e-6
                df[z_col_name] = (df[feature] - mean_val) / std_val
        else:
            for feature in ENGINEERED_FEATURES:
                df[f"{feature}_Z"] = 0

        # Step 1: Time Rounding (คงเดิม)
        df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True,
                                         errors='coerce')
        if df['Timestamp'].isna().all():
            start_t = pd.Timestamp.now()
            df['Timestamp'] = [start_t + pd.Timedelta(milliseconds=20 * i) for i in range(len(df))]

        df = df.dropna(subset=['Timestamp'])
        df['Rounded_Time'] = df['Timestamp'].dt.round('1s')
        grouped = df.groupby('Rounded_Time')

        cols_to_exclude = RAW_SENSORS + ENGINEERED_FEATURES + Z_FEATURES + ['Rounded_Time', 'Timestamp', 'Date', 'Time']
        meta_cols = [c for c in df.columns if c not in cols_to_exclude and c in df.columns]
        df_meta = grouped[meta_cols].first()

        # -----------------------------------------------------
        # 🟣 Step 3: คำนวณสถิติ (UPDATED)
        # -----------------------------------------------------
        ALL_TARGET_COLS = RAW_SENSORS + ENGINEERED_FEATURES + Z_FEATURES

        def safe_kurtosis(x):
            return x.kurt()

        def p75(x):
            return x.quantile(0.75)

        def p90(x):
            return x.quantile(0.90)

        def p95(x):
            return x.quantile(0.95)

        # ✅ แก้ไขจุดที่ 2: เพิ่ม 'min' และ 'max' ลงไปในรายการคำสั่ง aggregation
        # (เพื่อให้ Pandas หาค่าต่ำสุด/สูงสุดให้เราเลยตอน Group)
        basic_aggs = ['mean', 'median', 'std', 'min', 'max', 'skew', safe_kurtosis, p75, p90, p95]

        df_stats = grouped[ALL_TARGET_COLS].agg(basic_aggs)

        # -----------------------------------------------------
        # 🔴 Step 4: จัดเรียงและคำนวณ Features
        # -----------------------------------------------------
        df_main_raw = df_stats.loc[:, (RAW_SENSORS, 'mean')].copy()
        df_main_raw.columns = RAW_SENSORS

        extended_data = {}

        for col in ALL_TARGET_COLS:
            # 4.1 Basic Stats (ดึงจากที่คำนวณแล้ว)
            extended_data[f"{col}_Mean"] = df_stats[(col, 'mean')]
            extended_data[f"{col}_Median"] = df_stats[(col, 'median')]
            extended_data[f"{col}_SD"] = df_stats[(col, 'std')]

            # ✅ แก้ไขจุดที่ 3: ดึงค่า Min/Max และคำนวณ Range
            val_min = df_stats[(col, 'min')]
            val_max = df_stats[(col, 'max')]

            extended_data[f"{col}_Min"] = val_min
            extended_data[f"{col}_Max"] = val_max
            extended_data[f"{col}_Range"] = val_max - val_min  # คำนวณ Range (ผลต่าง) ตรงนี้

            extended_data[f"{col}_Skew"] = df_stats[(col, 'skew')]
            try:
                extended_data[f"{col}_Kurt"] = df_stats[(col, 'safe_kurtosis')]
            except:
                extended_data[f"{col}_Kurt"] = df_stats[(col, df_stats.columns.levels[1][-1])]
            extended_data[f"{col}_P75"] = df_stats[(col, 'p75')]
            extended_data[f"{col}_P90"] = df_stats[(col, 'p90')]
            extended_data[f"{col}_P95"] = df_stats[(col, 'p95')]

            # 4.2 Advanced Stats (FFT/Reg) - คงเดิม
            if col in ADVANCED_TARGET_COLS:
                raw_series_list = grouped[col]

                # FFT
                fft_res = raw_series_list.apply(get_fft_values)
                extended_data[f"{col}_FFT_DomFreq"] = fft_res.apply(lambda x: x[0])
                extended_data[f"{col}_FFT_Energy"] = fft_res.apply(lambda x: x[1])
                extended_data[f"{col}_FFT_Entropy"] = fft_res.apply(lambda x: x[2])

                # Regression
                poly_res = raw_series_list.apply(get_poly_values)
                extended_data[f"{col}_Reg_A"] = poly_res.apply(lambda x: x[0])
                extended_data[f"{col}_Reg_B"] = poly_res.apply(lambda x: x[1])
                extended_data[f"{col}_Reg_C"] = poly_res.apply(lambda x: x[2])
                extended_data[f"{col}_Reg_Error"] = poly_res.apply(lambda x: x[3])

        df_extended = pd.DataFrame(extended_data, index=df_stats.index)

        # Step 5: Merge (คงเดิม)
        df_1sec = pd.concat([df_meta, df_main_raw, df_extended], axis=1)
        df_1sec['Date'] = df_1sec.index.date
        df_1sec['Time'] = df_1sec.index.time
        df_1sec.reset_index(drop=True, inplace=True)

        return df_1sec

    except Exception as e:
        print(f"      ⚠️ Feature Calculation Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ==========================================
# 🚀 2. เริ่มกระบวนการประมวลผล (Main Loop)
# ==========================================
# (ส่วนนี้เหมือนเดิม 100%)
all_data_frames = []
subject_folders = [d for d in os.listdir(ROOT_DIR) if d.startswith('S_') and os.path.isdir(os.path.join(ROOT_DIR, d))]
subject_folders.sort()

print(f"🔎 พบ Subject ทั้งหมด {len(subject_folders)} โฟลเดอร์ (Mode: Min/Max/Range)")

for pk_id, sub_name in enumerate(subject_folders, start=1):
    if sub_name not in SUBJECT_DB: continue
    info = SUBJECT_DB[sub_name]
    print(f"\n✅ [{pk_id}] Processing: {sub_name}")

    baseline_stats = None
    normal_folder = os.path.join(ROOT_DIR, sub_name, 'test', BASELINE_SCENARIO)

    if os.path.exists(normal_folder):
        print(f"   ... Calibrating baseline from '{BASELINE_SCENARIO}'")
        normal_files = glob.glob(os.path.join(normal_folder, "*.csv"))
        temp_dfs = []
        for f in normal_files:
            try:
                tmp = pd.read_csv(f)
                tmp = remove_startup_noise(tmp)
                tmp = trim_idle_start_end(tmp)
                if not tmp.empty:
                    temp_dfs.append(tmp)
            except:
                pass

        if temp_dfs:
            df_normal_all = pd.concat(temp_dfs, ignore_index=True)
            df_normal_all = calculate_raw_features(df_normal_all)
            baseline_stats = {}
            for feature in ENGINEERED_FEATURES:
                baseline_stats[f"{feature}_Mean"] = df_normal_all[feature].mean()
                baseline_stats[f"{feature}_SD"] = df_normal_all[feature].std()
            print(f"       -> Baseline Calibrated ({len(baseline_stats) // 2} Features)")
        else:
            print(f"       ⚠️ Warning: ข้อมูล {BASELINE_SCENARIO} ว่างเปล่า")
    else:
        print(f"       ⚠️ Warning: ไม่พบโฟลเดอร์ {BASELINE_SCENARIO}")

    for scenario_name, label_code in LABEL_MAPPING.items():
        folder_path = os.path.join(ROOT_DIR, sub_name, 'test', scenario_name)
        if not os.path.exists(folder_path): continue

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                filename_only = os.path.basename(file_path)
                df = remove_startup_noise(df)
                if df.empty: continue
                df = trim_idle_start_end(df)
                if df.empty: continue

                if 'Date' not in df.columns: df['Date'] = "01/01/2025"
                df['Subject_ID'] = pk_id
                df['Name'] = sub_name
                df['Filename'] = filename_only
                df['Scenario'] = scenario_name
                df['Gender'] = info['Gender']
                df['BMI'] = info['BMI']
                df['BMI_Label'] = calculate_bmi_label(info['BMI'])
                df['Label'] = label_code

                df = process_file_to_1sec_features(df, baseline_stats)

                if df.empty: continue
                df = df.reindex(columns=FINAL_COLUMNS_ORDER)
                all_data_frames.append(df)

            except Exception as e:
                print(f"      ❌ Error reading file {os.path.basename(file_path)}: {e}")

if all_data_frames:
    print("\n⏳ Merging dataframes...")
    master_df = pd.concat(all_data_frames, ignore_index=True)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    full_save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    master_df.to_csv(full_save_path, index=False)

    print("=" * 60)
    print(f"✅ SUCCESS! บันทึกไฟล์เรียบร้อยที่:")
    print(f"👉 {full_save_path}")
    print(f"📊 Total Rows: {len(master_df):,}")
    print("-" * 30)

    # เช็คคอลัมน์ใหม่
    print("ตัวอย่างคอลัมน์ใหม่ (Min/Max/Range):")
    cols_check = ['ACC_SVM_Min', 'ACC_SVM_Max', 'ACC_SVM_Range']
    if all(c in master_df.columns for c in cols_check):
        print(master_df[cols_check].head(3))
    print("=" * 60)
else:
    print("❌ ไม่พบข้อมูล")