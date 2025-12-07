import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import tkinter as tk
from tkinter import filedialog

# ตั้งค่าฟอนต์
plt.rcParams['font.family'] = 'Tahoma' if os.name == 'nt' else 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def calculate_svm(df, prefix):
    cols = [f'{prefix}_X', f'{prefix}_Y', f'{prefix}_Z']
    if all(col in df.columns for col in cols):
        return np.sqrt(df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2)
    return None


def detect_activity_phases(df, threshold=15.0):
    """
    แยกช่วง Rest vs Move โดยใช้ Gyroscope SVM
    ถ้าค่า Gyro SVM มากกว่า threshold (deg/s) ถือว่ามีการเคลื่อนไหว
    """
    # คำนวณ Gyro SVM
    gyro_svm = calculate_svm(df, 'GYRO')

    # Smooth ข้อมูลลด Noise
    gyro_smooth = gyro_svm.rolling(window=10, center=True).mean().fillna(0)

    # สร้าง Label: 0 = Rest, 1 = Move
    # ใช้ hysteresis เล็กน้อย (ขยายช่วง move ให้ครอบคลุม)
    is_moving = gyro_smooth > threshold
    labels = is_moving.astype(int)

    # Clean label: ถ้ามีการขยับสั้นๆ น้อยกว่า 0.5 วินาที ให้ตัดทิ้ง (Noise)
    # (ในที่นี้ทำแบบง่ายคือใช้ rolling max เพื่อเชื่อมช่องว่าง)
    labels = labels.rolling(window=20, center=True, min_periods=1).max()

    return labels, gyro_smooth


def plot_human_readable_patterns(file_path):
    filename = os.path.basename(file_path)
    print(f"กำลังวิเคราะห์: {filename} ...")

    df = pd.read_csv(file_path)

    # 1. เตรียมข้อมูล
    labels, gyro_signal = detect_activity_phases(df)
    df['Label'] = labels
    status_colors = {0: '#2ecc71', 1: '#e74c3c'}  # เขียว=นิ่ง, แดง=ขยับ
    status_names = {0: 'Rest (นิ่ง)', 1: 'Move (ขยับ)'}

    colors_mapped = df['Label'].map(status_colors)

    # สร้าง Canvas ใหญ่
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Human-Readable Pattern Analysis : {filename}", fontsize=16, fontweight='bold')

    # --- Plot 1: Time Series (แบ่งสีตามช่วง) ---
    ax1 = fig.add_subplot(2, 3, (1, 3))  # แถวบนยาวตลอดแนว
    acc_svm = calculate_svm(df, 'ACC')
    time_idx = np.arange(len(df))

    ax1.plot(time_idx, acc_svm, color='gray', alpha=0.5, label='Signal (ACC SVM)')
    ax1.scatter(time_idx, acc_svm, c=colors_mapped, s=5, alpha=0.8)

    # สร้าง Legend ปลอมๆ เพื่อให้อ่านง่าย
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Resting (ยืนนิ่ง)'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Movement (เดิน/ขยับ)')]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.set_title("1. Timeline: แยกแยะช่วงเวลานิ่ง vs ขยับ (Timeline Segmentation)", fontsize=12)
    ax1.set_ylabel("Intensity (g)")
    ax1.margins(x=0)

    # เตรียมข้อมูลสำหรับ PCA/t-SNE
    features = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    X = df[features].dropna()
    X_std = StandardScaler().fit_transform(X)
    y_labels = df['Label'].loc[X.index]
    c_map_points = y_labels.map(status_colors)

    # --- Plot 2: 3D Trajectory (ACC X, Y, Z) ---
    ax2 = fig.add_subplot(2, 3, 4, projection='3d')
    ax2.scatter(df['ACC_X'], df['ACC_Y'], df['ACC_Z'], c=colors_mapped, s=5, alpha=0.6)
    ax2.set_xlabel('ACC X (Left-Right)')
    ax2.set_ylabel('ACC Y (Up-Down)')
    ax2.set_zlabel('ACC Z (Forward-Back)')
    ax2.set_title("2. Physical Space (3D Acceleration)", fontsize=12)

    # --- Plot 3: PCA (2D Pattern) ---
    ax3 = fig.add_subplot(2, 3, 5)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Plot แยกกลุ่ม
    for label_id in [0, 1]:
        mask = y_labels == label_id
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=status_colors[label_id], label=status_names[label_id],
                    s=10, alpha=0.6)
    ax3.set_title(f"3. PCA Pattern (Var: {np.sum(pca.explained_variance_ratio_):.1%})", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: t-SNE (Cluster Separation) ---
    ax4 = fig.add_subplot(2, 3, 6)
    try:
        # ลดจำนวนข้อมูลถ้าเยอะเกินไปเพื่อให้คำนวณเร็ว
        if len(X_std) > 2000:
            idx = np.random.choice(len(X_std), 2000, replace=False)
            X_subset = X_std[idx]
            c_subset = c_map_points.iloc[idx]
        else:
            X_subset = X_std
            c_subset = c_map_points

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_subset)

        ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_subset, s=10, alpha=0.7)
        ax4.set_title("4. t-SNE Clusters (ความชัดเจนของกลุ่มท่าทาง)", fontsize=12)
        ax4.grid(True, alpha=0.3)
    except Exception as e:
        ax4.text(0.5, 0.5, f"Error: {str(e)}", ha='center')

    plt.tight_layout()
    save_path = file_path.replace('.csv', '_PatternAnalysis.png')
    plt.savefig(save_path)
    plt.close()
    print(f"   --> บันทึกภาพเรียบร้อย: {os.path.basename(save_path)}")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("เลือกไฟล์ CSV ของคุณ (เลือกได้ทั้ง 5 ไฟล์พร้อมกัน)...")
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])

    if file_paths:
        for fp in file_paths:
            plot_human_readable_patterns(fp)
        print("\nเสร็จสิ้น! เปิดดูรูปภาพในโฟลเดอร์ได้เลยครับ")
    else:
        print("ไม่ได้เลือกไฟล์")
