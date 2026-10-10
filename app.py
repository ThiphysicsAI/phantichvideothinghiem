import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os

# Hàm xử lý giữ nguyên thuật toán
def phan_tich_video(video_path, chieu_cao_thuc_te):
    if video_path is None:
        return None, "Vui lòng tải lên một video!"
    if chieu_cao_thuc_te <= 0:
        return None, "Chiều cao thực tế phải lớn hơn 0!"

    cap = cv2.VideoCapture(video_path)
    toa_do_y, thoi_gian = [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, None, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                thoi_gian.append(frame_count / fps)
                toa_do_y.append(y)
        frame_count += 1
    cap.release()

    if len(toa_do_y) < 3:
        return None, "AI không tìm thấy vật thể rơi rõ ràng. Hãy thử lại video khác!"

    # --- Vẽ đồ thị ---
    plt.figure(figsize=(8, 4))
    plt.plot(thoi_gian, toa_do_y, 'ro-', markersize=4, label='Tọa độ vật')
    plt.gca().invert_yaxis()
    plt.title('Đồ thị Tọa độ y - Thời gian t (Rơi tự do)')
    plt.xlabel('Thời gian t (giây)')
    plt.ylabel('Tọa độ y (pixel)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_img.name, bbox_inches='tight')
    plt.close()

    # --- Hiệu chuẩn thông số Toán học ---
    do_roi_pixel = max(toa_do_y) - min(toa_do_y)
    if do_roi_pixel == 0: do_roi_pixel = 1

    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_pixel
    toa_do_y_met = [y * ti_le_pixel_met for y in toa_do_y]

    he_so = np.polyfit(thoi_gian, toa_do_y_met, 2)
    a, b, c_coef = he_so[0], he_so[1], he_so[2]

    g_do_duoc = 2 * a
    sai_so = abs(g_do_duoc - 9.8) / 9.8 * 100

    # --- Tạo Báo cáo ---
    bao_cao = f"📊 THÔNG SỐ ĐỘNG HỌC (Đã hiệu chuẩn theo độ cao {chieu_cao_thuc_te}m):\n"
    bao_cao += f"- Phương trình: y = {a:.2f}t² + {b:.2f}t + {c_coef:.2f}\n"
    bao_cao += f"- Vận tốc ban đầu (v0): {b:.2f} m/s\n"
    bao_cao += f"- Gia tốc rơi (g) đo được: {g_do_duoc:.2f} m/s²\n"
    bao_cao += f"- Sai số so với g chuẩn (9.8): {sai_so:.2f}%\n\n"

    bao_cao += "🤖 AI ĐÁNH GIÁ THÍ NGHIỆM:\n"
    if b < -0.5:
        bao_cao += "❌ LỖI THAO TÁC: Bạn đã ném nhẹ vật lên trên chứ không phải buông tay thả rơi tự do. Yêu cầu thực hiện lại!"
    elif sai_so > 20:
        bao_cao += "⚠️ CẢNH BÁO: Sai số lớn. Hãy kiểm tra lại góc đặt camera hoặc thông số chiều cao nhập vào có chính xác không."
    else:
        bao_cao += "✅ XUẤT SẮC: Thí nghiệm rất chuẩn xác. Thao tác tốt!"

    return temp_img.name, bao_cao

# --- Thiết kế Giao diện Web bằng Streamlit ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")
st.markdown("# 🚀 PHYSIX - Nền tảng AI Phân tích Thí nghiệm Vật lý")

col1, col2 = st.columns(2)

with col1:
    video_dau_vao = st.file_uploader("Tải file video thí nghiệm (chấp nhận mp4, mov, avi)", type=['mp4', 'mov', 'avi'])
    chieu_cao_input = st.number_input("Nhập chiều cao thả rơi thực tế (mét)", value=1.5, step=0.1)
    nut_phan_tich = st.button("Phân tích bằng AI", type="primary", use_container_width=True)

with col2:
    if nut_phan_tich:
        if video_dau_vao is not None:
            # Lưu video tạm thời để OpenCV có thể đọc được file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_dau_vao.read())
                temp_video_path = tfile.name

            with st.spinner('AI đang phân tích video từng khung hình...'):
                anh_do_thi, van_ban_bao_cao = phan_tich_video(temp_video_path, chieu_cao_input)

            if anh_do_thi is None: 
                st.error(van_ban_bao_cao)
            else:
                st.image(anh_do_thi, caption="Đồ thị Động học")
                st.text_area("Kết quả & Đánh giá", value=van_ban_bao_cao, height=250)

            # Xóa file video tạm sau khi phân tích xong để giải phóng bộ nhớ
            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video trước khi nhấn phân tích!")
