import cv2
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import tempfile

# Hàm xử lý đã được bổ sung biến số "chieu_cao_thuc_te"
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
    do_roi_pixel = max(toa_do_y) - min(toa_do_y) # Tự động tính quãng đường rơi bằng pixel
    if do_roi_pixel == 0: do_roi_pixel = 1

    # Tỉ lệ quy đổi động dựa trên chiều cao người dùng nhập
    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_pixel
    toa_do_y_met = [y * ti_le_pixel_met for y in toa_do_y]

    he_so = np.polyfit(thoi_gian, toa_do_y_met, 2)
    a, b, c_coef = he_so[0], he_so[1], he_so[2]

    g_do_duoc = 2 * a
    sai_so = abs(g_do_duoc - 9.8) / 9.8 * 100

    # --- Tạo Báo cáo ---
    bao_cao = f"📊 THÔNG SỐ ĐỘNG HỌC (Đã hiệu chuẩn theo độ cao {chieu_cao_thuc_te}m):\n"
    bao_cao += f"- Phương trình: y = {a:.2f}t^2 + {b:.2f}t + {c_coef:.2f}\n"
    bao_cao += f"- Vận tốc ban đầu (v0): {b:.2f} m/s\n"
    bao_cao += f"- Gia tốc rơi (g) đo được: {g_do_duoc:.2f} m/s^2\n"
    bao_cao += f"- Sai số so với g chuẩn (9.8): {sai_so:.2f}%\n\n"

    bao_cao += "🤖 AI ĐÁNH GIÁ THÍ NGHIỆM:\n"
    if b < -0.5:
        bao_cao += "❌ LỖI THAO TÁC: Bạn đã ném nhẹ vật lên trên chứ không phải buông tay thả rơi tự do. Yêu cầu thực hiện lại!"
    elif sai_so > 20:
        bao_cao += "⚠️ CẢNH BÁO: Sai số lớn. Hãy kiểm tra lại góc đặt camera hoặc thông số chiều cao nhập vào có chính xác không."
    else:
        bao_cao += "✅ XUẤT SẮC: Thí nghiệm rất chuẩn xác. Thao tác tốt!"

    return temp_img.name, bao_cao

# Thiết kế Giao diện Web được cập nhật
with gr.Blocks(theme=gr.themes.Soft()) as ung_dung:
    gr.Markdown("# 🚀 PHYSIX - Nền tảng AI Phân tích Thí nghiệm Vật lý")

    with gr.Row():
        with gr.Column():
            video_dau_vao = gr.File(label="Tải file video thí nghiệm (chấp nhận mọi đuôi file)", type="filepath")
            # Ô nhập liệu chiều cao mới được thêm vào
            chieu_cao_input = gr.Number(label="Nhập chiều cao thả rơi thực tế (mét)", value=1.5, step=0.1)
            nut_phan_tich = gr.Button("Phân tích bằng AI", variant="primary")

        with gr.Column():
            anh_do_thi = gr.Image(label="Đồ thị Động học")
            van_ban_bao_cao = gr.Textbox(label="Kết quả & Đánh giá", lines=8)

    # Kết nối dữ liệu từ ô nhập chiều cao vào thuật toán phân tích
    nut_phan_tich.click(fn=phan_tich_video, inputs=[video_dau_vao, chieu_cao_input], outputs=[anh_do_thi, van_ban_bao_cao])

ung_dung.launch(share=True)
