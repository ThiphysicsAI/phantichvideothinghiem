import cv2
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import tempfile
import requests # Dùng giao thức web trực tiếp, không bao giờ lỗi phiên bản thư viện!

# --- TÍCH HỢP CHÌA KHÓA GEMINI TẠI ĐÂY ---
GOOGLE_API_KEY = "AQ.Ab8RN6Jf3HJKbVinQ3IJ37zX9gWYg1ItdwGRTsxuJK0ncOkirA"

def goi_gemini_truc_tiep(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Lỗi kết nối từ máy chủ: {response.text}"
    except Exception as e:
        return f"Lỗi hệ thống: {e}"

def phan_tich_video(video_path, chieu_cao_thuc_te):
    if video_path is None: return None, "Vui lòng tải lên một video!"
    if chieu_cao_thuc_te <= 0: return None, "Chiều cao thực tế phải lớn hơn 0!"

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

    # 1. Vẽ đồ thị và lưu thành ảnh
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

    # 2. Tính toán Động học
    do_roi_pixel = max(toa_do_y) - min(toa_do_y)
    if do_roi_pixel == 0: do_roi_pixel = 1

    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_pixel
    toa_do_y_met = [y * ti_le_pixel_met for y in toa_do_y]

    he_so = np.polyfit(thoi_gian, toa_do_y_met, 2)
    a, b, c_coef = he_so[0], he_so[1], he_so[2]

    g_do_duoc = 2 * a
    sai_so = abs(g_do_duoc - 9.8) / 9.8 * 100

    # 3. CHUYỂN GIAO CHO GEMINI NHẬN XÉT
    prompt_kịch_bản = f"""
    Bạn là một trợ lý AI thông minh trong phòng thí nghiệm Vật lý.
    Học sinh vừa thực hiện thí nghiệm rơi tự do với chiều cao {chieu_cao_thuc_te}m.
    Hệ thống đo lường máy tính vừa tính toán ra các thông số sau:
    - Vận tốc ban đầu (v0): {b:.2f} m/s
    - Gia tốc trọng trường (g) đo được: {g_do_duoc:.2f} m/s^2 (Chuẩn là 9.8 m/s^2)
    - Sai số: {sai_so:.2f}%

    Dựa vào các thông số chính xác tuyệt đối này, hãy viết một đoạn nhận xét (khoảng 3-4 câu) gửi trực tiếp cho nhóm thực hành của lớp 12A7.
    Yêu cầu:
    - Lời lẽ mềm mỏng, mang tính khích lệ, xưng hô là "Cô AI" và gọi học sinh là "các em".
    - Nếu v0 là số âm đáng kể (< -0.5), hãy nhẹ nhàng nhắc học sinh rút kinh nghiệm vì đã ném nhẹ vật lên trên thay vì buông tay tự do hoàn toàn.
    - Nếu sai số cao (>20%), gợi ý các em kiểm tra lại góc đặt điện thoại quay video hoặc thông số chiều cao nhập vào đã chuẩn chưa.
    - Nếu kết quả đẹp (sai số thấp, v0 gần 0), hãy khen ngợi thao tác xuất sắc của nhóm.
    """

    loi_nhan_xet_tu_AI = goi_gemini_truc_tiep(prompt_kịch_bản)

    # 4. Đóng gói Báo cáo
    bao_cao = f"📊 THÔNG SỐ ĐỘNG HỌC:\n"
    bao_cao += f"- Vận tốc ban đầu (v0): {b:.2f} m/s\n"
    bao_cao += f"- Gia tốc rơi (g) đo được: {g_do_duoc:.2f} m/s^2\n"
    bao_cao += f"- Sai số so với g chuẩn (9.8): {sai_so:.2f}%\n"
    bao_cao += "----------------------------------------\n\n"
    bao_cao += f"🤖 LỜI NHẬN XÉT TỪ CÔ AI:\n{loi_nhan_xet_tu_AI}"

    return temp_img.name, bao_cao

# Giao diện Web
with gr.Blocks(theme=gr.themes.Soft()) as ung_dung:
    gr.Markdown("# 🚀 PHYSIX - Nền tảng AI Phân tích Thí nghiệm Vật lý")

    with gr.Row():
        with gr.Column():
            video_dau_vao = gr.File(label="Tải file video thí nghiệm (chấp nhận mọi đuôi file)", type="filepath")
            chieu_cao_input = gr.Number(label="Nhập chiều cao thả rơi thực tế (mét)", value=1.5, step=0.1)
            nut_phan_tich = gr.Button("Phân tích bằng AI", variant="primary")

        with gr.Column():
            anh_do_thi = gr.Image(label="Đồ thị Động học")
            van_ban_bao_cao = gr.Textbox(label="Kết quả & Đánh giá", lines=10)

    nut_phan_tich.click(fn=phan_tich_video, inputs=[video_dau_vao, chieu_cao_input], outputs=[anh_do_thi, van_ban_bao_cao])

ung_dung.launch(share=True)
