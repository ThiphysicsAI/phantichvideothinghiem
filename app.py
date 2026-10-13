import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
import google.generativeai as genai
from PIL import Image

# --- HÀM TÍNH HỆ SỐ TIN CẬY R-SQUARED ---
def tinh_r2(y_thuc_te, y_du_doan):
    ss_res = np.sum((np.array(y_thuc_te) - np.array(y_du_doan))**2)
    ss_tot = np.sum((np.array(y_thuc_te) - np.mean(y_thuc_te))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# --- HÀM XỬ LÝ PHÂN TÍCH ĐA ĐỘNG HỌC ---
def phan_tich_video(video_path, chieu_cao_thuc_te, loai_chuyen_dong):
    if video_path is None:
        return None, None, "Vui lòng tải lên một video!"
    if chieu_cao_thuc_te <= 0:
        return None, None, "Chiều cao thực tế phải lớn hơn 0!"

    cap = cv2.VideoCapture(video_path)
    toa_do_x, toa_do_y, thoi_gian = [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    prev_x, prev_y = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, None, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 80 < area < 3000:
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    circularity = 4 * np.pi * area / (peri * peri)
                    if circularity > 0.4:
                        valid_contours.append(c)

        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            if prev_x is not None and prev_y is not None:
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if dist > 150:
                    frame_count += 1
                    continue

            thoi_gian.append(frame_count / fps)
            toa_do_x.append(x)
            toa_do_y.append(y)
            prev_x, prev_y = x, y
            
        frame_count += 1
    cap.release()

    if len(toa_do_y) < 5:
        return None, None, "AI không tìm thấy quỹ đạo rõ ràng! Đảm bảo quả bóng không bị che khuất và có màu nổi bật."

    # Lọc điểm nảy (Bounce Filter)
    diem_cham_dat_idx = np.argmax(toa_do_y)
    toa_do_y = toa_do_y[:diem_cham_dat_idx + 1]
    toa_do_x = toa_do_x[:diem_cham_dat_idx + 1]
    thoi_gian = thoi_gian[:diem_cham_dat_idx + 1]

    if len(toa_do_y) < 5:
         return None, None, "❌ Chuyển động trước khi chạm mặt phẳng quá ngắn!"

    do_lech_x = max(toa_do_x) - min(toa_do_x)
    do_roi_y = max(toa_do_y) - min(toa_do_y)
    if do_roi_y == 0: do_roi_y = 1
    if do_lech_x == 0: do_lech_x = 1

    if loai_chuyen_dong == "Rơi tự do" and do_lech_x > do_roi_y * 0.4:
        return None, None, "❌ Cảnh báo vật thể bị ném ngang. Các em hãy thả rơi tự do thẳng đứng!"
    elif loai_chuyen_dong in ["Ném ngang", "Ném xiên"] and do_lech_x < do_roi_y * 0.15:
        return None, None, f"❌ Quỹ đạo quá thẳng đứng, giống rơi tự do hơn!"

    # --- QUY ĐỔI & TÍNH TOÁN VẬT LÝ SÂU ---
    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_y
    toa_do_y_met = [(y - min(toa_do_y)) * ti_le_pixel_met for y in toa_do_y]
    toa_do_x_met = [(x - min(toa_do_x)) * ti_le_pixel_met for x in toa_do_x]

    # Phân tích phương Y
    he_so_y = np.polyfit(thoi_gian, toa_do_y_met, 2)
    y_du_doan = np.polyval(he_so_y, thoi_gian)
    r2_y = tinh_r2(toa_do_y_met, y_du_doan)
    
    ay, by, cy = he_so_y[0], he_so_y[1], he_so_y[2]
    g_do_duoc = 2 * ay
    sai_so_g = abs(g_do_duoc - 9.8) / 9.8 * 100
    v0y = abs(by) # Vận tốc đầu trục Y

    thong_so = {
        "loai": loai_chuyen_dong,
        "phuong_trinh_y": f"y = {ay:.2f}t² + {by:.2f}t + {cy:.2f}",
        "g": g_do_duoc,
        "sai_so": sai_so_g,
        "v0y": v0y,
        "r2": r2_y * 100
    }

    # Phân tích phương X (nếu có)
    if loai_chuyen_dong in ["Ném ngang", "Ném xiên"]:
        he_so_x = np.polyfit(thoi_gian, toa_do_x_met, 1)
        x_du_doan = np.polyval(he_so_x, thoi_gian)
        v0x = abs(he_so_x[0])
        
        thong_so["phuong_trinh_x"] = f"x = {he_so_x[0]:.2f}t"
        thong_so["v0x"] = v0x
        
        # Riêng ném xiên tính thêm góc ném và Tầm xa/Tầm cao
        if loai_chuyen_dong == "Ném xiên":
            v0_tong = np.sqrt(v0x**2 + v0y**2)
            goc_nem = np.arctan(v0y / v0x) * 180 / np.pi if v0x != 0 else 90
            h_max_lt = (v0_tong**2 * (np.sin(goc_nem * np.pi / 180)**2)) / (2 * 9.8)
            h_max_tt = max(toa_do_y_met) - min(toa_do_y_met)
            
            thong_so.update({
                "goc_nem": goc_nem,
                "h_max_lt": h_max_lt,
                "h_max_tt": h_max_tt
            })

    # --- VẼ ĐỒ THỊ LỒNG GHÉP LÝ THUYẾT ---
    plt.figure(figsize=(8, 4))
    if loai_chuyen_dong == "Rơi tự do":
        plt.plot(thoi_gian, toa_do_y_met, 'ro-', markersize=4, label='Thực tế')
        plt.plot(thoi_gian, y_du_doan, 'r--', alpha=0.7, label='Lý thuyết (Fit line)')
        plt.title(f'Đồ thị Rơi tự do (Độ tin cậy R² = {thong_so["r2"]:.1f}%)')
        plt.xlabel('Thời gian t (giây)')
        plt.ylabel('Độ dịch chuyển y (m)')
    else:
        plt.plot(toa_do_x_met, toa_do_y_met, 'bo-', markersize=4, label='Thực tế')
        plt.plot(x_du_doan, y_du_doan, 'b--', alpha=0.7, label='Lý thuyết (Parabol)')
        plt.title(f'Đồ thị Quỹ đạo {loai_chuyen_dong} (Độ tin cậy R² = {thong_so["r2"]:.1f}%)')
        plt.xlabel('Tọa độ x (m)')
        plt.ylabel('Tọa độ y (m)')
        
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_img.name, bbox_inches='tight')
    plt.close()

    return temp_img.name, thong_so, None

# --- GIAO DIỆN WEB ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 1.15rem !important; }
    .title-physix { font-size: 2.2rem !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 5px; }
    div.stButton > button { font-size: 1.3rem !important; font-weight: bold !important; padding: 0.8rem 1.5rem !important; border-radius: 12px !important; background-color: #FF4B4B !important; color: white !important; border: none !important; width: 100% !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.1s; }
    div.stButton > button:active { transform: scale(0.98); }
    .stMarkdown p, label { font-size: 1.1rem !important; font-weight: 600 !important; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    @keyframes fall { 0% { transform: translateY(-5px); } 50% { transform: translateY(5px); } 100% { transform: translateY(-5px); } }
    @keyframes launch { 0% { transform: translateY(0) scale(1); } 50% { transform: translateY(-3px) scale(1.05); } 100% { transform: translateY(0) scale(1); } }
    .emoji-spin { display: inline-block; animation: spin 4s linear infinite; }
    .emoji-fall { display: inline-block; animation: fall 2s ease-in-out infinite; }
    .emoji-launch { display: inline-block; animation: launch 1.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='title-physix'>
    <span class='emoji-launch'>🚀</span> PHYSIX - NỀN TẢNG AI PHÂN TÍCH VẬT LÝ
</div>
<p style='text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 25px;'>
    Hãy chọn loại chuyển động và tải video lên để trợ lý AI phân tích nhé! <span class='emoji-fall'>🍎</span><span class='emoji-spin'>🪐</span>
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.1])

with col1:
    st.markdown("### 📥 1. Thiết lập bài thực hành")
    loai_chuyen_dong = st.radio("Chọn dạng chuyển động:", ["Rơi tự do", "Ném ngang", "Ném xiên"], horizontal=True)
    video_dau_vao = st.file_uploader(f"Tải video thí nghiệm ({loai_chuyen_dong})", type=['mp4', 'mov', 'avi'])
    chieu_cao_input = st.number_input("Nhập chiều cao/tầm xa max thực tế (mét)", value=1.5, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    nut_phan_tich = st.button("✨ PHÂN TÍCH & NHẬN XÉT BẰNG AI ✨", use_container_width=True)

with col2:
    if nut_phan_tich:
        if video_dau_vao is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_dau_vao.read())
                temp_video_path = tfile.name

            with st.spinner('⏳ Đang xử lý video, trích xuất dữ liệu và gọi giáo viên AI...'):
                anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input, loai_chuyen_dong)

                if loi:
                    st.error(loi)
                else:
                    st.markdown("### 📊 2. Kết quả & Đồ thị")
                    st.image(anh_do_thi, caption=f"Đồ thị trích xuất từ video: {loai_chuyen_dong}", use_container_width=True)
                    
                    try:
                        api_key_an = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key_an)
                        
                        model = genai.GenerativeModel('models/gemini-3.5-flash-lite')
                        img_to_send = Image.open(anh_do_thi)
                        
                        # --- CHUẨN BỊ BỘ DỮ LIỆU ĐỘNG HỌC GỬI CHO AI ---
                        du_lieu_bo_sung = ""
                        if loai_chuyen_dong == "Rơi tự do":
                            du_lieu_bo_sung = f"- Vận tốc đầu v0 đo được: {thong_so['v0y']:.2f} m/s (Lý thuyết phải = 0)."
                        elif loai_chuyen_dong == "Ném ngang":
                            du_lieu_bo_sung = f"- Vận tốc ném ngang v0x: {thong_so['v0x']:.2f} m/s.\n- Vận tốc rơi ban đầu v0y: {thong_so['v0y']:.2f} m/s (Lý thuyết phải = 0)."
                        elif loai_chuyen_dong == "Ném xiên":
                            du_lieu_bo_sung = f"- Góc ném ban đầu: {thong_so['goc_nem']:.1f} độ.\n- Tầm cao tối đa (Thực tế): {thong_so['h_max_tt']:.2f} m.\n- Tầm cao tối đa (Lý thuyết theo góc): {thong_so['h_max_lt']:.2f} m."

                        prompt = f"""
                        Bạn là một giáo viên Vật lí cấp 3 xuất sắc tên là cô Thi. Học sinh vừa hoàn thành thí nghiệm {thong_so['loai']}.
                        
                        HÃY ĐÁNH GIÁ DỰA TRÊN CÁC SỐ LIỆU CHUYÊN SÂU SAU:
                        - Phương trình trục Oy: {thong_so['phuong_trinh_y']}
                        - Gia tốc g đo được: {thong_so['g']:.2f} m/s² (Sai số: {thong_so['sai_so']:.2f}%)
                        - Độ tin cậy của đồ thị (R²): {thong_so['r2']:.1f}% 
                        {du_lieu_bo_sung}
                        
                        BAREM ĐÁNH GIÁ (TUYỆT ĐỐI TUÂN THỦ):
                        1. Độ tin cậy đồ thị (R²): Nếu >95% hãy khen các em quay video mượt và thao tác chuẩn; nếu <90% nhắc nhở đồ thị bị nhiễu do quay rung tay hoặc vật bị che.
                        2. Sai số gia tốc (g): Dưới 5% (Xuất sắc); 5-15% (Khá tốt, do lực cản không khí); >15% (Nhắc nhở kiểm tra lại góc quay camera bị lệch phối cảnh).
                        3. Kiểm chứng định luật: 
                           - Nếu là Rơi tự do/Ném ngang: Dựa vào v0y, nếu v0y > 0.5m/s thì nhắc học sinh lỗi "chưa buông tay tự nhiên mà lỡ ném vật".
                           - Nếu là Ném ngang/xiên: Khen ngợi việc ghi lại được quỹ đạo parabol độc lập ngang-dọc. Phân tích thêm sự chênh lệch tầm cao thực tế và lý thuyết (nếu là ném xiên).
                           
                        Yêu cầu trình bày:
                        - Dài khoảng 200 chữ. Xưng "cô" - "các em". Giọng văn chuyên môn, phân tích sắc sảo nhưng vẫn khích lệ, thân thiện, dùng emoji.
                        - KHÔNG DÙNG LATEX ($ hay \text). Viết đơn vị bình thường (VD: m/s2, v0).
                        """
                        
                        response = model.generate_content([prompt, img_to_send])
                        
                        st.markdown("### 🤖 3. Báo cáo Nhận xét từ cô Thi AI")
                        st.success(response.text)
                        
                    except KeyError:
                        st.error("Lỗi: Không tìm thấy GEMINI_API_KEY trong cấu hình.")
                    except Exception as e:
                        st.error(f"Lỗi khi kết nối với Gemini: {e}")

            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video trước!")
