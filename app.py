import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
import math
import google.generativeai as genai
from PIL import Image
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Vui lòng thêm 'streamlit-image-coordinates' vào file requirements.txt")

# --- HÀM XỬ LÝ PHÂN TÍCH VIDEO ---
def phan_tich_video(video_path, chieu_cao_thuc_te, frame_bat_dau, frame_ket_thuc, goc_x, goc_y, ti_le_pixel_met):
    cap = cv2.VideoCapture(video_path)
    toa_do_x, toa_do_y, thoi_gian = [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_bat_dau)
    frame_count = frame_bat_dau
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    while cap.isOpened() and frame_count <= frame_ket_thuc:
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
            
            thoi_gian.append((frame_count - frame_bat_dau) / fps)
            # Áp dụng gốc tọa độ mới do người dùng click chọn
            toa_do_x.append(x - goc_x)
            toa_do_y.append(y - goc_y)
                
        frame_count += 1
    cap.release()

    if len(toa_do_y) < 5:
        return None, None, "AI không tìm thấy vật thể. Hãy chọn lại khung hình hoặc đảm bảo quả bóng có màu nổi bật!"

    # --- TÍNH TOÁN VẬT LÝ ---
    toa_do_y_met = [y * ti_le_pixel_met for y in toa_do_y]

    he_so = np.polyfit(thoi_gian, toa_do_y_met, 2)
    a, b, c_coef = he_so[0], he_so[1], he_so[2]
    g_do_duoc = 2 * a
    sai_so = abs(g_do_duoc - 9.8) / 9.8 * 100

    thong_so_tho = {
        "phuong_trinh": f"y = {a:.2f}t² + {b:.2f}t + {c_coef:.2f}",
        "v0": b,
        "g": g_do_duoc,
        "sai_so": sai_so
    }

    # --- VẼ ĐỒ THỊ ---
    plt.figure(figsize=(8, 4))
    plt.plot(thoi_gian, toa_do_y_met, 'ro-', markersize=4, label='Tọa độ vật (Đã trừ Gốc O)')
    plt.gca().invert_yaxis()
    plt.title('Đồ thị Tọa độ y - Thời gian t (Rơi tự do)')
    plt.xlabel('Thời gian t (giây)')
    plt.ylabel('Tọa độ y (mét)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_img.name, bbox_inches='tight')
    plt.close()

    return temp_img.name, thong_so_tho, None

# --- GIAO DIỆN WEB TƯƠNG TÁC CAO ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 1.15rem !important; }
    .title-physix { font-size: 2.2rem !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 5px; }
    div.stButton > button { font-size: 1.2rem !important; font-weight: bold !important; border-radius: 8px !important; background-color: #FF4B4B !important; color: white !important; width: 100% !important; }
</style>
<div class='title-physix'>🚀 PHYSIX - NỀN TẢNG AI PHÂN TÍCH VẬT LÝ</div>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ tạm để lưu 2 điểm click chuột
if 'diem_thuoc_do' not in st.session_state:
    st.session_state.diem_thuoc_do = []

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📥 1. Tải lên & Hiệu chỉnh Video")
    video_dau_vao = st.file_uploader("Tải file video thí nghiệm", type=['mp4', 'mov', 'avi'])
    
    if video_dau_vao is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_dau_vao.read())
            temp_video_path = tfile.name
            
        cap = cv2.VideoCapture(temp_video_path)
        tong_so_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.markdown("**✂️ Cắt video:** Kéo thanh trượt để chọn đúng đoạn vật bắt đầu rơi.")
        frame_range = st.slider("Khung hình", 0, tong_so_frame, (0, tong_so_frame))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            st.markdown("---")
            st.markdown("### 📍 2. Chấm tọa độ và Thước chuẩn")
            st.write("🔹 **Click lần 1:** Chọn Gốc tọa độ (Vd: điểm bắt đầu rơi).")
            st.write("🔹 **Click lần 2:** Chọn điểm thứ 2 để đo khoảng cách chuẩn (Vd: mặt sàn).")
            
            if st.button("🔄 Làm lại (Xóa các điểm đã chấm)"):
                st.session_state.diem_thuoc_do = []
                st.rerun()
                
            # Lấy dữ liệu click chuột
            click_val = streamlit_image_coordinates(frame_rgb, key="click_thuoc")
            
            # Xử lý logic lưu 2 điểm
            if click_val is not None:
                diem_moi = (click_val['x'], click_val['y'])
                # Tránh tình trạng lưu trùng 1 điểm khi app tải lại
                if len(st.session_state.diem_thuoc_do) == 0 or st.session_state.diem_thuoc_do[-1] != diem_moi:
                    if len(st.session_state.diem_thuoc_do) < 2:
                        st.session_state.diem_thuoc_do.append(diem_moi)

            goc_x, goc_y = 0, 0
            pixel_do_duoc = 100 # Chống lỗi chia 0
            
            # Hiển thị thông báo hướng dẫn tương tác
            if len(st.session_state.diem_thuoc_do) == 0:
                st.info("👈 Hãy click chuột vào ảnh để chọn điểm đầu tiên...")
            elif len(st.session_state.diem_thuoc_do) == 1:
                goc_x, goc_y = st.session_state.diem_thuoc_do[0]
                st.success(f"✅ Đã lưu Gốc O tại (X: {goc_x}, Y: {goc_y}). Hãy click điểm thứ 2!")
            elif len(st.session_state.diem_thuoc_do) >= 2:
                goc_x, goc_y = st.session_state.diem_thuoc_do[0]
                x2, y2 = st.session_state.diem_thuoc_do[1]
                st.success(f"✅ Gốc O: ({goc_x}, {goc_y}) | Điểm thứ hai: ({x2}, {y2})")
                
                # Áp dụng công thức Toán học Pytago để tính khoảng cách giữa 2 điểm
                pixel_do_duoc = math.sqrt((x2 - goc_x)**2 + (y2 - goc_y)**2)
                st.warning(f"📏 Phần mềm đã đo được khoảng cách trên ảnh là: **{pixel_do_duoc:.2f} Pixel**")
                
            st.markdown("**Nhập chiều dài thực tế giữa 2 điểm vừa click:**")
            chieu_cao_input = st.number_input("Chiều dài ngoài đời (mét)", value=1.00, step=0.1)
            
            ti_le_pixel_met = chieu_cao_input / pixel_do_duoc if pixel_do_duoc > 0 else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            nut_phan_tich = st.button("✨ XỬ LÝ SỐ LIỆU & GỌI AI NHẬN XÉT ✨", use_container_width=True)

with col2:
    if video_dau_vao is not None and 'nut_phan_tich' in locals() and nut_phan_tich:
        if len(st.session_state.diem_thuoc_do) < 2:
            st.error("Vui lòng click đủ 2 điểm trên màn hình để làm thước chuẩn trước khi Phân tích!")
        else:
            with st.spinner('⏳ Đang phân tích chuyển động theo hệ quy chiếu của bạn...'):
                anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input, frame_range[0], frame_range[1], goc_x, goc_y, ti_le_pixel_met)

                if loi:
                    st.error(loi)
                else:
                    st.markdown("### 📊 3. Kết quả & Đồ thị")
                    st.image(anh_do_thi, caption="Đồ thị Động học trích xuất từ video của em", use_container_width=True)
                    
                    try:
                        api_key_an = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key_an)
                        
                        ten_model_phu_hop = 'models/gemini-3.5-flash-lite'
                        model = genai.GenerativeModel(ten_model_phu_hop)
                        img_to_send = Image.open(anh_do_thi)
                        
                        prompt = f"""
                        Bạn là cô Thi, giáo viên Vật lí cấp 3.
                        Học sinh vừa làm xong thí nghiệm thả rơi tự do. Nhóm đã tự thiết lập Gốc tọa độ và Thước chuẩn trên ứng dụng.
                        - Phương trình tọa độ y theo t: {thong_so['phuong_trinh']}
                        - Gia tốc g đo được: {thong_so['g']:.2f} m/s² (Sai số: {thong_so['sai_so']:.2f}%)
                        
                        Hãy viết nhận xét khích lệ. Khen ngợi nếu sai số thấp. KHÔNG dùng định dạng toán học LaTeX.
                        """
                        
                        response = model.generate_content([prompt, img_to_send])
                        st.markdown("### 🤖 4. Nhận xét từ cô Thi AI")
                        st.success(response.text)
                        
                    except Exception as e:
                        st.error(f"Lỗi khi kết nối với Gemini: {e}")
