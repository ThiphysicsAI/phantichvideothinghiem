import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
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
    
    # Đặt video chạy từ khung hình bắt đầu do người dùng chọn
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

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                thoi_gian.append((frame_count - frame_bat_dau) / fps) # Thời gian bắt đầu từ 0
                
                # Áp dụng gốc tọa độ mới (Dịch chuyển hệ quy chiếu)
                toa_do_x.append(x - goc_x)
                toa_do_y.append(y - goc_y)
                
        frame_count += 1
    cap.release()

    if len(toa_do_y) < 5:
        return None, None, "AI không tìm thấy vật thể. Hãy chọn lại khung hình hoặc video khác!"

    # --- TÍNH TOÁN ---
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

    # Vẽ đồ thị
    plt.figure(figsize=(8, 4))
    plt.plot(thoi_gian, toa_do_y, 'ro-', markersize=4, label='Tọa độ vật')
    plt.gca().invert_yaxis()
    plt.title('Đồ thị Tọa độ y - Thời gian t (Rơi tự do)')
    plt.xlabel('Thời gian t (giây)')
    plt.ylabel(f'Tọa độ y (pixel) so với gốc ({goc_x}, {goc_y})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_img.name, bbox_inches='tight')
    plt.close()

    return temp_img.name, thong_so_tho, None

# --- GIAO DIỆN WEB ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 1.15rem !important; }
    .title-physix { font-size: 2.2rem !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 5px; }
    div.stButton > button { font-size: 1.3rem !important; font-weight: bold !important; padding: 0.8rem 1.5rem !important; border-radius: 12px !important; background-color: #FF4B4B !important; color: white !important; border: none !important; width: 100% !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.1s; }
</style>
<div class='title-physix'>🚀 PHYSIX - NỀN TẢNG AI PHÂN TÍCH VẬT LÝ</div>
""", unsafe_allow_html=True)

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
        
        st.markdown("**✂️ Cắt video (chọn khung hình thả vật và kết thúc):**")
        frame_range = st.slider("Khung hình", 0, tong_so_frame, (0, tong_so_frame))
        
        # Lấy khung hình đầu tiên để hiển thị cho người dùng click chọn tọa độ
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.markdown("**📍 Chọn gốc tọa độ & 📏 Thước chuẩn (Click vào ảnh dưới):**")
            # Dùng thư viện image_coordinates để lấy điểm click
            toa_do_click = streamlit_image_coordinates(frame_rgb, key="click_toa_do")
            
            goc_x, goc_y = 0, 0
            if toa_do_click:
                goc_x, goc_y = toa_do_click['x'], toa_do_click['y']
                st.success(f"Đã chọn gốc tọa độ tại Pixel (X: {goc_x}, Y: {goc_y})")
            else:
                st.info("Hãy click vào ảnh trên để đặt gốc tọa độ O(0,0).")
                
            # Mô phỏng tính tỷ lệ pixel/mét (Vì Streamlit khó hỗ trợ click 2 điểm liên tiếp mượt mà, ta cho người dùng nhập khoảng cách Pixel)
            st.markdown("📏 **Hiệu chuẩn khoảng cách (Scale):**")
            col_a, col_b = st.columns(2)
            with col_a:
                chieu_cao_input = st.number_input("Chiều dài thực tế (mét)", value=1.5, step=0.1)
            with col_b:
                pixel_input = st.number_input("Tương ứng với số Pixel trên video", value=500)
            
            ti_le_pixel_met = chieu_cao_input / pixel_input if pixel_input > 0 else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            nut_phan_tich = st.button("✨ PHÂN TÍCH & NHẬN XÉT BẰNG AI ✨", use_container_width=True)

with col2:
    if video_dau_vao is not None and 'nut_phan_tich' in locals() and nut_phan_tich:
        with st.spinner('⏳ Đang xử lý video theo hệ tọa độ của bạn...'):
            anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input, frame_range[0], frame_range[1], goc_x, goc_y, ti_le_pixel_met)

            if loi:
                st.error(loi)
            else:
                st.markdown("### 📊 2. Kết quả & Đồ thị")
                st.image(anh_do_thi, caption="Đồ thị Động học trích xuất từ video của em", use_container_width=True)
                
                try:
                    api_key_an = st.secrets["GEMINI_API_KEY"]
                    genai.configure(api_key=api_key_an)
                    
                    ten_model_phu_hop = 'models/gemini-3.5-flash-lite'
                    model = genai.GenerativeModel(ten_model_phu_hop)
                    img_to_send = Image.open(anh_do_thi)
                    
                    prompt = f"""
                    Bạn là một giáo viên Vật lí cấp 3 tâm huyết, thân thiện tên là cô Thi.
                    Học sinh vừa làm xong thí nghiệm thả rơi tự do. Học sinh đã tự hiệu chỉnh khung hình, gốc tọa độ và thước đo.
                    Dựa vào bức ảnh đồ thị y-t đính kèm và số liệu:
                    - Phương trình: {thong_so['phuong_trinh']}
                    - Gia tốc g: {thong_so['g']:.2f} m/s² (Sai số: {thong_so['sai_so']:.2f}%)
                    
                    Hãy nhận xét khích lệ học sinh về kết quả này (không dùng mã LaTeX).
                    """
                    
                    response = model.generate_content([prompt, img_to_send])
                    st.markdown("### 🤖 3. Nhận xét từ cô Thi AI")
                    st.success(response.text)
                    
                except Exception as e:
                    st.error(f"Lỗi khi kết nối với Gemini: {e}")
