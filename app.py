import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
import google.generativeai as genai
from PIL import Image

# --- HÀM XỬ LÝ PHÂN TÍCH ĐA ĐỘNG HỌC (NÂNG CẤP NHẬN DIỆN MÀU SẮC) ---
def phan_tich_video(video_path, chieu_cao_thuc_te, loai_chuyen_dong):
    if video_path is None:
        return None, None, "Vui lòng tải lên một video!"
    if chieu_cao_thuc_te <= 0:
        return None, None, "Chiều cao thực tế phải lớn hơn 0!"

    cap = cv2.VideoCapture(video_path)
    toa_do_x, toa_do_y, thoi_gian = [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    # Lọc màu vàng (HSV) để tránh nhiễu từ người và giá đỡ
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Chuyển đổi khung hình sang không gian màu HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tạo mặt nạ chỉ lấy những vùng có màu vàng
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Khử nhiễu
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 50: # Giảm ngưỡng diện tích để bắt bóng tốt hơn
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                thoi_gian.append(frame_count / fps)
                toa_do_x.append(x)
                toa_do_y.append(y)
        frame_count += 1
    cap.release()

    if len(toa_do_y) < 5:
        return None, None, "AI không tìm thấy quả bóng màu vàng. Vui lòng đảm bảo vật thể thả rơi có màu vàng nổi bật!"

    # --- BỘ LỌC ĐỘNG HỌC THEO LOẠI CHUYỂN ĐỘNG ---
    do_lech_x = max(toa_do_x) - min(toa_do_x)
    do_roi_y = max(toa_do_y) - min(toa_do_y)
    if do_roi_y == 0: do_roi_y = 1
    if do_lech_x == 0: do_lech_x = 1

    if loai_chuyen_dong == "Rơi tự do":
        if do_lech_x > do_roi_y * 0.4:
            return None, None, "❌ Chế độ Rơi tự do: Cảnh báo vật thể bị ném ngang. Các em hãy chọn đúng chế độ hoặc thả lại video!"
    elif loai_chuyen_dong in ["Ném ngang", "Ném xiên"]:
        if do_lech_x < do_roi_y * 0.15:
            return None, None, f"❌ Chế độ {loai_chuyen_dong}: Quỹ đạo quá thẳng đứng, giống rơi tự do hơn. Các em kiểm tra lại nhé!"

    # --- QUY ĐỔI HỆ QUY CHIẾU & TÍNH TOÁN ---
    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_y
    toa_do_y_met = [(y - min(toa_do_y)) * ti_le_pixel_met for y in toa_do_y]
    toa_do_x_met = [(x - min(toa_do_x)) * ti_le_pixel_met for x in toa_do_x]

    he_so_y = np.polyfit(thoi_gian, toa_do_y_met, 2)
    ay, by, cy = he_so_y[0], he_so_y[1], he_so_y[2]
    g_do_duoc = 2 * ay
    sai_so = abs(g_do_duoc - 9.8) / 9.8 * 100

    thong_so_tho = {
        "loai": loai_chuyen_dong,
        "phuong_trinh_y": f"y = {ay:.2f}t² + {by:.2f}t + {cy:.2f}",
        "g": g_do_duoc,
        "sai_so": sai_so
    }

    if loai_chuyen_dong in ["Ném ngang", "Ném xiên"]:
        he_so_x = np.polyfit(thoi_gian, toa_do_x_met, 1)
        thong_so_tho["phuong_trinh_x"] = f"x = {he_so_x[0]:.2f}t"
        thong_so_tho["vx"] = he_so_x[0]

    # --- VẼ ĐỒ THỊ ---
    plt.figure(figsize=(8, 4))
    if loai_chuyen_dong == "Rơi tự do":
        plt.plot(thoi_gian, toa_do_y_met, 'ro-', markersize=4, label='Tọa độ y theo t')
        plt.title('Đồ thị Động học (Rơi tự do)')
        plt.xlabel('Thời gian t (giây)')
        plt.ylabel('Độ dịch chuyển y (m)')
    else:
        plt.plot(toa_do_x_met, toa_do_y_met, 'bo-', markersize=4, label='Quỹ đạo Parabol')
        plt.title(f'Đồ thị Quỹ đạo x-y ({loai_chuyen_dong})')
        plt.xlabel('Tọa độ x (m)')
        plt.ylabel('Tọa độ y (m)')
        
    plt.gca().invert_yaxis()
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
    chieu_cao_input = st.number_input("Nhập chiều cao max quỹ đạo thực tế (mét)", value=1.5, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    nut_phan_tich = st.button("✨ PHÂN TÍCH & NHẬN XÉT BẰNG AI ✨", use_container_width=True)

with col2:
    if nut_phan_tich:
        if video_dau_vao is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_dau_vao.read())
                temp_video_path = tfile.name

            with st.spinner('⏳ Đang xử lý video từng khung hình & kết nối với trợ lý AI...'):
                anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input, loai_chuyen_dong)

                if loi:
                    st.error(loi)
                else:
                    st.markdown("### 📊 2. Kết quả & Đồ thị")
                    st.image(anh_do_thi, caption=f"Đồ thị trích xuất từ video: {loai_chuyen_dong}", use_container_width=True)
                    
                    try:
                        api_key_an = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key_an)
                        
                        ten_model_phu_hop = 'models/gemini-3.5-flash-lite'
                        model = genai.GenerativeModel(ten_model_phu_hop)
                        img_to_send = Image.open(anh_do_thi)
                        
                        thong_so_x_txt = f"\n- Phương trình trục Ox: {thong_so.get('phuong_trinh_x', 'Không có')}" if loai_chuyen_dong != "Rơi tự do" else ""
                        
                        prompt = f"""
                        Bạn là một giáo viên Vật lí cấp 3 tâm huyết, thân thiện tên là cô Thi.
                        Học sinh vừa làm xong thí nghiệm {thong_so['loai']} vật thể.
                        Dựa vào bức ảnh đồ thị đính kèm và các số liệu đo được dưới đây, hãy viết một lời nhận xét ngắn gọn (khoảng 150 chữ) gửi trực tiếp cho học sinh.
                        - Phương trình trục Oy: {thong_so['phuong_trinh_y']}{thong_so_x_txt}
                        - Gia tốc g đo được (từ Oy): {thong_so['g']:.2f} m/s²
                        - Sai số so với g chuẩn (9.8): {thong_so['sai_so']:.2f}%
                        
                        Yêu cầu:
                        - Khen ngợi nếu sai số thấp (<10%), động viên và chỉ ra nguyên nhân sai số (lực cản không khí, góc máy quay chưa vuông góc mặt phẳng chuyển động).
                        - Nếu là Ném ngang/Ném xiên, hãy khen quỹ đạo Parabol các em ghi lại được.
                        - Xưng hô "cô" và "các em". Dùng giọng văn khích lệ, thấu hiểu và thật nhiều emoji liên quan đến trường học, vật lý.
                        - QUAN TRỌNG: KHÔNG sử dụng định dạng toán học LaTeX (không dùng dấu $ hay \text). Viết đơn vị và công thức bằng văn bản bình thường.
                        """
                        
                        response = model.generate_content([prompt, img_to_send])
                        
                        st.markdown("### 🤖 3. Nhận xét từ cô Thi AI")
                        st.success(response.text)
                        
                    except KeyError:
                        st.error("Lỗi: Không tìm thấy GEMINI_API_KEY trong cấu hình.")
                    except Exception as e:
                        st.error(f"Lỗi khi kết nối với Gemini: {e}")

            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video trước!")
