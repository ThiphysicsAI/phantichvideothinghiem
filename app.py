import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
import google.generativeai as genai
from PIL import Image

# --- HÀM XỬ LÝ PHÂN TÍCH VIDEO ---
def phan_tich_video(video_path, chieu_cao_thuc_te):
    if video_path is None:
        return None, None, "Vui lòng tải lên một video!"
    if chieu_cao_thuc_te <= 0:
        return None, None, "Chiều cao thực tế phải lớn hơn 0!"

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
        return None, None, "AI không tìm thấy vật thể rơi rõ ràng. Hãy thử lại video khác!"

    # Vẽ đồ thị
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

    # Tính toán thông số
    do_roi_pixel = max(toa_do_y) - min(toa_do_y)
    if do_roi_pixel == 0: do_roi_pixel = 1
    ti_le_pixel_met = chieu_cao_thuc_te / do_roi_pixel
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

    return temp_img.name, thong_so_tho, None


# --- GIAO DIỆN WEB TỐI ƯU HÓA MOBILE & MÁY TÍNH ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")

# Nhúng Custom CSS để phóng to chữ, đổi màu nút và thêm emoji chuyển động
st.markdown("""
<style>
    /* Phóng to toàn bộ cỡ chữ cơ bản */
    html, body, [class*="css"] {
        font-size: 1.15rem !important;
    }
    
    /* Làm nổi bật tiêu đề lớn và thêm hoạt ảnh */
    .title-physix {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* Thiết kế nút bấm siêu lớn, dễ chạm trên điện thoại */
    div.stButton > button {
        font-size: 1.3rem !important;
        font-weight: bold !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 12px !important;
        background-color: #FF4B4B !important;
        color: white !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.1s;
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Phóng to các nhãn nhập liệu */
    .stMarkdown p, label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Hiệu ứng hoạt ảnh Emoji xoay tròn và rơi tự do */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fall {
        0% { transform: translateY(-5px); }
        50% { transform: translateY(5px); }
        100% { transform: translateY(-5px); }
    }
    @keyframes launch {
        0% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-3px) scale(1.05); }
        100% { transform: translateY(0) scale(1); }
    }

    .emoji-spin {
        display: inline-block;
        animation: spin 4s linear infinite;
    }
    .emoji-fall {
        display: inline-block;
        animation: fall 2s ease-in-out infinite;
    }
    .emoji-launch {
        display: inline-block;
        animation: launch 1.5s ease-in-out infinite;
    }
    
    /* Giao diện khung kết quả nổi bật */
    .result-box {
        background-color: #f0f7f4;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.15rem !important;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề lớn có chèn emoji động
st.markdown("""
<div class='title-physix'>
    <span class='emoji-launch'>🚀</span> PHYSIX - NỀN TẢNG AI PHÂN TÍCH VẬT LÝ
</div>
<p style='text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 25px;'>
    Hãy tải video thả rơi tự do để "trợ lý AI" của cô Thi giúp các em phân tích nhé! <span class='emoji-fall'>🍎</span><span class='emoji-spin'>🪐</span>
</p>
""", unsafe_allow_html=True)

# Streamlit sẽ tự động biến 2 cột này thành 1 cột dọc trên điện thoại để vừa khít màn hình
col1, col2 = st.columns([1, 1.1])

with col1:
    st.markdown("### 📥 1. Tải lên bài làm")
    video_dau_vao = st.file_uploader("Tải file video thí nghiệm (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])
    chieu_cao_input = st.number_input("Nhập chiều cao thả rơi thực tế (mét)", value=1.5, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    nut_phan_tich = st.button("✨ PHÂN TÍCH & NHẬN XÉT BẰNG AI ✨", use_container_width=True)

with col2:
    if nut_phan_tich:
        if video_dau_vao is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_dau_vao.read())
                temp_video_path = tfile.name

            # Thông báo tải lớn, rõ ràng kèm emoji xoay tròn
            with st.spinner('⏳ Đang xử lý video từng khung hình & kết nối với cô Thi AI...'):
                anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input)

                if loi:
                    st.error(loi)
                else:
                    st.markdown("### 📊 2. Kết quả & Đồ thị")
                    st.image(anh_do_thi, caption="Đồ thị Động học trích xuất từ video của em", use_container_width=True)
                    
                    try:
                        api_key_an = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key_an)
                        
                        # Sử dụng thẳng gemini-2.5-flash để phản hồi cực nhanh
                        ten_model_phu_hop = 'models/gemini-3.5-flash-lite'
                        model = genai.GenerativeModel(ten_model_phu_hop)
                        img_to_send = Image.open(anh_do_thi)
                        
                        prompt = f"""
                        Bạn là một giáo viên Vật lí cấp 3 tâm huyết, thân thiện tên là cô Thi.
                        Học sinh vừa làm xong thí nghiệm thả rơi tự do vật thể.
                        Dựa vào bức ảnh đồ thị y-t đính kèm và các số liệu đo được dưới đây, hãy viết một lời nhận xét ngắn gọn (khoảng 150 chữ) gửi trực tiếp cho học sinh.
                        - Phương trình chuyển động: {thong_so['phuong_trinh']}
                        - Vận tốc ban đầu v0: {thong_so['v0']:.2f} m/s
                        - Gia tốc g đo được: {thong_so['g']:.2f} m/s²
                        - Sai số so với g chuẩn (9.8): {thong_so['sai_so']:.2f}%
                        
                        Yêu cầu:
                        - Khen ngợi nếu sai số thấp (<10%), động viên và chỉ ra nguyên nhân (vận tốc đầu khác 0 do lực ném tay, lực cản không khí, góc quay camera) nếu sai số cao.
                        - Xưng hô "cô" và "các em". Dùng giọng văn khích lệ, thấu hiểu và thật nhiều emoji liên quan đến trường học, vật lý.
                        """
                        
                        response = model.generate_content([prompt, img_to_send])
                        
                        st.markdown("### 🤖 3. Nhận xét từ cô Thi AI")
                        # Đưa văn bản nhận xét vào khung lớn đặc biệt
                        st.markdown(f"""
                        <div class='result-box'>
                            {response.text.replace('\n', '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except KeyError:
                        st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Hãy kiểm tra lại phần cài đặt Secrets trên Streamlit Cloud.")
                    except Exception as e:
                        st.error(f"Lỗi khi kết nối với Gemini: {e}")

            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video trước!")
