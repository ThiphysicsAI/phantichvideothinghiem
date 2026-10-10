import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tempfile
import os
import google.generativeai as genai
from PIL import Image

# Hàm xử lý phân tích video vật lí
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

# --- Giao diện Web ---
st.set_page_config(page_title="PHYSIX - Phân tích thí nghiệm", layout="wide")

st.markdown("# 🚀 PHYSIX - Nền tảng AI Phân tích Thí nghiệm Vật lý")

col1, col2 = st.columns([1, 1.2])

with col1:
    video_dau_vao = st.file_uploader("Tải file video thí nghiệm (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])
    chieu_cao_input = st.number_input("Nhập chiều cao thả rơi thực tế (mét)", value=1.5, step=0.1)
    nut_phan_tich = st.button("Phân tích & Nhận xét bằng AI", type="primary", use_container_width=True)

with col2:
    if nut_phan_tich:
        if video_dau_vao is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_dau_vao.read())
                temp_video_path = tfile.name

            with st.spinner('Đang xử lý hình ảnh và gọi Gemini AI nhận xét...'):
                anh_do_thi, thong_so, loi = phan_tich_video(temp_video_path, chieu_cao_input)

                if loi:
                    st.error(loi)
                else:
                    st.image(anh_do_thi, caption="Đồ thị Động học trích xuất từ video")
                    
                    # Tự động lấy API Key và quét tên model
                    try:
                        api_key_an = st.secrets["GEMINI_API_KEY"]
                        genai.configure(api_key=api_key_an)
                        
                        # --- THUẬT TOÁN TỰ ĐỘNG TÌM TÊN MODEL ---
                        danh_sach_model = []
                        ten_model_phu_hop = None
                        
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                danh_sach_model.append(m.name)
                                
                        if 'models/gemini-1.5-flash' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-1.5-flash'
                        elif 'models/gemini-1.5-pro' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-1.5-pro'
                        elif 'models/gemini-pro-vision' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-pro-vision'
                        else:
                            ten_model_phu_hop = danh_sach_model[-1] if danh_sach_model else 'gemini-1.5-flash'
                        # ----------------------------------------
                        
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
                        - Xưng hô "cô" và "các em". Dùng giọng văn khích lệ, thấu hiểu.
                        """
                        
                        response = model.generate_content([prompt, img_to_send])
                        st.success(f"🤖 **Nhận xét từ AI (đang dùng {ten_model_phu_hop}):**")
                        st.markdown(response.text)
                        
                    except KeyError:
                        st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Hãy kiểm tra lại phần cài đặt Secrets trên Streamlit Cloud.")
                    except Exception as e:
                        st.error(f"Lỗi khi kết nối với Gemini: {e}")
                        try:
                            st.info(f"Hệ thống tìm thấy các model sau trong tài khoản của cô: {danh_sach_model}")
                        except:
                            pass

            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video!")
