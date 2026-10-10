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
                        
                        # Quét toàn bộ model được cấp phép cho API Key này
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                danh_sach_model.append(m.name)
                                
                        # Ưu tiên chọn các model đọc được ảnh (từ mới nhất đến cũ)
                        if 'models/gemini-1.5-flash' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-1.5-flash'
                        elif 'models/gemini-1.5-pro' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-1.5-pro'
                        elif 'models/gemini-pro-vision' in danh_sach_model:
                            ten_model_phu_hop = 'models/gemini-pro-vision'
                        else:
                            # Nếu không khớp tên nào, lấy model cuối cùng trong danh sách
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
                        # Nếu vẫn lỗi, in toàn bộ danh sách model ra màn hình để bắt bệnh
                        try:
                            st.info(f"Hệ thống tìm thấy các model sau trong tài khoản của cô: {danh_sach_model}")
                        except:
                            pass

            os.remove(temp_video_path)
        else:
            st.warning("⚠️ Vui lòng tải lên một video!")
