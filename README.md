# SmartDoc AI - Base RAG System

Hệ thống RAG cho phép upload tài liệu PDF và hỏi đáp dựa trên nội dung sử dụng mô hình LLM Qwen2.5:7b chạy hoàn toàn local.

## Yêu cầu môi trường
- Python 3.8+
- Ollama runtime

## Cài đặt và chạy thử (Sử dụng WSL - Ubuntu/Linux)

Vì máy tính Windows của bạn chưa cài Python, sử dụng WSL (Windows Subsystem for Linux) là phương án tuyệt vời nhất.

1. **Mở WSL Terminal** (Mở Ubuntu hoặc gõ `wsl` trong PowerShell).
2. **Di chuyển đến thư mục dự án**:
   (WSL tự động mount ổ C vào `/mnt/c`)
```bash
cd /mnt/c/Users/ngoco/Desktop/OS_project
```

3. **Tạo môi trường ảo (Khuyến nghị) và cài đặt thư viện**:
```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Cài đặt mô hình Ollama**:
- Tải [Ollama](https://ollama.ai) và cài đặt.
- Mở terminal chạy lệnh để tải model `qwen2.5:7b`:
```bash
ollama pull qwen2.5:7b
```

3. **Khởi chạy ứng dụng**:
```bash
streamlit run app.py
```
- Mở trình duyệt và truy cập `http://localhost:8501`.

## Hướng dẫn sử dụng
1. Ở giao diện chính, bấm "Browse files" hoặc kéo thả file PDF.
2. Đợi hệ thống xử lý nội dung văn bản (splitting, embedding, vectorizing).
3. Đặt câu hỏi trong hộp thoại phía dưới và chờ sinh câu trả lời.
