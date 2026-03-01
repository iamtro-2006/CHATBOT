# chatbot-teaching-en

## Yêu cầu
- Python 3.10+
- Node.js 18+
- MongoDB (hoặc MongoDB Atlas)

## Cài đặt backend
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Cấu hình môi trường
Tạo file `.env` ở thư mục gốc (không commit):
```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=english_coach_db

# Vertex AI (nếu dùng Gemini)
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Chạy backend
```bash
python api.py
```
Backend chạy ở `http://localhost:8000`.

## Cài đặt frontend
```bash
cd fe-source
npm install
```

## Chạy frontend
```bash
npm run dev
```
Frontend chạy ở `http://localhost:5173`.

## Ghi chú
- Login/Signup gọi API: `/login`, `/signup`.
- Chat gọi API: `/chat`.
- Kiểm tra tiến độ: `/progress`, `/daily-status`.
