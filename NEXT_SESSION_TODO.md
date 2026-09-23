# 📌 KẾ HOẠCH & NHIỆM VỤ TIẾP TỤC SESSION SAU (NEXT SESSION ROADMAP)

> **Dự án:** Capstone Project — Hybrid Multi-Task Transformer for Stock Trend Prediction & Self-Rationalization  
> **Thời điểm cập nhật:** 23/09/2026  
> **Trạng thái session hiện tại:** Đã dừng theo yêu cầu của User để nghỉ, sẵn sàng tiếp tục ngay trong session tới.

---

## 🚀 1. Nhiệm Vụ Đang Thực Hiện Dở Dang Cần Chạy Tiếp Ngay (Priority 1)

### 🎯 Tóm tắt Tài chính Lai Toàn Bộ 133 Báo Cáo (`summarize_financial_reports.py`)
* **Mục tiêu:** Sinh các bản tóm tắt định tính chuẩn mực (600 – 800 từ) kết hợp **Số liệu thực tế Ground-Truth từ API** (Giá, % tăng trưởng, P/E, EPS, D/E, Biên lãi) + **Văn bản giải trình BCTN** (MD&A, Thuyết minh rủi ro, ESG).
* **Tiến độ đã đạt được:**
  * Đã tóm tắt hoàn thành và lưu vĩnh viễn **8/133 báo cáo** trong thư mục [`data/text/summaries/`](data/text/summaries/):
    * `ACB_2021_SUMMARY.txt`
    * `BID_2022_SUMMARY.txt`, `BID_2025_SUMMARY.txt`
    * `BVH_2024_SUMMARY.txt`
    * `CTG_2021_SUMMARY.txt`, `CTG_2022_SUMMARY.txt`, `CTG_2023_SUMMARY.txt`, `CTG_2024_SUMMARY.txt`
  * Tất cả 8 file này đều đạt chất lượng văn phong CFA / Equity Analyst xuất sắc, sạch 100% rác số ma trận kế toán.
* **Cấu hình sẵn sàng cho session sau:**
  * Đã xác thực thành công **5 Gemini API Keys** của User trong file [`.env`](.env) (đã gitignore an toàn).
  * Đã kiểm tra mô hình: Sử dụng **`gemini-3.5-flash`** (fallback sang `gemini-3.6-flash`).
* **Lệnh kích hoạt ngay khi mở session mới:**
  ```bash
  conda activate capstone
  python summarize_financial_reports.py
  ```
  *(Thời gian chạy dự kiến với 5 keys song song: khoảng 10 – 12 phút cho 125 báo cáo còn lại. Hệ thống có cơ chế cache tự động bỏ qua 8 file đã xong).*

---

## 📊 2. Hiện Trạng Tài Sản Dữ Liệu Đã Hoàn Thành (Assets Snapshot)

1. **Chuỗi dữ liệu số:**
   * [`data/processed/all_stocks_prices.csv`](data/processed/all_stocks_prices.csv) & `.parquet`: **91.908 dòng** giá OHLCV liên tục từ 01/01/2019 đến 23/09/2026 của 48 mã.
   * [`data/processed/all_stocks_ratios.csv`](data/processed/all_stocks_ratios.csv) & `.parquet`: Toàn bộ chỉ số tài chính cơ bản theo quý.
2. **Văn bản BCTN sạch (2021 – 2025):**
   * [`extract_financial_text.py`](extract_financial_text.py): Đã nâng cấp bộ lọc loại bỏ tường số kế toán vỡ layout (Digit-Wall Cleaner), nhắm trúng 6 đề mục giải mã của Thuyết minh BCTC và mở rộng từ khóa ESG toàn diện (Net-Zero, Scope 1-3, OHS, GRI, VNSI, SDGs).
   * [`data/processed/all_stocks_financial_texts.csv`](data/processed/all_stocks_financial_texts.csv) & `.parquet`: **133 báo cáo số**, tổng cộng **4.370.945 từ vựng** sạch có mật độ thông tin cao.
3. **Văn bản Quý lẻ năm hiện tại 2026 (Phương án A):**
   * [`data/processed/all_stocks_quarterly_texts.csv`](data/processed/all_stocks_quarterly_texts.csv) & `.parquet`: Toàn bộ giải trình kết quả kinh doanh Quý 1 & Quý 2/2026.

---

## 🛠️ 3. Danh Mục Việc Cần Làm Tiếp Theo (To-Do List Hậu Tóm Tắt)

### 📌 [TODO 1] Biên dịch tập tóm tắt hoàn chỉnh
* Sau khi chạy xong `summarize_financial_reports.py`:
  * Tự động sinh `data/processed/all_stocks_summaries.csv` và `all_stocks_summaries.parquet`.
  * Hợp nhất cột `summary_text` vào `all_stocks_financial_texts.parquet` để làm đầu vào trực tiếp cho Text Encoder của mô hình.

### 📌 [TODO 2] Bù đắp Ma trận Phủ kín 100% Thời gian (Bước 2 đã thỏa thuận)
* **Xử lý 59 file PDF scan ảnh:** (ACB, SSI, DGC, ITA, CII, SAB, VIB, SHB, SBT, VND...).
  * *Phương án ưu tiên:* Tải trực tiếp bản Digital PDF đẹp từ trang Quan hệ cổ đông (IR) của công ty.
  * *Phương án 2:* Chạy OCR tự động (PaddleOCR / Tesseract-OCR tiếng Việt) hoặc Google Docs OCR.
* **Bổ sung 48 báo cáo thiếu link BCTN:** Fallback sang nguồn BCTC kiểm toán năm để phủ kín 100% $N=48 \text{ mã} \times T=5 \text{ năm} = 240 \text{ báo cáo}$.

### 📌 [TODO 3] Xây dựng DataLoader & Kiến trúc Mô hình (Model Architecture)
* Tokenize chuỗi văn bản tóm tắt định tính.
* Tạo Patch chuỗi thời gian giá 30 ngày ($T_{ts}$).
* Xây dựng Shared Backbone Transformer (Qwen2.5-0.5B / LLaMA-3.2-1B) kết hợp MLP Prediction Head + Self-Rationalization Head.

---

## 🔑 4. Ghi Chú Kỹ Thuật Quan Trọng Cho Agent Mới Trong Session Tới
* **Environment:** Conda env `capstone` tại `/home/zafkiel/miniconda3/envs/capstone/bin/python`.
* **Hardware:** GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) hỗ trợ CUDA 13.x.
* **API Keys:** Đã lưu đầy đủ trong file `.env` (gồm 5 Gemini API keys hoạt động tốt).
* **Workspace:** `/home/zafkiel/Workspace/CapstoneProject`.
