# Đồ Án Tốt Nghiệp: Hybrid Multi-Task Transformer for Stock Prediction & Self-Rationalization

> **Đề tài:** Nghiên cứu và hiện thực kiến trúc Decoder-Only Transformer đa nhiệm xử lý dữ liệu lai (Chuỗi thời gian giá – Báo cáo tài chính – ESG) cho dự đoán xu hướng cổ phiếu và tự động sinh giải trình.  
> **English Title:** *A Multi-Task Decoder-Only Transformer for Hybrid Time-Series, Financial-Text, and ESG Data in Stock Trend Prediction and Self-Rationalization.*

---

## 📌 1. Tổng quan Kiến trúc Đề tài

Hệ thống xây dựng theo mô hình Deep Learning đa phương thức (Multi-Modal) hợp nhất 3 nguồn dữ liệu không đồng nhất vào một chuỗi đầu vào thống nhất:

$$\mathbf{H}_0 = [\mathbf{T}_{ts} \; ; \; \mathbf{E}_{text} \; ; \; \mathbf{E}_{esg} \; ; \; [\text{DECISION}]]$$

1. **Chuỗi thời gian giá ($\mathbf{T}_{ts}$):** Chuỗi giá lịch sử OHLCV 30 ngày (patch 30 ngày), chiếu qua Linear Projection kết hợp Temporal Positional Embedding.
2. **Văn bản tài chính ($\mathbf{E}_{text}$):** Bản Thuyết minh Báo cáo tài chính (Notes to Financial Statements) kết hợp Báo cáo của Ban Điều hành (MD&A), mã hóa qua Tokenizer chuyên biệt và Segment Embedding.
3. **Chỉ số ESG ($\mathbf{E}_{esg}$):** Điểm số phát triển bền vững liên tục (Continuous Feature Embedding) kết hợp văn bản thuyết minh ESG.
4. **Shared Backbone:** Mô hình Decoder-Only cỡ nhỏ (**Qwen2.5-0.5B** làm trọng tâm, **LLaMA-3.2-1B** làm đối chứng) tối ưu hóa qua QLoRA 4-bit và Gradient Checkpointing.
5. **Multi-Task Output Heads:**
   * **Prediction Head (MLP):** Dự đoán xu hướng giá Tăng/Giảm (Up/Down sau 5 ngày) kèm độ tin cậy.
   * **Explanation Head (Language Head):** Tự động sinh văn bản lý giải nguyên nhân tài chính (Self-Rationalization).
   * **Consistency Loss:** Phạt mâu thuẫn giữa nhãn dự đoán và nội dung giải trình bằng mô hình NLI (DeBERTa-MNLI).

---

## 📂 2. Cấu trúc Thư mục & Dữ liệu Dự án

```
CapstoneProject/
├── stocks_universe.json             # Danh mục 48 cổ phiếu thực nghiệm (Point-in-time & Anti-Survivorship Bias)
├── Doc Idea.docx                    # Bản thuyết minh chi tiết ý tưởng đồ án
│
├── download_market_data.py          # Script tự động tải giá OHLCV 5 năm và Chỉ số BCTC từ Vnstock
├── extract_financial_text.py        # Module thuật toán Page-Scoring tự động bóc tách MD&A, Thuyết minh BCTC, ESG
├── download_and_extract_reports.py  # Script tải BCTN và bóc tách tự động cho toàn bộ danh mục cổ phiếu
├── extract_bctc_sample.py          # Script mẫu kiểm tra kết nối Vnstock và nhận diện PDF scan/digital
│
└── data/
    ├── processed/                   # ⭐ CÁC FILE DỮ LIỆU TỔNG HỢP (DÙNG ĐỂ GỬI BẠN BÈ / LOAD MODEL)
    │   ├── all_stocks_prices.csv            # 71.886 dòng giá OHLCV 5 năm (3.8 MB) -> Gửi import DB
    │   ├── all_stocks_prices.parquet        # Bản nén đọc siêu tốc cho PyTorch DataLoader (1.2 MB)
    │   ├── all_stocks_ratios.csv            # Chỉ số tài chính quý (P/E, EPS, ROE, Margin...) (237 KB)
    │   ├── all_stocks_ratios.parquet        # Bản Parquet chỉ số BCTC (55 KB)
    │   ├── all_stocks_financial_texts.csv   # Toàn bộ văn bản MD&A + THUYẾT MINH BCTC + ESG (>1.28M từ, 7.36 MB)
    │   ├── all_stocks_financial_texts.parquet # Bản Parquet văn bản tài chính (3.24 MB)
    │   └── all_stocks_reports_meta.csv      # Metadata trạng thái thu thập tài liệu từng mã
    │
    ├── raw/                         # Dữ liệu thô từng mã riêng lẻ
    │   ├── prices/                  # Từng file CSV giá OHLCV riêng của 48 mã (ACB.csv, FPT.csv, HPG.csv...)
    │   └── ratios/                  # Từng file CSV chỉ số tài chính quý riêng của 48 mã
    │
    └── text/                        # Dữ liệu văn bản bóc tách
        ├── raw_pdf/                 # File PDF Báo cáo thường niên tải về (đã gitignore vì dung lượng lớn)
        └── extracted_text/          # Các file .txt text thuần đã bóc tách:
            ├── {TICKER}_2023_MDA.txt    # Báo cáo đánh giá của Ban Điều hành (MD&A)
            ├── {TICKER}_2023_NOTES.txt  # Bản Thuyết minh Báo cáo tài chính (Segment, Vay nợ, Tồn kho...)
            └── {TICKER}_2023_ESG.txt    # Báo cáo Phát triển bền vững / ESG
```

---

## 📊 3. Chi tiết các File Dữ liệu Tổng hợp (`data/processed/`)

### 3.1. `all_stocks_prices.csv` (Giá OHLCV hàng ngày)
* **Số lượng bản ghi:** 71.886 dòng (từ 01/01/2019 đến 31/12/2024).
* **Cấu trúc cột:**
  * `time`: Ngày giao dịch (`YYYY-MM-DD 07:00:00`)
  * `open`, `high`, `low`, `close`: Giá mở cửa, cao nhất, thấp nhất, đóng cửa (đã điều chỉnh chia cổ tức).
  * `volume`: Khối lượng giao dịch khớp lệnh.
  * `ticker`: Mã cổ phiếu (`VCB`, `FPT`, `HPG`, `NVL`, `HAG`...).

### 3.2. `all_stocks_ratios.csv` (Chỉ số tài chính Quý)
* **Số lượng:** Đầy đủ cho 48 mã theo từng quý.
* **Các chỉ tiêu chính:** P/E, P/B, EPS, ROE, ROA, Net Margin, Biên lợi nhuận gộp, Hệ số thanh toán ngắn hạn, Tỷ lệ Nợ/Vốn chủ sở hữu (D/E)...

### 3.3. `all_stocks_financial_texts.csv` (Văn bản Tài chính Đa thành phần)
* **Quy mô:** Hơn **1.280.000 từ vựng** text thuần tiếng Việt chuẩn Unicode UTF-8 từ 24 tập đoàn kinh tế lớn.
* **Cấu trúc cột:**
  * `ticker`: Mã cổ phiếu
  * `year`: Năm báo cáo (`2023`)
  * `notes_text`: **Toàn văn Bản Thuyết minh Báo cáo tài chính** (Báo cáo bộ phận mảng kinh doanh, chi tiết vay nợ ngân hàng, dự phòng giảm giá hàng tồn kho, nợ xấu, nợ tiềm tàng). Tổng: **684.622 từ**.
  * `mda_text`: **Báo cáo của Ban Điều hành / Ban Tổng Giám đốc** (Đánh giá kết quả kinh doanh, phân tích nguyên nhân biến động, bối cảnh vĩ mô, rủi ro và định hướng). Tổng: **427.272 từ**.
  * `esg_text`: **Báo cáo ESG** (Môi trường, xã hội, quản trị, phát thải khí nhà kính). Tổng: **168.000+ từ**.
  * `notes_words`, `mda_words`, `esg_words`: Thống kê số lượng từ.

---

## 🚀 4. Hướng dẫn Import dữ liệu vào Database

### Cách 1: Nạp nhanh bằng Python (Hỗ trợ PostgreSQL, MySQL, SQLite)
Chạy script sau để tạo bảng và import toàn bộ chỉ trong vài giây:

```python
import pandas as pd
from sqlalchemy import create_engine

# Kết nối Database (thay đổi thông tin kết nối phù hợp)
engine = create_engine("postgresql://username:password@localhost:5432/stock_db")

# 1. Import bảng Giá OHLCV
df_prices = pd.read_csv("data/processed/all_stocks_prices.csv")
df_prices.to_sql("stock_prices", engine, if_exists="replace", index=False)
print("Import xong bảng stock_prices!")

# 2. Import bảng Văn bản tài chính
df_texts = pd.read_csv("data/processed/all_stocks_financial_texts.csv")
df_texts.to_sql("financial_texts", engine, if_exists="replace", index=False)
print("Import xong bảng financial_texts!")
```

### Cách 2: Lệnh SQL trực tiếp (PostgreSQL)
```sql
CREATE TABLE stock_prices (
    time TIMESTAMP,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    ticker VARCHAR(10)
);

COPY stock_prices FROM '/path/to/data/processed/all_stocks_prices.csv' WITH (FORMAT csv, HEADER true);
```

---

## 🛠️ 5. Checklist Những Thứ Còn Thiếu Cần Làm Thủ Công / Bổ Sung

Dưới đây là các phần nhóm cần phối hợp hoàn thiện thủ công để bộ dữ liệu đạt 100% độ phủ:

### 1. Bổ sung Văn bản cho các mã bị "Scan PDF" (ACB, BID, SSI, VNM, DGC, FRT...)
* **Thực trạng:** Trong 48 mã, khoảng 15–20 mã khi nộp tài liệu lên cổng thông tin là bản scan đóng dấu mộc đỏ (ảnh chụp), nên thuật toán bóc text layer tự động không lấy được chữ (file text ra 0 KB).
* **2 Cách giải quyết:**
  * **Cách A (Nhanh và Chuẩn nhất - KHÔNG CẦN OCR):**
    1. Truy cập trực tiếp vào mục **Quan hệ Cổ đông (Investor Relations)** trên website chính thức của công ty (Ví dụ: `acb.com.vn`, `bidv.com.vn`, `vinamilk.com.vn`, `ssi.com.vn`).
    2. Các tập đoàn này trên trang chủ luôn phát hành bản **Digital PDF chất lượng cao (hoặc file Word)** cho cổ đông và quỹ ngoại.
    3. Mở file, copy phần *"Báo cáo của Ban Giám đốc"* và *"Thuyết minh BCTC"* rồi dán tương ứng vào:
       * `data/text/extracted_text/{TICKER}_2023_MDA.txt`
       * `data/text/extracted_text/{TICKER}_2023_NOTES.txt`
  * **Cách B (Dùng OCR):**
    Nếu bắt buộc dùng file scan:
    * Dùng công cụ **VietOCR** hoặc tải file PDF lên **Google Drive -> Chuột phải -> Mở bằng Google Tài liệu (Google Docs)** để Google tự động OCR tiếng Việt miễn phí và cực kỳ chuẩn xác.
    * Sau đó copy phần text thu được vào thư mục `data/text/extracted_text/`.

### 2. Nhập điểm số định lượng ESG (E_score, S_score, G_score)
* **Thực trạng:** Đề tài đã có văn bản thuyết minh ESG (`esg_text`), nhưng nhánh toán học $E_{esg}$ trong đồ án cần 3 số thực: Điểm E, Điểm S, Điểm G (thang điểm 0–100).
* **Cách thực hiện:**
  1. Tạo file CSV: `data/processed/esg_numerical_scores.csv`.
  2. Tra cứu điểm số công bố hàng năm trong rổ **VNSI (Chỉ số Phát triển Bền vững của Sở GDCK TP.HCM - HOSE)** hoặc bảng đánh giá phát triển bền vững trong Báo cáo thường niên của công ty.
  3. Cấu trúc bảng gồm 5 cột: `ticker`, `year`, `e_score`, `s_score`, `g_score`.

### 3. Mở rộng dải năm (2021, 2022) nếu muốn thêm dữ liệu huấn luyện
* Hiện tại dữ liệu giá OHLCV đã có trọn vẹn **5 năm (2019 – 2024)**. Dữ liệu văn bản đã có đầy đủ cho năm **2023**.
* Nếu nhóm muốn mở rộng thêm văn bản các năm 2021 (thời kỳ Uptrend) và 2022 (thời kỳ Downtrend), chỉ cần chạy:
  ```bash
  python download_and_extract_reports.py
  ```
  *(Chỉnh sửa tham số `years=[2023, 2022, 2021]` trong script để tự động tải thêm).*

---

## 💻 6. Môi trường & Cách chạy Scripts

Môi trường khuyến nghị:
```bash
conda activate capstone
```

* **Tải lại dữ liệu giá & chỉ số tài chính:**
  ```bash
  python download_market_data.py
  ```
* **Tải và bóc tách Báo cáo thường niên (MD&A, Thuyết minh, ESG):**
  ```bash
  python download_and_extract_reports.py
  ```
* **Kiểm tra trích xuất text 1 file PDF bất kỳ:**
  ```bash
  python extract_financial_text.py
  ```
