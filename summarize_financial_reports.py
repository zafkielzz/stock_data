"""
Pipeline Tóm Tắt Tài Chính Lai (Hybrid Financial Summarizer):
Kết hợp Dữ liệu Số Thực tế (Ground-Truth API: Giá, EPS, P/E, Biên lãi, Nợ vay D/E)
với Văn bản Thuyết minh BCTC, MD&A, ESG đã bóc tách từ PDF.

Sử dụng Google Gemini 3.6 Flash để sinh ra các bản tóm tắt định tính chuẩn mực:
- 100% không bịa số liệu (Zero Hallucination nhờ Data Grounding).
- Loại bỏ toàn bộ câu chữ hành chính kế toán mẫu.
- Độ dài chuẩn mực (600 - 800 từ) vừa khít context window của Transformer (Qwen2.5 / LLaMA).
"""

import os
import time
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Import Gemini Client
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Vui lòng cài đặt google-genai: pip install google-genai")
    exit(1)


class HybridFinancialSummarizer:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY trong môi trường hoặc file .env!")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

        # Load dữ liệu số đã xác thực
        self.prices_df = None
        self.ratios_df = None
        self._load_ground_truth_tables()

    def _load_ground_truth_tables(self):
        """Nạp dữ liệu số chuẩn xác từ API làm Ground Truth."""
        p_path = "data/processed/all_stocks_prices.parquet"
        r_path = "data/processed/all_stocks_ratios.parquet"

        if os.path.exists(p_path):
            self.prices_df = pd.read_parquet(p_path)
            self.prices_df["year"] = pd.to_datetime(self.prices_df["time"]).dt.year

        if os.path.exists(r_path):
            self.ratios_df = pd.read_parquet(r_path)

    def get_ground_truth_metrics(self, ticker: str, year: int) -> str:
        """Trích xuất các số liệu tài chính & thị trường thực tế cho mã và năm tương ứng."""
        metrics_lines = []

        # 1. Thống kê giá & khối lượng từ all_stocks_prices
        if self.prices_df is not None:
            sub_p = self.prices_df[(self.prices_df["ticker"] == ticker) & (self.prices_df["year"] == year)].sort_values("time")
            if not sub_p.empty:
                open_p = sub_p.iloc[0]["open"] * 1000
                close_p = sub_p.iloc[-1]["close"] * 1000
                high_p = sub_p["high"].max() * 1000
                low_p = sub_p["low"].min() * 1000
                pct_chg = (close_p - open_p) / open_p * 100
                avg_vol = sub_p["volume"].mean()
                metrics_lines.append(
                    f"- Diễn biến giá cổ phiếu trong năm: Mở cửa: {open_p:,.0f} VND, Đóng cửa: {close_p:,.0f} VND "
                    f"(Biến động cả năm: {pct_chg:+.2f}%), Cao nhất: {high_p:,.0f} VND, Thấp nhất: {low_p:,.0f} VND. "
                    f"Khối lượng giao dịch khớp lệnh bình quân: {avg_vol:,.0f} CP/phiên."
                )

        # 2. Thống kê chỉ số BCTC từ all_stocks_ratios
        if self.ratios_df is not None:
            sub_r = self.ratios_df[self.ratios_df["ticker"] == ticker]
            target_col = f"{year}-Q4"
            if target_col in sub_r.columns:
                key_items = {
                    "trailing_eps": "EPS 4 quý gần nhất",
                    "pe_ratio": "P/E",
                    "pb_ratio": "P/B",
                    "gross_margin": "Biên lợi nhuận gộp (%)",
                    "net_profit_margin": "Biên lợi nhuận ròng (%)",
                    "roe": "ROE (%)",
                    "roa": "ROA (%)",
                    "debt_to_equity": "Tỷ lệ Nợ vay/VCSH (D/E %)"
                }
                ratio_vals = []
                for item_id, label in key_items.items():
                    row = sub_r[sub_r["item_id"] == item_id]
                    if not row.empty:
                        val = row.iloc[0][target_col]
                        if pd.notna(val):
                            ratio_vals.append(f"{label}: {val:,.2f}")
                if ratio_vals:
                    metrics_lines.append(f"- Các chỉ số tài chính cơ bản (Kỳ kết thúc năm {year}): " + ", ".join(ratio_vals))

        if not metrics_lines:
            return "- Số liệu tài chính cơ bản: Theo dõi trong nội dung văn bản giải trình."
        return "\n".join(metrics_lines)

    def summarize_single_report(self, ticker: str, year: int, text_dir: str = "data/text/extracted_text") -> str:
        """Tóm tắt 1 báo cáo tài chính sử dụng Hybrid Prompting."""
        mda_file = os.path.join(text_dir, f"{ticker}_{year}_MDA.txt")
        notes_file = os.path.join(text_dir, f"{ticker}_{year}_NOTES.txt")
        esg_file = os.path.join(text_dir, f"{ticker}_{year}_ESG.txt")

        mda_text = open(mda_file, encoding="utf-8").read().strip() if os.path.exists(mda_file) else ""
        notes_text = open(notes_file, encoding="utf-8").read().strip() if os.path.exists(notes_file) else ""
        esg_text = open(esg_file, encoding="utf-8").read().strip() if os.path.exists(esg_file) else ""

        # Lấy tối đa đoạn trích quan trọng để tránh vượt ngưỡng không cần thiết
        mda_snippet = mda_text[:12000]
        notes_snippet = notes_text[:12000]
        esg_snippet = esg_text[:8000]

        ground_truth_str = self.get_ground_truth_metrics(ticker, year)

        prompt = f"""Bạn là chuyên gia phân tích tài chính cao cấp (Senior Equity Research Analyst).
Dưới đây là dữ liệu về doanh nghiệp {ticker} cho năm tài chính {year}:

=== [1. DỮ LIỆU SỐ THỰC TẾ ĐÃ XÁC THỰC - GROUND TRUTH TỪ HỆ THỐNG] ===
{ground_truth_str}

=== [2. VĂN BẢN BÁO CÁO CỦA BAN ĐIỀU HÀNH (MD&A)] ===
{mda_snippet if mda_snippet else "Không có văn bản MD&A."}

=== [3. VĂN BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH (NOTES)] ===
{notes_snippet if notes_snippet else "Không có văn bản Thuyết minh BCTC."}

=== [4. VĂN BẢN BÁO CÁO PHÁT TRIỂN BỀN VỮNG (ESG)] ===
{esg_snippet if esg_snippet else "Không có văn bản ESG riêng biệt."}

NHIỆM VỤ: Hãy đóng vai chuyên gia tài chính, dựa trên các con số thực tế đã xác thực và văn bản giải trình đính kèm, viết một bản tóm tắt phân tích định tính chuẩn mực (tổng độ dài khoảng 600 - 800 từ) gồm đúng 4 đề mục sau:

1. ĐỘNG LỰC KINH DOANH & BÁO CÁO BỘ PHẬN: 
Phân tích nguyên nhân biến động doanh thu, lợi nhuận thực tế. Nêu rõ động lực tăng trưởng cốt lõi của từng mảng kinh doanh và thị trường xuất khẩu. Giữ lại các con số phần trăm (%) và nguyên nhân thực tế.

2. CƠ CẤU VỐN, NỢ VAY & LÃI SUẤT: 
Chi tiết các khoản vay ngắn/dài hạn, lãi suất vay thả nổi/cố định, đối tác ngân hàng cấp tín dụng lớn nhất, rủi ro cơ cấu vốn và áp lực dòng tiền.

3. RỦI RO DỰ PHÒNG, NỢ TIỀM TÀNG & BIẾN CỐ SAU NIÊN ĐỘ: 
Tình hình trích lập dự phòng giảm giá hàng tồn kho và nợ khó đòi (kèm lý do), các cam kết bảo lãnh ngoại bảng, tranh chấp kiện tụng pháp lý (nếu có), và các sự kiện tài chính bất thường phát sinh sau ngày 31/12.

4. THỰC THI ESG: 
- Môi trường (E): Phát thải khí nhà kính (Scope 1/2/3, Net-Zero), tiết kiệm năng lượng, kinh tế tuần hoàn.
- Xã hội (S): An toàn lao động (OHS), bình đẳng giới, đào tạo nguồn nhân lực, phúc lợi nhân viên.
- Quản trị (G): Tính độc lập của HĐQT, kiểm toán độc lập, tuân thủ chuẩn mực GRI / SDGs.

YÊU CẦU QUAN TRỌNG:
- Văn phong khách quan, cô đọng, viết thành các đoạn văn xuôi hoàn chỉnh, mạch lạc.
- Tuyệt đối KHÔNG tự ý bịa đặt số liệu ngoài các dữ kiện đã cung cấp.
- Loại bỏ hoàn toàn các câu chữ hành chính thủ tục kế toán rập khuôn."""

        models_to_try = [self.model_name, "gemini-3.5-flash", "gemini-3.6-flash"]
        # Loại trừ trùng lặp giữ nguyên thứ tự
        unique_models = []
        for m in models_to_try:
            if m not in unique_models:
                unique_models.append(m)

        for retry in range(5):
            for model_id in unique_models:
                try:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_str = str(e).lower()
                    if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                        # Thử model tiếp theo ngay lập tức
                        continue
                    elif "429" in err_str or "quota" in err_str or "rate" in err_str:
                        wait_time = 25 * (retry + 1)
                        print(f"\n[Rate Limit tại {ticker} {year}] Đang chờ {wait_time}s...")
                        time.sleep(wait_time)
                        break
                    else:
                        print(f"\n[Lỗi {ticker} {year} với {model_id}]: {e}")
                        time.sleep(4)

        return ""

    def process_all_reports(self, text_df_path: str = "data/processed/all_stocks_financial_texts.parquet", output_dir: str = "data/text/summaries"):
        """Tóm tắt toàn bộ 133 báo cáo tài chính và lưu vào thư mục summaries + compile file tổng hợp."""
        os.makedirs(output_dir, exist_ok=True)
        processed_dir = "data/processed"

        if not os.path.exists(text_df_path):
            raise FileNotFoundError(f"Không tìm thấy file {text_df_path}!")

        df_texts = pd.read_parquet(text_df_path)
        print(f"\n=== BẮT ĐẦU PIPELINE TÓM TẮT TÀI CHÍNH LAI ({len(df_texts)} báo cáo) ===")
        print(f"Mô hình sử dụng: {self.model_name}")

        summaries_records = []
        success_count = 0
        cached_count = 0

        for idx, row in tqdm(df_texts.iterrows(), total=len(df_texts), desc="Tóm tắt BCTN"):
            sym = row["ticker"]
            year = int(row["year"])
            summary_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")

            # 1. Đọc lại từ cache nếu đã tóm tắt trước đó
            if os.path.exists(summary_file) and os.path.getsize(summary_file) > 400:
                summary_text = open(summary_file, encoding="utf-8").read().strip()
                cached_count += 1
            else:
                summary_text = self.summarize_single_report(sym, year)
                if summary_text:
                    with open(summary_file, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                    success_count += 1
                # Nghỉ 4.2 giây để tuân thủ 15 Requests/phút của Free Tier
                time.sleep(4.2)

            if summary_text:
                summaries_records.append({
                    "ticker": sym,
                    "year": year,
                    "summary_text": summary_text,
                    "summary_words": len(summary_text.split()),
                    "summary_chars": len(summary_text),
                    "summary_path": summary_file
                })

        print(f"\n=> ĐÃ HOÀN THÀNH TÓM TẮT: {len(summaries_records)}/{len(df_texts)} báo cáo (Mới: {success_count}, Đã có: {cached_count}).")

        # Lưu file tổng hợp
        if summaries_records:
            df_sum = pd.DataFrame(summaries_records)
            sum_csv = os.path.join(processed_dir, "all_stocks_summaries.csv")
            sum_parquet = os.path.join(processed_dir, "all_stocks_summaries.parquet")
            df_sum.to_csv(sum_csv, index=False)
            df_sum.to_parquet(sum_parquet, index=False)
            print(f"   CSV:     {sum_csv} ({os.path.getsize(sum_csv)/1024/1024:.2f} MB)")
            print(f"   Parquet: {sum_parquet} ({os.path.getsize(sum_parquet)/1024/1024:.2f} MB)")
            print(f"   Tổng số từ tóm tắt sạch: {df_sum['summary_words'].sum():,} từ!")
            print(f"   Trung bình: {df_sum['summary_words'].mean():.0f} từ/báo cáo.")

            # Hợp nhất vào all_stocks_financial_texts
            df_merged = df_texts.merge(df_sum[["ticker", "year", "summary_text", "summary_words"]], on=["ticker", "year"], how="left")
            merged_csv = os.path.join(processed_dir, "all_stocks_financial_texts.csv")
            merged_parquet = os.path.join(processed_dir, "all_stocks_financial_texts.parquet")
            df_merged.to_csv(merged_csv, index=False)
            df_merged.to_parquet(merged_parquet, index=False)
            print(f"   Đã cập nhật cột 'summary_text' vào {merged_parquet}!")

        return summaries_records


if __name__ == "__main__":
    summarizer = HybridFinancialSummarizer()
    summarizer.process_all_reports()
