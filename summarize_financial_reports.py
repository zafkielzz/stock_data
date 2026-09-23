"""
Pipeline Tóm Tắt Tài Chính Lai Đa Luồng & Song Ngữ (Multi-Key Bilingual Financial Summarizer):
Kết hợp Dữ liệu Số Thực tế (Ground-Truth API: Giá, EPS, P/E, Biên lãi, Nợ vay D/E)
với Toàn Bộ Văn Bản: Báo cáo Thường niên (MD&A), Thuyết minh BCTC (Notes), và Báo cáo Bền vững (ESG).

Tính năng:
1. Đa ngôn ngữ (Bilingual): Tự động tạo cả 2 bản tóm tắt: Tiếng Việt (SUMMARY.txt) và Tiếng Anh chuẩn mực CFA (SUMMARY_EN.txt).
2. Tự động dự phòng thông minh (Model Cascade): Tự động luân chuyển giữa gemini-3.6-flash -> gemini-3.5-flash-lite -> gemini-3.1-flash-lite khi gặp 503 hoặc hạn mức quota.
3. Chạy song song đa luồng (Worker Pool) qua 5 Gemini API Keys để hoàn thành với tốc độ cao nhất.
"""

import os
import time
import queue
import threading
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
    def __init__(self, api_keys: list = None, model_name: str = "gemini-3.6-flash"):
        if not api_keys:
            raw_keys = os.environ.get("GEMINI_API_KEYS", "")
            if raw_keys:
                api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            else:
                single_key = os.environ.get("GEMINI_API_KEY")
                if single_key:
                    api_keys = [single_key.strip()]

        if not api_keys:
            raise ValueError("Không tìm thấy GEMINI_API_KEY hoặc GEMINI_API_KEYS trong môi trường hoặc file .env!")

        self.api_keys = api_keys
        self.clients = [genai.Client(api_key=k) for k in self.api_keys]
        self.model_name = model_name
        self.models_cascade = [model_name, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
        # Loại bỏ trùng lặp giữ nguyên thứ tự
        self.unique_models = []
        for m in self.models_cascade:
            if m not in self.unique_models:
                self.unique_models.append(m)

        print(f"[*] Đã nạp thành công {len(self.clients)} Gemini Client(s) chạy song song.")
        print(f"[*] Thứ tự ưu tiên mô hình (Model Cascade): {' -> '.join(self.unique_models)}")

        # Load dữ liệu số đã xác thực làm Ground Truth
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

    def _call_gemini_with_fallback(self, client: genai.Client, prompt: str, task_desc: str = "") -> str:
        """Thực thi sinh nội dung với cơ chế fallback tự động qua danh sách mô hình."""
        for retry in range(4):
            for model_id in self.unique_models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_str = str(e).lower()
                    # Nếu model quá tải (503) hoặc chạm quota model (429), thử ngay model tiếp theo
                    if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                        continue
                    elif "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                        continue
                    else:
                        time.sleep(2)
                        continue

            # Nếu cả 4 model đều quá tải hoặc rate-limit, nghỉ ngắn rồi thử lại
            wait_time = 15 * (retry + 1)
            print(f"\n[Toàn bộ models bận tại {task_desc}] Đang chờ {wait_time}s rồi thử lại...")
            time.sleep(wait_time)

        return ""

    def summarize_single_report(self, ticker: str, year: int, client: genai.Client, text_dir: str = "data/text/extracted_text") -> str:
        """Tóm tắt 1 báo cáo tài chính sang tiếng Việt kết hợp Ground-Truth + MD&A + Notes + ESG."""
        mda_file = os.path.join(text_dir, f"{ticker}_{year}_MDA.txt")
        notes_file = os.path.join(text_dir, f"{ticker}_{year}_NOTES.txt")
        esg_file = os.path.join(text_dir, f"{ticker}_{year}_ESG.txt")

        mda_text = open(mda_file, encoding="utf-8").read().strip() if os.path.exists(mda_file) else ""
        notes_text = open(notes_file, encoding="utf-8").read().strip() if os.path.exists(notes_file) else ""
        esg_text = open(esg_file, encoding="utf-8").read().strip() if os.path.exists(esg_file) else ""

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

        return self._call_gemini_with_fallback(client, prompt, task_desc=f"{ticker} {year} (Tóm tắt VI)")

    def translate_to_english(self, vi_text: str, ticker: str, year: int, client: genai.Client) -> str:
        """Dịch bản tóm tắt tiếng Việt sang tiếng Anh học thuật chuẩn mực CFA / Equity Research."""
        prompt = f"""You are a Senior Equity Research Analyst at an institutional investment bank.
Translate the following Vietnamese financial report summary into natural, professional English suitable for institutional equity research.

REQUIREMENTS:
- Preserve all financial figures, percentages, debt metrics, and corporate names accurately.
- Retain the exact 4 numbered section headers:
  1. BUSINESS DYNAMICS & SEGMENT REPORTING
  2. CAPITAL STRUCTURE, DEBT & INTEREST RATES
  3. PROVISIONS, CONTINGENT LIABILITIES & POST-BALANCE SHEET EVENTS
  4. ESG IMPLEMENTATION
- Use sophisticated CFA-style finance terminology.
- Output ONLY the translated summary without any introductory or concluding conversational remarks.

VIETNAMESE FINANCIAL SUMMARY:
{vi_text}"""

        return self._call_gemini_with_fallback(client, prompt, task_desc=f"{ticker} {year} (Dịch EN)")

    def process_all_reports(self, text_df_path: str = "data/processed/all_stocks_financial_texts.parquet", output_dir: str = "data/text/summaries"):
        """Tóm tắt và dịch toàn bộ 133 báo cáo song song qua 5 API keys."""
        os.makedirs(output_dir, exist_ok=True)
        processed_dir = "data/processed"

        if not os.path.exists(text_df_path):
            raise FileNotFoundError(f"Không tìm thấy file {text_df_path}!")

        df_texts = pd.read_parquet(text_df_path)
        total_reports = len(df_texts)
        print(f"\n=== BẮT ĐẦU PIPELINE TÓM TẮT & DỊCH TÀI CHÍNH SONG NGỮ ({total_reports} báo cáo) ===")
        print(f"Số lượng API Keys chạy song song: {len(self.clients)}")

        # 1. Phân loại các mục cần làm
        pending_items = []
        cached_vi = 0
        cached_en = 0

        for idx, row in df_texts.iterrows():
            sym = row["ticker"]
            year = int(row["year"])
            vi_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")
            en_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")

            has_vi = os.path.exists(vi_file) and os.path.getsize(vi_file) > 400
            has_en = os.path.exists(en_file) and os.path.getsize(en_file) > 400

            if has_vi:
                cached_vi += 1
            if has_en:
                cached_en += 1

            if not has_vi or not has_en:
                pending_items.append((sym, year, not has_vi, not has_en))

        print(f"-> Hiện trạng Tiếng Việt : Đã có {cached_vi}/{total_reports} bản tóm tắt.")
        print(f"-> Hiện trạng Tiếng Anh  : Đã có {cached_en}/{total_reports} bản tóm tắt.")
        print(f"-> Số báo cáo cần xử lý tiếp: {len(pending_items)} báo cáo.")

        if pending_items:
            task_queue = queue.Queue()
            for item in pending_items:
                task_queue.put(item)

            pbar = tqdm(total=len(pending_items), desc="Tóm tắt & Dịch Song Ngữ")
            lock = threading.Lock()
            completed_count = [0]
            fail_count = [0]

            def worker_fn(worker_idx, client):
                while True:
                    try:
                        sym, year, need_vi, need_en = task_queue.get_nowait()
                    except queue.Empty:
                        break

                    vi_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")
                    en_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")
                    success = True

                    try:
                        # Bước 1: Sinh bản tiếng Việt nếu chưa có
                        vi_text = ""
                        if os.path.exists(vi_file) and os.path.getsize(vi_file) > 400:
                            vi_text = open(vi_file, encoding="utf-8").read().strip()
                        elif need_vi:
                            vi_text = self.summarize_single_report(sym, year, client=client)
                            if vi_text and len(vi_text) > 400:
                                with open(vi_file, "w", encoding="utf-8") as f:
                                    f.write(vi_text)
                            else:
                                success = False

                        # Bước 2: Dịch sang tiếng Anh nếu chưa có
                        if vi_text and need_en:
                            time.sleep(1.5)  # Giãn cách giữa 2 call
                            en_text = self.translate_to_english(vi_text, sym, year, client=client)
                            if en_text and len(en_text) > 400:
                                with open(en_file, "w", encoding="utf-8") as f:
                                    f.write(en_text)
                            else:
                                success = False

                        with lock:
                            if success:
                                completed_count[0] += 1
                            else:
                                fail_count[0] += 1
                    except Exception as e:
                        with lock:
                            fail_count[0] += 1
                            print(f"\n[Worker {worker_idx} Lỗi tại {sym} {year}]: {e}")
                    finally:
                        with lock:
                            pbar.update(1)
                        task_queue.task_done()
                        # Rate limit delay per worker
                        time.sleep(3.5)

            threads = []
            num_workers = min(len(self.clients), len(pending_items))
            for i in range(num_workers):
                t = threading.Thread(target=worker_fn, args=(i + 1, self.clients[i]), daemon=True)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            pbar.close()
            print(f"\n=> KẾT QUẢ ĐỢT CHẠY: Thành công {completed_count[0]}/{len(pending_items)} báo cáo (Lỗi: {fail_count[0]}).")

        # 2. Biên tập dataset song ngữ tổng hợp
        print("\n=== ĐANG BIÊN TẬP DATASET TỔNG HỢP SONG NGỮ ===")
        summaries_records = []
        for idx, row in df_texts.iterrows():
            sym = row["ticker"]
            year = int(row["year"])
            vi_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")
            en_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")

            vi_text = open(vi_file, encoding="utf-8").read().strip() if os.path.exists(vi_file) and os.path.getsize(vi_file) > 400 else ""
            en_text = open(en_file, encoding="utf-8").read().strip() if os.path.exists(en_file) and os.path.getsize(en_file) > 400 else ""

            if vi_text or en_text:
                summaries_records.append({
                    "ticker": sym,
                    "year": year,
                    "summary_text": vi_text,
                    "summary_words": len(vi_text.split()) if vi_text else 0,
                    "summary_text_en": en_text,
                    "summary_words_en": len(en_text.split()) if en_text else 0,
                    "summary_path_vi": vi_file if vi_text else "",
                    "summary_path_en": en_file if en_text else ""
                })

        if summaries_records:
            df_sum = pd.DataFrame(summaries_records)
            sum_csv = os.path.join(processed_dir, "all_stocks_summaries.csv")
            sum_parquet = os.path.join(processed_dir, "all_stocks_summaries.parquet")
            df_sum.to_csv(sum_csv, index=False)
            df_sum.to_parquet(sum_parquet, index=False)
            print(f"   [✓] File CSV:     {sum_csv} ({os.path.getsize(sum_csv)/1024/1024:.2f} MB)")
            print(f"   [✓] File Parquet: {sum_parquet} ({os.path.getsize(sum_parquet)/1024/1024:.2f} MB)")
            print(f"   [✓] Số bản ghi có tóm tắt: {len(df_sum)}/{total_reports}")
            print(f"   [✓] Bản Tiếng Việt: {sum(df_sum['summary_words'] > 0)}/{total_reports} (Tổng {df_sum['summary_words'].sum():,} từ)")
            print(f"   [✓] Bản Tiếng Anh : {sum(df_sum['summary_words_en'] > 0)}/{total_reports} (Tổng {df_sum['summary_words_en'].sum():,} từ)")

            # Hợp nhất vào all_stocks_financial_texts
            cols_to_merge = ["ticker", "year", "summary_text", "summary_words", "summary_text_en", "summary_words_en"]
            df_merged = df_texts.merge(df_sum[cols_to_merge], on=["ticker", "year"], how="left")
            merged_csv = os.path.join(processed_dir, "all_stocks_financial_texts.csv")
            merged_parquet = os.path.join(processed_dir, "all_stocks_financial_texts.parquet")
            df_merged.to_csv(merged_csv, index=False)
            df_merged.to_parquet(merged_parquet, index=False)
            print(f"   [✓] Đã cập nhật song ngữ vào {merged_parquet} và {merged_csv}!")

        return summaries_records


if __name__ == "__main__":
    summarizer = HybridFinancialSummarizer()
    summarizer.process_all_reports()
