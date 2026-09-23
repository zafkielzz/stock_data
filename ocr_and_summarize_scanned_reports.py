"""
Pipeline OCR & Tóm Tắt Song Ngữ Cho 59 Báo Cáo PDF Scan (Multimodal Scanned Report Processor):
Bù đắp ma trận thời gian cho các doanh nghiệp có BCTN là file scan ảnh thuần (ACB, SSI, VHM, GAS, VNM, MWG, SAB...).

Quy trình:
1. Đọc trực tiếp file PDF scan trong data/text/raw_pdf/ qua Gemini File API (sử dụng khả năng thị giác OCR tài liệu bản địa của Gemini).
2. Kết hợp với Dữ liệu số thực tế (Ground-Truth API: Giá OHLCV, Biên độ, Khối lượng, EPS, P/E, Margin, D/E).
3. Sinh bản tóm tắt định tính tiếng Việt chuẩn CFA (SUMMARY.txt) và dịch sang tiếng Anh chuẩn Wall Street (SUMMARY_EN.txt).
4. Phân tải song song qua 5 Gemini API Keys với Model Cascade tự động luân chuyển.
5. Tự động hợp nhất vào all_stocks_summaries.parquet (nâng quy mô từ 133 lên 192 báo cáo!).
"""

import os
import time
import queue
import threading
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Vui lòng cài đặt google-genai: pip install google-genai")
    exit(1)


class ScannedFinancialReportSummarizer:
    def __init__(self, api_keys: list = None, model_name: str = "gemini-3.5-flash-lite"):
        if not api_keys:
            raw_keys = os.environ.get("GEMINI_API_KEYS", "")
            if raw_keys:
                api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            else:
                single_key = os.environ.get("GEMINI_API_KEY")
                if single_key:
                    api_keys = [single_key.strip()]

        if not api_keys:
            raise ValueError("Không tìm thấy GEMINI_API_KEY hoặc GEMINI_API_KEYS!")

        self.api_keys = api_keys
        self.clients = [genai.Client(api_key=k) for k in self.api_keys]
        self.model_name = model_name
        self.models_cascade = [model_name, "gemini-3.6-flash", "gemini-3.1-flash-lite"]
        self.unique_models = []
        for m in self.models_cascade:
            if m not in self.unique_models:
                self.unique_models.append(m)

        print(f"[*] Đã nạp {len(self.clients)} Gemini Client(s) cho pipeline OCR PDF Scan.")
        print(f"[*] Model Cascade: {' -> '.join(self.unique_models)}")

        # Nạp Ground Truth
        self.prices_df = None
        self.ratios_df = None
        self._load_ground_truth_tables()

    def _load_ground_truth_tables(self):
        p_path = "data/processed/all_stocks_prices.parquet"
        r_path = "data/processed/all_stocks_ratios.parquet"
        if os.path.exists(p_path):
            self.prices_df = pd.read_parquet(p_path)
            self.prices_df["year"] = pd.to_datetime(self.prices_df["time"]).dt.year
        if os.path.exists(r_path):
            self.ratios_df = pd.read_parquet(r_path)

    def get_ground_truth_metrics(self, ticker: str, year: int) -> str:
        metrics_lines = []
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
                    metrics_lines.append(f"- Các chỉ số tài chính cơ bản: {', '.join(ratio_vals)}")

        return "\n".join(metrics_lines) if metrics_lines else "- Số liệu cơ bản: Theo dõi trong nội dung BCTN."

    def find_scanned_pdf_path(self, ticker: str, year: int, pdf_dir: str = "data/text/raw_pdf") -> str:
        candidates = [
            os.path.join(pdf_dir, f"{ticker}_BCTN_{year}.pdf"),
            os.path.join(pdf_dir, f"{ticker}_Baocaothuongnien_{year}.pdf"),
            os.path.join(pdf_dir, f"{ticker}_Baocao_Thuongnien_{year}.pdf"),
            os.path.join(pdf_dir, f"{ticker}_BCTN_{year}_VN.pdf"),
            os.path.join(pdf_dir, f"{ticker}_Baocaothuongnien_{year}_VN.pdf"),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 10000:
                return c

        if os.path.exists(pdf_dir):
            for fname in os.listdir(pdf_dir):
                if fname.startswith(f"{ticker}_") and str(year) in fname and fname.endswith(".pdf"):
                    p = os.path.join(pdf_dir, fname)
                    if os.path.getsize(p) > 10000:
                        return p
        return ""

    def process_single_scanned_pdf(self, ticker: str, year: int, pdf_path: str, client: genai.Client) -> tuple:
        """Tải PDF scan lên Gemini, sinh tóm tắt tiếng Việt và tiếng Anh kết hợp Ground-Truth."""
        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            gt_str = self.get_ground_truth_metrics(ticker, year)

            prompt_vi = f"""Bạn là chuyên gia phân tích tài chính cao cấp (Senior Equity Research Analyst).
Dưới đây là Báo cáo Thường niên (tài liệu PDF scan đính kèm) và Số liệu thực tế đã xác thực của doanh nghiệp {ticker} cho năm {year}:

=== [DỮ LIỆU SỐ THỰC TẾ GROUND TRUTH TỪ HỆ THỐNG] ===
{gt_str}

NHIỆM VỤ: Hãy đọc toàn bộ tài liệu BCTN scan đính kèm và dựa trên các con số thực tế đã xác thực, viết một bản tóm tắt phân tích định tính chuẩn mực (tổng độ dài khoảng 600 - 800 từ) gồm đúng 4 đề mục sau:

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

            # Call model với fallback
            vi_summary = ""
            for model_id in self.unique_models:
                try:
                    res = client.models.generate_content(
                        model=model_id,
                        contents=[uploaded_file, prompt_vi]
                    )
                    if res and res.text:
                        vi_summary = res.text.strip()
                        break
                except Exception as e:
                    time.sleep(2)
                    continue

            if not vi_summary or len(vi_summary) < 300:
                return "", ""

            # Dịch sang tiếng Anh
            time.sleep(1.5)
            prompt_en = f"""You are a Senior Equity Research Analyst at an institutional investment bank.
Translate the following Vietnamese financial report summary into natural, professional English suitable for institutional equity research. Preserve all financial figures, percentages, debt metrics, corporate names, and the exact 4 numbered section headers.

VIETNAMESE FINANCIAL SUMMARY:
{vi_summary}"""

            en_summary = ""
            for model_id in self.unique_models:
                try:
                    res_en = client.models.generate_content(
                        model=model_id,
                        contents=prompt_en
                    )
                    if res_en and res_en.text:
                        en_summary = res_en.text.strip()
                        break
                except Exception as e:
                    time.sleep(2)
                    continue

            return vi_summary, en_summary

        finally:
            # Luôn dọn dẹp file upload trên Gemini
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    def run(self, output_dir: str = "data/text/summaries", processed_dir: str = "data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(processed_dir, "all_stocks_reports_meta.parquet")
        ft_path = os.path.join(processed_dir, "all_stocks_financial_texts.parquet")

        df_meta = pd.read_parquet(meta_path)
        df_ext = pd.read_parquet(ft_path)

        meta_keys = set(zip(df_meta["ticker"], df_meta["year"]))
        ext_keys = set(zip(df_ext["ticker"], df_ext["year"]))
        scanned_keys = sorted(list(meta_keys - ext_keys))

        print(f"\n=== BẮT ĐẦU PIPELINE OCR & TÓM TẮT SONG NGỮ CHO {len(scanned_keys)} BÁO CÁO SCAN ===")
        print(f"Số lượng workers: {len(self.clients)}")

        # Lọc danh sách cần xử lý
        pending_items = []
        cached_count = 0
        for sym, year in scanned_keys:
            vi_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")
            en_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")
            if os.path.exists(vi_file) and os.path.getsize(vi_file) > 400 and os.path.exists(en_file) and os.path.getsize(en_file) > 400:
                cached_count += 1
            else:
                pdf_path = self.find_scanned_pdf_path(sym, year)
                if pdf_path:
                    pending_items.append((sym, year, pdf_path))
                else:
                    print(f"[!] Không tìm thấy PDF cho {sym} {year}")

        print(f"-> Đã có sẵn trong cache: {cached_count}/{len(scanned_keys)} báo cáo.")
        print(f"-> Cần OCR & Tóm tắt mới: {len(pending_items)} báo cáo.")

        if pending_items:
            task_queue = queue.Queue()
            for item in pending_items:
                task_queue.put(item)

            pbar = tqdm(total=len(pending_items), desc="OCR & Tóm tắt PDF Scan")
            lock = threading.Lock()
            success_count = [0]
            fail_count = [0]

            def worker_fn(worker_idx, client):
                while True:
                    try:
                        sym, year, pdf_path = task_queue.get_nowait()
                    except queue.Empty:
                        break

                    vi_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY.txt")
                    en_file = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")

                    try:
                        vi_sum, en_sum = self.process_single_scanned_pdf(sym, year, pdf_path, client)
                        if vi_sum and en_sum:
                            with open(vi_file, "w", encoding="utf-8") as f:
                                f.write(vi_sum)
                            with open(en_file, "w", encoding="utf-8") as f:
                                f.write(en_sum)
                            with lock:
                                success_count[0] += 1
                        else:
                            with lock:
                                fail_count[0] += 1
                                print(f"\n[Worker {worker_idx} Thất bại {sym} {year}]")
                    except Exception as e:
                        with lock:
                            fail_count[0] += 1
                            print(f"\n[Worker {worker_idx} Lỗi {sym} {year}]: {e}")
                    finally:
                        with lock:
                            pbar.update(1)
                        task_queue.task_done()
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
            print(f"\n=> KẾT QUẢ: Hoàn thành mới {success_count[0]}/{len(pending_items)} báo cáo scan (Lỗi: {fail_count[0]}).")

        # 3. Biên tập dataset tổng hợp cập nhật
        print("\n=== ĐANG BIÊN TẬP LẠI TOÀN BỘ DATASET TỔNG HỢP VỚI CÁC BÁO CÁO SCAN ===")
        all_summary_files = [f for f in os.listdir(output_dir) if f.endswith("_SUMMARY.txt")]
        print(f"Tổng số báo cáo đã có tóm tắt trong thư mục: {len(all_summary_files)}")

        records = []
        for f in sorted(all_summary_files):
            parts = f.replace("_SUMMARY.txt", "").split("_")
            if len(parts) != 2:
                continue
            sym, year = parts[0], int(parts[1])
            vi_path = os.path.join(output_dir, f)
            en_path = os.path.join(output_dir, f"{sym}_{year}_SUMMARY_EN.txt")

            vi_text = open(vi_path, encoding="utf-8").read().strip() if os.path.exists(vi_path) else ""
            en_text = open(en_path, encoding="utf-8").read().strip() if os.path.exists(en_path) else ""

            if vi_text:
                records.append({
                    "ticker": sym,
                    "year": year,
                    "summary_text": vi_text,
                    "summary_words": len(vi_text.split()),
                    "summary_text_en": en_text,
                    "summary_words_en": len(en_text.split()) if en_text else 0,
                    "summary_path_vi": vi_path,
                    "summary_path_en": en_path if en_text else ""
                })

        if records:
            df_sum = pd.DataFrame(records).sort_values(["ticker", "year"]).reset_index(drop=True)
            sum_csv = os.path.join(processed_dir, "all_stocks_summaries.csv")
            sum_parquet = os.path.join(processed_dir, "all_stocks_summaries.parquet")
            df_sum.to_csv(sum_csv, index=False)
            df_sum.to_parquet(sum_parquet, index=False)
            print(f"   [✓] File CSV:     {sum_csv} ({os.path.getsize(sum_csv)/1024/1024:.2f} MB)")
            print(f"   [✓] File Parquet: {sum_parquet} ({os.path.getsize(sum_parquet)/1024/1024:.2f} MB)")
            print(f"   [✓] TỔNG SỐ BÁO CÁO TÓM TẮT: {len(df_sum)} BÁO CÁO (Tăng từ 133 lên {len(df_sum)}!)")
            print(f"   [✓] Tổng từ tiếng Việt: {df_sum['summary_words'].sum():,} từ")
            print(f"   [✓] Tổng từ tiếng Anh : {df_sum['summary_words_en'].sum():,} từ")

            # Cập nhật all_stocks_financial_texts
            # Đối với 59 báo cáo scan, nếu chưa có trong df_ext, thêm vào với summary_text làm đại diện
            ext_keys_set = set(zip(df_ext["ticker"], df_ext["year"]))
            new_rows = []
            for _, r in df_sum.iterrows():
                key = (r["ticker"], r["year"])
                if key not in ext_keys_set:
                    new_rows.append({
                        "ticker": r["ticker"],
                        "year": r["year"],
                        "mda_text": "",
                        "notes_text": "",
                        "esg_text": "",
                        "mda_words": 0,
                        "notes_words": 0,
                        "esg_words": 0,
                        "summary_text": r["summary_text"],
                        "summary_words": r["summary_words"],
                        "summary_text_en": r["summary_text_en"],
                        "summary_words_en": r["summary_words_en"]
                    })

            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_ext_updated = pd.concat([df_ext, df_new], ignore_index=True).sort_values(["ticker", "year"]).reset_index(drop=True)
            else:
                df_ext_updated = df_ext

            merged_csv = os.path.join(processed_dir, "all_stocks_financial_texts.csv")
            merged_parquet = os.path.join(processed_dir, "all_stocks_financial_texts.parquet")
            df_ext_updated.to_csv(merged_csv, index=False)
            df_ext_updated.to_parquet(merged_parquet, index=False)
            print(f"   [✓] Đã cập nhật all_stocks_financial_texts lên {len(df_ext_updated)} bản ghi!")


if __name__ == "__main__":
    summarizer = ScannedFinancialReportSummarizer()
    summarizer.run()
