"""
Pipeline tự động tải Báo cáo Thường niên (BCTN) và trích xuất tự động MD&A + Thuyết minh BCTC + ESG cho 48 mã cổ phiếu.
Hỗ trợ mở rộng đa năm (2021 - 2023), tự động nén và tổng hợp vào data/processed/.
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from extract_financial_text import FinancialReportExtractor


def load_universe(config_path: str = "stocks_universe.json") -> list:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    symbols = []
    for cat_data in data["categories"].values():
        symbols.extend(cat_data["symbols"])
    return sorted(list(set(symbols)))


def find_bctn_url(sym: str, year: int) -> str:
    """Tìm URL BCTN trên CDN Vietstock theo các mẫu chuẩn."""
    patterns = [
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTN/VN/{sym}_Baocaothuongnien_{year}.pdf",
        f"https://static2.vietstock.vn/data/HNX/{year}/BCTN/VN/{sym}_Baocaothuongnien_{year}.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTN/VN/{sym}_BCTN_{year}.pdf",
        f"https://static2.vietstock.vn/data/HNX/{year}/BCTN/VN/{sym}_BCTN_{year}.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTN/VN/{sym}_Baocao_Thuongnien_{year}.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTN/VN/{sym}_Baocaothuongnien_{year}_VN.pdf",
    ]
    for url in patterns:
        try:
            r = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if r.status_code == 200:
                return url
        except Exception:
            pass
    return None


def compile_financial_texts(text_dir: str = "data/text/extracted_text", processed_dir: str = "data/processed"):
    """Tổng hợp toàn bộ các file .txt đã bóc tách thành CSV & Parquet duy nhất."""
    records = []
    files = [f for f in os.listdir(text_dir) if f.endswith("_MDA.txt")]
    for f in sorted(files):
        parts = f.replace("_MDA.txt", "").split("_")
        if len(parts) != 2:
            continue
        ticker, year = parts[0], int(parts[1])
        mda_file = os.path.join(text_dir, f"{ticker}_{year}_MDA.txt")
        notes_file = os.path.join(text_dir, f"{ticker}_{year}_NOTES.txt")
        esg_file = os.path.join(text_dir, f"{ticker}_{year}_ESG.txt")

        mda_text = open(mda_file, encoding="utf-8").read().strip() if os.path.exists(mda_file) else ""
        notes_text = open(notes_file, encoding="utf-8").read().strip() if os.path.exists(notes_file) else ""
        esg_text = open(esg_file, encoding="utf-8").read().strip() if os.path.exists(esg_file) else ""

        # Chỉ lưu các bản ghi có text thực sự
        if len(mda_text) > 100 or len(notes_text) > 100 or len(esg_text) > 100:
            records.append({
                "ticker": ticker,
                "year": year,
                "mda_text": mda_text,
                "notes_text": notes_text,
                "esg_text": esg_text,
                "mda_words": len(mda_text.split()),
                "notes_words": len(notes_text.split()),
                "esg_words": len(esg_text.split())
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["ticker", "year"]).reset_index(drop=True)
        csv_path = os.path.join(processed_dir, "all_stocks_financial_texts.csv")
        parquet_path = os.path.join(processed_dir, "all_stocks_financial_texts.parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        print(f"\n=> ĐÃ TỔNG HỢP {len(df)} BẢN GHI VĂN BẢN TÀI CHÍNH:")
        print(f"   CSV:     {csv_path} ({os.path.getsize(csv_path)/1024/1024:.2f} MB)")
        print(f"   Parquet: {parquet_path} ({os.path.getsize(parquet_path)/1024/1024:.2f} MB)")
        print(f"   Tổng số từ MD&A:        {df['mda_words'].sum():,}")
        print(f"   Tổng số từ Thuyết minh: {df['notes_words'].sum():,}")
        print(f"   Tổng số từ ESG:         {df['esg_words'].sum():,}")
        print(f"   TỔNG LƯỢNG TỪ:          {df['mda_words'].sum() + df['notes_words'].sum() + df['esg_words'].sum():,} từ!")
    return df


def download_and_extract_all(symbols: list, years: list = [2023, 2022, 2021]):
    pdf_dir = "data/text/raw_pdf"
    text_dir = "data/text/extracted_text"
    processed_dir = "data/processed"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    extractor = FinancialReportExtractor()
    records = []

    print(f"\n=== BẮT ĐẦU PIPELINE TẢI VÀ BÓC TÁCH VĂN BẢN ({len(symbols)} mã, Năm: {years}) ===")

    for year in sorted(years, reverse=True):
        print(f"\n>>> ĐANG XỬ LÝ NĂM {year} <<<")
        for sym in tqdm(symbols, desc=f"Năm {year}"):
            mda_file = os.path.join(text_dir, f"{sym}_{year}_MDA.txt")
            notes_file = os.path.join(text_dir, f"{sym}_{year}_NOTES.txt")
            esg_file = os.path.join(text_dir, f"{sym}_{year}_ESG.txt")

            # 1. Đọc lại từ cache nếu đã bóc tách
            if (os.path.exists(mda_file) and os.path.getsize(mda_file) > 1000) or \
               (os.path.exists(notes_file) and os.path.getsize(notes_file) > 1000):
                records.append({
                    "ticker": sym,
                    "year": year,
                    "mda_len": os.path.getsize(mda_file) if os.path.exists(mda_file) else 0,
                    "notes_len": os.path.getsize(notes_file) if os.path.exists(notes_file) else 0,
                    "esg_len": os.path.getsize(esg_file) if os.path.exists(esg_file) else 0,
                    "status": "cached",
                    "mda_path": mda_file,
                    "notes_path": notes_file,
                    "esg_path": esg_file
                })
                continue

            # 2. Kiểm tra nếu file PDF đã có sẵn trên máy
            pdf_path = os.path.join(pdf_dir, f"{sym}_BCTN_{year}.pdf")
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 50000:
                # 3. Tìm URL và tải PDF
                url = find_bctn_url(sym, year)
                if not url:
                    continue

                try:
                    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
                    if r.status_code == 200 and len(r.content) > 50000:
                        with open(pdf_path, "wb") as f:
                            f.write(r.content)
                    else:
                        continue
                except Exception as e:
                    print(f"Lỗi tải PDF {sym} {year}: {e}")
                    continue
            else:
                url = "local_cache"

            # 4. Tự động bóc tách MD&A, Thuyết minh BCTC và ESG
            try:
                res = extractor.process_pdf(pdf_path, ticker=sym, year=year, output_dir=text_dir)
                res["status"] = "extracted"
                res["pdf_url"] = url
                records.append(res)
            except Exception as e:
                print(f"Lỗi bóc tách {sym} {year}: {e}")

            time.sleep(0.3)

    if records:
        df_meta = pd.DataFrame(records)
        meta_csv = os.path.join(processed_dir, "all_stocks_reports_meta.csv")
        meta_parquet = os.path.join(processed_dir, "all_stocks_reports_meta.parquet")
        df_meta.to_csv(meta_csv, index=False)
        df_meta.to_parquet(meta_parquet, index=False)

    # Tổng hợp toàn bộ vào CSV & Parquet
    compile_financial_texts(text_dir, processed_dir)

    return records


if __name__ == "__main__":
    symbols = load_universe("stocks_universe.json")
    # Mở rộng toàn diện cho tất cả các năm đã khép sổ: 2021 -> 2025
    download_and_extract_all(symbols, years=[2025, 2024, 2023, 2022, 2021])
