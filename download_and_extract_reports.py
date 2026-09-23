"""
Pipeline tự động tải Báo cáo Thường niên (BCTN) và trích xuất tự động MD&A + ESG cho 48 mã cổ phiếu.
Lưu text vào data/text/extracted_text/ và tổng hợp metadata vào data/processed/.
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


def download_and_extract_all(symbols: list, years: list = [2023, 2022]):
    pdf_dir = "data/text/raw_pdf"
    text_dir = "data/text/extracted_text"
    processed_dir = "data/processed"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    extractor = FinancialReportExtractor()
    records = []

    print(f"\n=== BẮT ĐẦU PIPELINE TẢI VÀ BÓC TÁCH VĂN BẢN ({len(symbols)} mã, Năm: {years}) ===")

    for sym in symbols:
        for year in years:
            mda_file = os.path.join(text_dir, f"{sym}_{year}_MDA.txt")
            esg_file = os.path.join(text_dir, f"{sym}_{year}_ESG.txt")

            # 1. Nếu đã bóc tách text trước đó -> Đọc lại từ cache
            if os.path.exists(mda_file) and os.path.getsize(mda_file) > 1000:
                records.append({
                    "ticker": sym,
                    "year": year,
                    "mda_len": os.path.getsize(mda_file),
                    "esg_len": os.path.getsize(esg_file) if os.path.exists(esg_file) else 0,
                    "status": "cached",
                    "mda_path": mda_file,
                    "esg_path": esg_file
                })
                continue

            # 2. Tìm URL
            url = find_bctn_url(sym, year)
            if not url:
                continue

            pdf_path = os.path.join(pdf_dir, f"{sym}_BCTN_{year}.pdf")

            # 3. Tải PDF nếu chưa có
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 10000:
                try:
                    print(f"\n[Đang tải] {sym} {year} từ {url}...")
                    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    if r.status_code == 200:
                        with open(pdf_path, "wb") as f:
                            f.write(r.content)
                    else:
                        continue
                except Exception as e:
                    print(f"Lỗi tải PDF {sym} {year}: {e}")
                    continue

            # 4. Tự động bóc tách MD&A và ESG
            try:
                res = extractor.process_pdf(pdf_path, ticker=sym, year=year, output_dir=text_dir)
                res["status"] = "extracted"
                res["pdf_url"] = url
                records.append(res)
            except Exception as e:
                print(f"Lỗi bóc tách {sym} {year}: {e}")

            time.sleep(0.5)

    if records:
        df_meta = pd.DataFrame(records)
        meta_csv = os.path.join(processed_dir, "all_stocks_reports_meta.csv")
        meta_parquet = os.path.join(processed_dir, "all_stocks_reports_meta.parquet")
        df_meta.to_csv(meta_csv, index=False)
        df_meta.to_parquet(meta_parquet, index=False)

        print(f"\n=> ĐÃ HOÀN TẤT BÓC TÁCH VĂN BẢN:")
        print(f"   Tổng số tài liệu đã xử lý: {len(records)} tài liệu.")
        print(f"   Metadata đã lưu: {meta_csv}")

    return records


if __name__ == "__main__":
    symbols = load_universe("stocks_universe.json")
    print(f"Danh mục: {len(symbols)} mã.")
    # Chạy cho năm 2023 (năm gần nhất có đầy đủ BCTN)
    download_and_extract_all(symbols, years=[2023])
