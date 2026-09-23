"""
Module thu thập Văn bản Báo cáo Tài chính & Giải trình Chênh lệch Lợi nhuận Quý (Q1/2026, Q2/2026).
Triển khai theo Phương án A để bổ sung các 'quý lẻ' của năm hiện tại (2026).
"""

import os
import time
import json
import requests
import pandas as pd
import pymupdf
from tqdm import tqdm


def load_universe(config_path: str = "stocks_universe.json") -> list:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    symbols = []
    for cat_data in data["categories"].values():
        symbols.extend(cat_data["symbols"])
    return sorted(list(set(symbols)))


def find_quarterly_url(sym: str, year: int, quarter: str) -> str:
    """Tìm URL BCTC & Giải trình Quý trên CDN Vietstock."""
    q_num = quarter.replace("Q", "").strip()
    patterns = [
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTC/VN/QUY%20{q_num}/{sym}_Baocaotaichinh_Q{q_num}_{year}_Hopnhat.pdf",
        f"https://static2.vietstock.vn/data/HNX/{year}/BCTC/VN/QUY%20{q_num}/{sym}_Baocaotaichinh_Q{q_num}_{year}_Hopnhat.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTC/VN/QUY%20{q_num}/{sym}_Baocaotaichinh_Q{q_num}_{year}.pdf",
        f"https://static2.vietstock.vn/data/HNX/{year}/BCTC/VN/QUY%20{q_num}/{sym}_Baocaotaichinh_Q{q_num}_{year}.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTC/VN/QUY%20{q_num}/{sym}_BCTC_Q{q_num}_{year}_HN.pdf",
        f"https://static2.vietstock.vn/data/HOSE/{year}/BCTC/VN/QUY%20{q_num}/{sym}_BCTC_Q{q_num}_{year}.pdf",
    ]
    for url in patterns:
        try:
            r = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if r.status_code == 200:
                return url
        except Exception:
            pass
    return None


def extract_quarterly_text(pdf_path: str, max_pages: int = 20) -> dict:
    """Bóc tách văn bản tóm tắt tình hình kinh doanh, giải trình và thuyết minh quý."""
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return {"summary_text": "", "notes_text": "", "is_scanned": True, "total_pages": 0}

    total_pages = len(doc)
    text_pages = []
    notes_pages = []
    is_notes = False

    for pno in range(min(total_pages, 45)):
        text = doc[pno].get_text().strip()
        if len(text) < 50:
            continue
        text_pages.append(text)
        
        # Nhận diện thuyết minh quý
        text_upper = text.upper()
        if "THUYẾT MINH BÁO CÁO TÀI CHÍNH" in text_upper:
            is_notes = True
        if is_notes:
            notes_pages.append(text)
            if len(notes_pages) >= max_pages:
                break

    doc.close()
    is_scanned = len(text_pages) < 2
    summary = "\n\n".join(text_pages[:8])  # 8 trang đầu thường là tổng quan, bảng biểu tóm tắt
    notes = "\n\n".join(notes_pages)

    return {
        "summary_text": summary,
        "notes_text": notes,
        "is_scanned": is_scanned,
        "total_pages": total_pages,
        "text_pages_count": len(text_pages)
    }


def download_and_process_quarters_2026(symbols: list, quarters: list = ["Q1", "Q2"]):
    pdf_dir = "data/text/quarterly_2026/raw_pdf"
    text_dir = "data/text/quarterly_2026/extracted_text"
    processed_dir = "data/processed"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    records = []
    print(f"\n=== BẮT ĐẦU THU THẬP BÁO CÁO QUÝ LẺ NĂM 2026 ({len(symbols)} mã, Quý: {quarters}) ===")

    for q in quarters:
        print(f"\n>>> ĐANG XỬ LÝ {q}/2026 <<<")
        for sym in tqdm(symbols, desc=f"Quý {q}"):
            txt_file = os.path.join(text_dir, f"{sym}_2026_{q}.txt")
            pdf_path = os.path.join(pdf_dir, f"{sym}_2026_{q}.pdf")

            # 1. Đọc lại từ cache nếu đã có
            if os.path.exists(txt_file) and os.path.getsize(txt_file) > 100:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                records.append({
                    "ticker": sym,
                    "year": 2026,
                    "quarter": q,
                    "text": content,
                    "words_count": len(content.split()),
                    "status": "cached"
                })
                continue

            # 2. Tìm link tải
            url = find_quarterly_url(sym, 2026, q)
            if not url:
                continue

            # 3. Tải PDF
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 10000:
                try:
                    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    if r.status_code == 200 and len(r.content) > 10000:
                        with open(pdf_path, "wb") as f:
                            f.write(r.content)
                    else:
                        continue
                except Exception:
                    continue

            # 4. Bóc tách
            res = extract_quarterly_text(pdf_path)
            combined_text = (res["summary_text"] + "\n\n" + res["notes_text"]).strip()
            
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(combined_text)

            if len(combined_text) > 100:
                records.append({
                    "ticker": sym,
                    "year": 2026,
                    "quarter": q,
                    "text": combined_text,
                    "words_count": len(combined_text.split()),
                    "status": "extracted",
                    "pdf_url": url
                })

            time.sleep(0.3)

    if records:
        df_q = pd.DataFrame(records)
        df_q = df_q.sort_values(by=["ticker", "quarter"]).reset_index(drop=True)
        csv_path = os.path.join(processed_dir, "all_stocks_quarterly_texts.csv")
        parquet_path = os.path.join(processed_dir, "all_stocks_quarterly_texts.parquet")
        df_q.to_csv(csv_path, index=False)
        df_q.to_parquet(parquet_path, index=False)

        print(f"\n=> ĐÃ HOÀN TẤT BỔ SUNG QUÝ LẺ 2026:")
        print(f"   Tổng số bản ghi: {len(df_q)} bản ghi quý.")
        print(f"   Tổng số từ vựng: {df_q['words_count'].sum():,} từ.")
        print(f"   CSV:     {csv_path} ({os.path.getsize(csv_path)/1024:.1f} KB)")
        print(f"   Parquet: {parquet_path} ({os.path.getsize(parquet_path)/1024:.1f} KB)")

    return records


if __name__ == "__main__":
    symbols = load_universe("stocks_universe.json")
    download_and_process_quarters_2026(symbols, quarters=["Q1", "Q2"])
