"""
Script tự động tải toàn bộ dữ liệu lịch sử giá OHLCV và chỉ số tài chính cho 48 cổ phiếu.
Hỗ trợ:
- Tự động bỏ qua mã đã tải (Resume / Cache)
- Tự động xử lý giới hạn tốc độ (Rate Limit 20 req/min của Vnstock)
- Xuất file CSV (cho Database) và Parquet (cho PyTorch)
"""

import os
import time
import json
import pandas as pd
from tqdm import tqdm

os.environ['VNSTOCK_TELEMETRY'] = 'off'

try:
    from vnstock import Quote, Fundamental
except ImportError:
    print("Vui lòng kích hoạt môi trường 'capstone' trước khi chạy!")
    exit(1)


def load_universe(config_path: str = "stocks_universe.json") -> list:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    symbols = []
    for cat_data in data["categories"].values():
        symbols.extend(cat_data["symbols"])
    return sorted(list(set(symbols)))


def download_single_price(sym: str, start_date: str, end_date: str, max_retries: int = 3):
    """Tải giá 1 mã với cơ chế tự động chờ nếu chạm giới hạn rate limit."""
    for attempt in range(max_retries):
        try:
            q = Quote(symbol=sym)
            df = q.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df["ticker"] = sym
                return df
            return None
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "giới hạn api" in err_msg or "429" in err_msg:
                print(f"\n[Chạm rate limit khi tải {sym}] Tạm nghỉ 60s để reset hạn mức...")
                time.sleep(60)
            else:
                print(f"\n[Lỗi {sym} lần {attempt+1}]: {e}")
                time.sleep(5)
    return None


def download_all_stock_prices(symbols: list, start_date: str = "2019-01-01", end_date: str = "2024-12-31"):
    raw_dir = "data/raw/prices"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    all_dfs = []
    failed_symbols = []

    print(f"\n--- [1/2] Thu thập dữ liệu Giá OHLCV ({len(symbols)} mã, 2019 -> 2024) ---")
    for sym in tqdm(symbols, desc="Xử lý OHLCV"):
        raw_file = os.path.join(raw_dir, f"{sym}.csv")
        
        # Nếu đã tải trước đó -> Đọc lại từ cache
        if os.path.exists(raw_file) and os.path.getsize(raw_file) > 1000:
            df = pd.read_csv(raw_file)
            all_dfs.append(df)
            continue

        # Nếu chưa có -> Tải từ Vnstock với giãn cách 3.2s (đảm bảo <= 20 req/phút)
        df = download_single_price(sym, start_date, end_date)
        if df is not None and not df.empty:
            df.to_csv(raw_file, index=False)
            all_dfs.append(df)
        else:
            failed_symbols.append(sym)
        
        time.sleep(3.2)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df["time"] = pd.to_datetime(merged_df["time"])
        merged_df = merged_df.sort_values(by=["ticker", "time"]).reset_index(drop=True)

        csv_path = os.path.join(processed_dir, "all_stocks_prices.csv")
        merged_df.to_csv(csv_path, index=False)

        parquet_path = os.path.join(processed_dir, "all_stocks_prices.parquet")
        merged_df.to_parquet(parquet_path, index=False)

        print(f"\n=> Tải thành công {len(all_dfs)}/{len(symbols)} mã OHLCV.")
        print(f"   Tổng số bản ghi: {len(merged_df):,} dòng.")
        print(f"   File CSV (cho Database): {csv_path} ({os.path.getsize(csv_path) / (1024*1024):.2f} MB)")
        print(f"   File Parquet (cho PyTorch): {parquet_path} ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")

    if failed_symbols:
        print(f"Mã lỗi ({len(failed_symbols)} mã): {failed_symbols}")

    return merged_df


def download_all_financial_ratios(symbols: list):
    raw_dir = "data/raw/ratios"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    all_ratios = []
    failed_symbols = []
    download_count = 0

    print(f"\n--- [2/2] Thu thập Chỉ số Tài chính Quý (BCTC Ratios) ---")
    fun = Fundamental()

    for sym in tqdm(symbols, desc="Xử lý BCTC Ratios"):
        raw_file = os.path.join(raw_dir, f"{sym}_ratios.csv")
        
        # Nếu đã tải trong cache -> Đọc lại
        if os.path.exists(raw_file) and os.path.getsize(raw_file) > 500:
            df_ratio = pd.read_csv(raw_file)
            all_ratios.append(df_ratio)
            continue

        success = False
        for retry in range(3):
            try:
                eq = fun.equity(sym)
                df_ratio = eq.ratio(period="quarter")
                if df_ratio is not None and not df_ratio.empty:
                    df_ratio["ticker"] = sym
                    df_ratio.to_csv(raw_file, index=False)
                    all_ratios.append(df_ratio)
                    success = True
                    download_count += 1
                    break
            except BaseException as e:
                print(f"\n[Chạm rate limit hoặc lỗi tại {sym}]: {e}. Tạm nghỉ 65s để reset giới hạn...")
                time.sleep(65)

        if not success:
            failed_symbols.append(sym)

        # Giãn cách 6.5s (vì mỗi lần gọi tốn ~2 request nội bộ -> 2 req/6.5s = ~18 req/phút < 20)
        time.sleep(6.5)
        # Cứ mỗi 7 mã mới tải, nghỉ thêm 15s để làm sạch sliding window
        if download_count > 0 and download_count % 7 == 0:
            time.sleep(15)

    if all_ratios:
        merged_ratios = pd.concat(all_ratios, ignore_index=True)
        csv_path = os.path.join(processed_dir, "all_stocks_ratios.csv")
        merged_ratios.to_csv(csv_path, index=False)
        parquet_path = os.path.join(processed_dir, "all_stocks_ratios.parquet")
        merged_ratios.to_parquet(parquet_path, index=False)

        print(f"\n=> Tải thành công {len(all_ratios)}/{len(symbols)} mã Chỉ số BCTC.")
        print(f"   File CSV (cho Database): {csv_path}")
        print(f"   File Parquet (cho PyTorch): {parquet_path}")

    return all_ratios


if __name__ == "__main__":
    symbols = load_universe("stocks_universe.json")
    print(f"Danh mục thực nghiệm: {len(symbols)} mã.")
    
    # 1. Tải giá OHLCV
    download_all_stock_prices(symbols, start_date="2019-01-01", end_date="2024-12-31")
    
    # 2. Tải Chỉ số BCTC
    download_all_financial_ratios(symbols)
    
    print("\n[HOÀN TẤT TOÀN BỘ] Dữ liệu đã sẵn sàng trong thư mục data/processed/")
