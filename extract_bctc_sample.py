"""
Script mẫu: Thu thập và xử lý dữ liệu Tài chính (Vnstock + Xử lý PDF BCTC)
Môi trường khuyến nghị: conda activate capstone
"""

import os
import sys
import pandas as pd

# Tắt thông báo đo lường của vnstock nếu không cần
os.environ['VNSTOCK_TELEMETRY'] = 'off'

try:
    from vnstock import Quote, Fundamental, Company, Listing
    import pymupdf
    import pypdf
except ImportError as e:
    print(f"Lỗi import: {e}. Vui lòng chạy trên conda env 'capstone'.")


def get_numerical_financial_data(symbol: str = "HPG", start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    """
    1. Lấy dữ liệu chuỗi giá OHLCV (cho nhánh Time Series)
    2. Lấy dữ liệu BCTC và các chỉ số định lượng (P/E, EPS, ROE, Net Margin...)
    Không cần OCR hay đọc file PDF cho phần số này!
    """
    print(f"\n--- [1] Thu thập dữ liệu định lượng cho mã {symbol} qua Vnstock ---")
    
    # 1.1 Chuỗi giá OHLCV
    quote = Quote(symbol=symbol)
    df_price = quote.history(start=start_date, end=end_date)
    print(f"Dữ liệu giá OHLCV ({len(df_price)} phiên giao dịch):")
    print(df_price.head(3))
    
    # 1.2 Chỉ số tài chính theo Quý (P/E, EPS, ROE, Margin, BVPS...)
    fun = Fundamental()
    eq = fun.equity(symbol)
    df_ratio = eq.ratio(period="quarter")
    print(f"\nChỉ số tài chính Quý:")
    print(df_ratio.head(5))
    
    return df_price, df_ratio


def inspect_and_extract_pdf_text(pdf_path: str, max_pages: int = 5):
    """
    Kiểm tra file PDF BCTC:
    - Nếu là Digital PDF (có Text Layer): Trích xuất text thuần 100% sạch sẽ, KHÔNG CẦN OCR.
    - Nếu là Scanned PDF (chỉ có ảnh chụp): Báo hiệu cần dùng OCR (VietOCR / PaddleOCR).
    """
    if not os.path.exists(pdf_path):
        print(f"Không tìm thấy file: {pdf_path}")
        return None

    print(f"\n--- [2] Kiểm tra & Trích xuất file PDF: {os.path.basename(pdf_path)} ---")
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"Tổng số trang: {total_pages}")

    extracted_pages = []
    is_scanned = True

    for i in range(min(total_pages, max_pages)):
        page = doc[i]
        text = page.get_text().strip()
        # Nếu có văn bản > 50 ký tự trong trang -> Có Text Layer (Digital PDF)
        if len(text) > 50:
            is_scanned = False
        extracted_pages.append((i + 1, text))

    if is_scanned:
        print("=> KẾT QUẢ: File PDF này là DẠNG SCAN (Ảnh chụp).")
        print("   -> Cần dùng mô hình OCR (như PaddleOCR hoặc VietOCR) để nhận diện chữ.")
    else:
        print("=> KẾT QUẢ: File PDF này là DIGITAL PDF (Có sẵn Text Layer nguyên bản).")
        print("   -> KHÔNG CẦN OCR! Có thể đọc trực tiếp bằng PyMuPDF hoặc pypdf.")
        print(f"   -> Mẫu text trang 1 (500 ký tự đầu):\n{extracted_pages[0][1][:500]}...")

    doc.close()
    return is_scanned, extracted_pages


if __name__ == "__main__":
    # Test thu thập dữ liệu qua Vnstock
    df_price, df_ratio = get_numerical_financial_data(symbol="FPT")
    print("\n[Done] Pipeline kiểm tra dữ liệu hoàn tất.")
