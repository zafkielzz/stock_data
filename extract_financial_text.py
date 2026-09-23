"""
Module tự động nhận diện và bóc tách các phân đoạn quan trọng từ Báo cáo Thường niên (Annual Reports):
1. MD&A (Báo cáo của Ban Điều hành / Ban Tổng Giám đốc về kết quả kinh doanh và rủi ro)
2. Báo cáo ESG (Phát triển bền vững, Môi trường, Xã hội, Quản trị)
Không cần đọc thủ công - Tự động quét và chấm điểm dải trang theo ngữ nghĩa.
"""

import os
import pymupdf
import pandas as pd


class FinancialReportExtractor:
    def __init__(self):
        # Bộ từ khóa nhận diện phần MD&A
        self.mda_positive_kw = [
            'ban điều hành', 'ban tổng giám đốc', 'ban giám đốc', 'đánh giá của ban', 
            'phân tích hoạt động kinh doanh', 'kết quả kinh doanh', 'bối cảnh chung', 
            'rủi ro', 'triển vọng', 'kế hoạch kinh doanh', 'chiến lược',
            'báo cáo của ban giám đốc', 'báo cáo của ban điều hành', 'báo cáo của tổng giám đốc',
            'thông điệp tổng giám đốc', 'thông điệp ceo', 'tình hình hoạt động',
            'báo cáo hoạt động', 'đánh giá kết quả hoạt động'
        ]
        self.mda_negative_kw = [
            'bảng cân đối kế toán', 'thuyết minh báo cáo tài chính', 'ý kiến kiểm toán',
            'danh sách cổ đông', 'sơ yếu lý lịch', 'quản trị công ty'
        ]

        # Bộ từ khóa nhận diện phần ESG
        self.esg_positive_kw = [
            'báo cáo esg', 'phát triển bền vững', 'phát thải khí nhà kính', 
            'môi trường, xã hội', 'môi trường và xã hội', 'năng lượng tái tạo', 
            'tiêu chuẩn lao động', 'báo cáo phát triển bền vững', 'trách nhiệm xã hội',
            'quản trị môi trường', 'tiết kiệm năng lượng', 'báo cáo tác động môi trường'
        ]

    def extract_mda(self, pdf_path: str, max_pages: int = 25) -> str:
        """Tự động tìm và rút trích chương MD&A từ file PDF."""
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        
        page_scores = []
        for pno in range(total_pages):
            text = doc[pno].get_text().lower()
            score = 0
            for kw in self.mda_positive_kw:
                score += text.count(kw) * 2
            for kw in self.mda_negative_kw:
                score -= text.count(kw) * 3
            page_scores.append((pno, score))
        
        # Lọc các trang có điểm cao
        relevant_pages = [pno for pno, score in page_scores if score >= 3]
        
        extracted_text = []
        for pno in relevant_pages[:max_pages]:
            extracted_text.append(doc[pno].get_text().strip())
        
        doc.close()
        return "\n\n".join(extracted_text)

    def extract_esg(self, pdf_path: str, max_pages: int = 20) -> str:
        """Tự động tìm và rút trích chương ESG từ file PDF."""
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        
        esg_pages = []
        for pno in range(len(doc)):
            text = doc[pno].get_text().lower()
            score = sum(text.count(k) for k in self.esg_positive_kw)
            if score >= 2:
                esg_pages.append(pno)

        extracted_text = []
        for pno in esg_pages[:max_pages]:
            extracted_text.append(doc[pno].get_text().strip())
        
        doc.close()
        return "\n\n".join(extracted_text)

    def extract_notes(self, pdf_path: str, max_pages: int = 60) -> str:
        """Tự động tìm và rút trích Bản Thuyết minh Báo cáo Tài chính (Doanh nghiệp & Ngân hàng)."""
        doc = pymupdf.open(pdf_path)
        notes_pages = []
        found_start = False
        
        note_triggers = [
            'THUYẾT MINH BÁO CÁO TÀI CHÍNH',
            'BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH',
            'CÁC THUYẾT MINH BÁO CÁO TÀI CHÍNH',
            'THUYẾT MINH BCTC',
            'B 09 – DN', 'B09-DN', 'B 09 - DN', 'B09/DN', 'B 09 – DN/HN', 'B09 - DN/HN',
            'B05/TCTD', 'B 05/TCTD', 'B05 - TCTD', 'B05-TCTD', 'B 05 - TCTD',
            'MẪU B 09', 'MẪU B09', 'MẪU B 05', 'MẪU B05'
        ]
        
        for pno in range(len(doc)):
            text = doc[pno].get_text()
            text_upper = text.upper()
            if not found_start:
                # Bỏ qua các trang mục lục
                is_toc = 'MỤC LỤC' in text_upper or ('NỘI DUNG' in text_upper and 'TRANG' in text_upper)
                if not is_toc:
                    if any(trig in text_upper for trig in note_triggers):
                        found_start = True
                    elif 'THUYẾT MINH' in text_upper and any(k in text_upper for k in ['CHÍNH SÁCH KẾ TOÁN', 'CƠ SỞ LẬP BÁO CÁO', 'ĐẶC ĐIỂM HOẠT ĐỘNG', 'BỘ PHẬN HỢP THÀNH']):
                        found_start = True
            if found_start:
                notes_pages.append(text.strip())
                if len(notes_pages) >= max_pages:
                    break
                    
        doc.close()
        return "\n\n".join(notes_pages)

    def process_pdf(self, pdf_path: str, ticker: str, year: int, output_dir: str = "data/text/extracted_text"):
        """Xử lý 1 file PDF và lưu ra các file text chuẩn hóa."""
        os.makedirs(output_dir, exist_ok=True)
        
        mda_text = self.extract_mda(pdf_path)
        esg_text = self.extract_esg(pdf_path)
        notes_text = self.extract_notes(pdf_path)
        
        mda_file = os.path.join(output_dir, f"{ticker}_{year}_MDA.txt")
        esg_file = os.path.join(output_dir, f"{ticker}_{year}_ESG.txt")
        notes_file = os.path.join(output_dir, f"{ticker}_{year}_NOTES.txt")
        
        with open(mda_file, "w", encoding="utf-8") as f:
            f.write(mda_text)
            
        with open(esg_file, "w", encoding="utf-8") as f:
            f.write(esg_text)
            
        with open(notes_file, "w", encoding="utf-8") as f:
            f.write(notes_text)
            
        print(f"[{ticker} {year}] Trích xuất thành công:")
        print(f"   MD&A:       {len(mda_text):,} ký tự -> {mda_file}")
        print(f"   Thuyết minh:{len(notes_text):,} ký tự -> {notes_file}")
        print(f"   ESG:        {len(esg_text):,} ký tự -> {esg_file}")
        
        return {
            "ticker": ticker,
            "year": year,
            "mda_len": len(mda_text),
            "notes_len": len(notes_text),
            "esg_len": len(esg_text),
            "mda_path": mda_file,
            "notes_path": notes_file,
            "esg_path": esg_file
        }


if __name__ == "__main__":
    extractor = FinancialReportExtractor()
    sample_pdf = "data/text/raw_pdf/FPT_BCTN_2023.pdf"
    if os.path.exists(sample_pdf):
        res = extractor.process_pdf(sample_pdf, ticker="FPT", year=2023)
        print("Kết quả:", res)
