"""
Module tự động nhận diện và bóc tách các phân đoạn quan trọng từ Báo cáo Thường niên (Annual Reports):
1. MD&A (Báo cáo của Ban Điều hành / Ban Tổng Giám đốc về kết quả kinh doanh và rủi ro)
2. Báo cáo ESG (Phát triển bền vững, Môi trường, Xã hội, Quản trị)
3. Thuyết minh BCTC Định tính (Targeted Narrative NOTES): Bóc tách các câu giải mã, rủi ro nợ vay,
   dự phòng, nợ tiềm tàng, báo cáo bộ phận; lọc bỏ triệt để các bảng số liệu ma trận vỡ layout (Digit-walls).
"""

import os
import re
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

        # Bộ từ khóa nhận diện phần ESG toàn diện (Theo chuẩn GRI, HOSE VNSI, Net-Zero, SDGs)
        self.esg_positive_kw = [
            # Môi trường (Environmental)
            'báo cáo esg', 'phát triển bền vững', 'báo cáo phát triển bền vững',
            'phát thải khí nhà kính', 'khí nhà kính', 'giảm phát thải', 'net zero', 'trung hòa carbon',
            'scope 1', 'scope 2', 'scope 3', 'dấu chân carbon', 'tín chỉ carbon',
            'năng lượng tái tạo', 'năng lượng mặt trời', 'tiết kiệm năng lượng', 'hiệu quả năng lượng',
            'kinh tế tuần hoàn', 'tái chế chất thải', 'xử lý nước thải', 'chất thải nguy hại',
            'bảo tồn đa dạng sinh học', 'môi trường và xã hội', 'môi trường, xã hội', 'quản trị môi trường',
            # Xã hội (Social)
            'trách nhiệm xã hội', 'an toàn vệ sinh lao động', 'an toàn lao động', 'sức khỏe nghề nghiệp',
            'bình đẳng giới', 'tỷ lệ lao động nữ', 'phát triển nguồn nhân lực', 'giờ đào tạo',
            'chế độ phúc lợi', 'tiêu chuẩn lao động', 'hoạt động cộng đồng', 'an sinh xã hội',
            # Quản trị & Tiêu chuẩn (Governance & Standards)
            'quản trị công ty', 'đạo đức kinh doanh', 'chống tham nhũng', 'chống hối lộ', 'quy tắc ứng xử',
            'chuẩn mực gri', 'tiêu chuẩn gri', 'chỉ số vnsi', 'mục tiêu phát triển bền vững', 'sdgs'
        ]

        # Trang loại trừ trong Thuyết minh (BCTC chính & Báo cáo kiểm toán)
        self.negative_notes_pages = [
            'báo cáo kiểm toán độc lập', 'ý kiến của kiểm toán viên', 'trách nhiệm của kiểm toán viên',
            'bảng cân đối kế toán hợp nhất', 'báo cáo kết quả hoạt động kinh doanh hợp nhất',
            'báo cáo lưu chuyển tiền tệ hợp nhất', 'báo cáo tình hình tài chính hợp nhất',
            'báo cáo kết quả hoạt động hợp nhất'
        ]

        # Trang chính sách kế toán lặp lại (Boilerplate VAS)
        self.policy_boilerplate = [
            'các chính sách kế toán chủ yếu', 'tóm tắt các chính sách kế toán chủ yếu',
            'cơ sở lập báo cáo tài chính và các chính sách kế toán'
        ]

        # Các đề mục Thuyết minh có giá trị giải mã thông tin tài chính cao nhất (Narrative Disclosures)
        self.narrative_notes_kw = [
            'báo cáo bộ phận', 'thông tin bộ phận', 'lĩnh vực kinh doanh', 'theo vùng địa lý',
            'vay và nợ', 'vay ngắn hạn', 'vay dài hạn', 'trái phiếu phát hành', 'phát hành trái phiếu',
            'lãi suất', 'tài sản bảo đảm', 'tài sản thế chấp', 'cam kết tài chính',
            'dự phòng', 'nợ khó đòi', 'nợ xấu', 'giảm giá hàng tồn kho', 'trích lập dự phòng', 'hoàn nhập dự phòng',
            'nợ tiềm tàng', 'cam kết và nợ', 'cam kết ngoại bảng', 'tranh chấp', 'kiện tụng', 'nghĩa vụ bảo lãnh',
            'bên liên quan', 'giao dịch với các bên liên quan',
            'sự kiện phát sinh sau', 'sự kiện sau ngày kết thúc', 'sau ngày kết thúc kỳ kế toán',
            'quản lý rủi ro', 'rủi ro tín dụng', 'rủi ro thanh khoản', 'rủi ro lãi suất'
        ]

        # Cụm từ điều hướng tiêu đề / chân trang thường lặp lại
        self.nav_stops = [
            'báo cáo thường niên', 'thông điệp ban lãnh đạo', 'chiến lược phát triển',
            'quản trị công ty', 'báo cáo esg', 'báo cáo tài chính', 'tổng quan về',
            'mục lục', 'mẫu số b 09', 'mẫu số b09', 'mẫu số b 05', 'mẫu số b05',
            'tập đoàn hòa phát - báo cáo thường niên', 'làm chủ', 'công nghệ chiến lược',
            'dấu ấn', 'phân tích hoạt động kinh doanh', 'báo cào tài chính', 'báo cáo tài chính hợp nhất',
            'mô hình quản trị và vai trò'
        ]

    def is_junk_line(self, line: str) -> bool:
        """Kiểm tra và loại bỏ các dòng rác, header điều hướng, và các cột số liệu vỡ bảng."""
        s = line.strip()
        if not s:
            return True
        sl = s.lower()

        # 1. Bỏ qua header / footer điều hướng ngắn
        if any(nav in sl for nav in self.nav_stops) and len(s) < 65:
            return True

        # Đếm chữ cái tiếng Việt và chữ số
        letters = re.findall(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]', s)
        digits = re.findall(r'\d', s)

        # 2. Phải có ít nhất 4 chữ cái (loại bỏ các số trang đơn lẻ, ký tự phân cách)
        if len(letters) < 4:
            return True

        # 3. Digit-wall filter: Nếu chữ số áp đảo chữ cái (tỷ lệ > 75%) -> Đây là hàng bảng số liệu ma trận kế toán
        if len(digits) > len(letters) * 0.75:
            return True

        # 4. Loại bỏ các nhãn tiêu đề cột bảng trơ trọi không có ngữ cảnh
        words = s.split()
        if len(words) <= 3 and any(k in sl for k in [
            'số cuối năm', 'số đầu năm', 'vnd', 'đồng', 'triệu vnd', 'mã số', 'tổng cộng', 'thuyết minh số'
        ]):
            return True

        return False

    def clean_text(self, text: str) -> str:
        """Lọc sạch văn bản, loại bỏ các dòng số liệu kế toán vỡ layout và header lặp lại."""
        lines = text.split('\n')
        cleaned = [l.strip() for l in lines if not self.is_junk_line(l)]
        return '\n'.join(cleaned)

    def extract_mda(self, pdf_path: str, max_pages: int = 25) -> str:
        """Tự động tìm và rút trích chương MD&A (Báo cáo Ban Điều hành) từ file PDF."""
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        
        page_scores = []
        for pno in range(total_pages):
            text = doc[pno].get_text().lower()
            score = sum(text.count(kw) * 2 for kw in self.mda_positive_kw) - sum(text.count(kw) * 3 for kw in self.mda_negative_kw)
            page_scores.append((pno, score))
        
        relevant_pages = sorted([pno for pno, score in page_scores if score >= 3][:max_pages])
        extracted = [self.clean_text(doc[pno].get_text()) for pno in relevant_pages]
        doc.close()
        return "\n\n".join([e for e in extracted if len(e) > 80])

    def extract_esg(self, pdf_path: str, max_pages: int = 20) -> str:
        """Tự động tìm và rút trích chương ESG từ file PDF."""
        doc = pymupdf.open(pdf_path)
        esg_pages = []
        for pno in range(len(doc)):
            text = doc[pno].get_text().lower()
            score = sum(text.count(k) for k in self.esg_positive_kw)
            if score >= 2:
                esg_pages.append(pno)

        extracted = [self.clean_text(doc[pno].get_text()) for pno in esg_pages[:max_pages]]
        doc.close()
        return "\n\n".join([e for e in extracted if len(e) > 80])

    def extract_notes(self, pdf_path: str, max_pages: int = 25) -> str:
        """
        Bóc tách Thuyết minh BCTC Định tính (Targeted Narrative NOTES):
        - Bỏ qua BCTC chính (Bảng CĐKT, KQKD, LCTT) và Báo cáo kiểm toán độc lập.
        - Bỏ qua các trang chính sách kế toán chung (VAS boilerplate).
        - Nhắm mục tiêu chính xác các đề mục có câu giải mã: Báo cáo bộ phận, Vay nợ, Lãi suất,
          Dự phòng rủi ro, Cam kết ngoại bảng, Nợ tiềm tàng, Kiện tụng, Bên liên quan, Sự kiện sau niên độ.
        - Lọc bỏ triệt để các cột số ma trận kế toán (Digit Wall Filter).
        """
        doc = pymupdf.open(pdf_path)
        candidates = []
        for pno in range(len(doc)):
            text = doc[pno].get_text()
            text_lower = text.lower()

            # 1. Bỏ qua trang BCTC chính / kiểm toán
            if any(neg in text_lower for neg in self.negative_notes_pages):
                continue

            # 2. Bỏ qua trang chính sách kế toán lặp lại
            if any(pol in text_lower for pol in self.policy_boilerplate):
                continue

            # 3. Phải nằm trong phạm vi Thuyết minh BCTC
            is_notes = ('thuyết minh' in text_lower or 'b09' in text_lower or 'b05' in text_lower or 'mẫu số b' in text_lower or 'ghi chú' in text_lower)
            if not is_notes:
                continue

            # 4. Chấm điểm mật độ xuất hiện của các đề mục giải mã định tính
            score = sum(text_lower.count(kw) * 3 for kw in self.narrative_notes_kw)
            if score >= 3:
                candidates.append((pno, score))

        # Chọn các trang có điểm giải thích cao nhất và giữ nguyên thứ tự xuất hiện
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_pages = sorted([pno for pno, _ in candidates[:max_pages]])

        extracted = []
        for pno in selected_pages:
            cleaned = self.clean_text(doc[pno].get_text())
            if len(cleaned) > 80:
                extracted.append(cleaned)
        doc.close()
        return "\n\n".join(extracted)

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
            
        print(f"[{ticker} {year}] Trích xuất định tính thành công:")
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
    sample_pdf = "data/text/raw_pdf/FPT_BCTN_2025.pdf"
    if os.path.exists(sample_pdf):
        res = extractor.process_pdf(sample_pdf, ticker="FPT", year=2025)
        print("Kết quả:", res)
