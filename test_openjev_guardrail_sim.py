"""Simulation Test: OpenJev as a Calibrated Financial Risk Guardrail for Capstone Project.

Evaluates OpenJev (ModernBERT-151M) on GPU across 4 realistic Capstone scenarios:
1. FPT FY2021: Strong fundamentals -> Approved Bullish Dispatch.
2. DXG FY2024: Qwen predicts UP but text describes rising interest & -19.5% price fall -> BLOCKED (Hallucination Alert!).
3. Stock Forum Rumor: Unverified hype without financials -> ABSTAINED (__insufficient_evidence__).
4. VNM Stagnation: Weak sideways signal -> LOW CONFIDENCE WARNING.
"""

from __future__ import annotations

import time
import torch
from rlcd import DecisionEngine, Choice, Option

def main():
    print("=" * 90)
    print("🛡️ CAPSTONE FINANCIAL GUARDRAIL SIMULATION: OPENJEV (SYSTEM 1) EVALUATION")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phần cứng thực thi : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"Kiến trúc mô phỏng : System 1 Calibrated Guardrail giám sát mô hình sinh Qwen2.5-0.5B")
    print("-" * 90)

    # 1. Khởi tạo OpenJev trên GPU
    model_id = "heman10x/rlcd-modernbert-151m"
    print(f"Đang nạp trọng số OpenJev ({model_id}) vào GPU...")
    t0 = time.perf_counter()
    engine = DecisionEngine(model_name_or_path=model_id, device=device)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"✅ Guardrail Engine sẵn sàng hoạt động trong {init_ms:.1f} ms\n")

    # 2. Định nghĩa Schema kiểm định an toàn
    guardrail_query = Choice(
        id="financial_stance",
        question="What is the fundamental investment trajectory described in this financial report?",
        options=[
            Option(
                id="BULLISH_UP",
                description="strong revenue growth, expanding profit margins, and safe balance sheet supporting stock price increase",
            ),
            Option(
                id="BEARISH_DOWN",
                description="rising interest debt burden, stock price decline, and liquidity pressure indicating severe financial risk",
            ),
            Option(
                id="NEUTRAL_SIDEWAYS",
                description="stagnant flat revenue, mature market saturation, with no clear growth catalyst",
            ),
        ],
    )

    # 3. Danh sách 4 ca kiểm thử thực tế từ dữ liệu Capstone
    sim_cases = [
        {
            "case_id": "CASE-1",
            "stock": "FPT (Niên độ 2021)",
            "scenario": "Trường hợp chuẩn mực: Qwen dự đoán UP, Báo cáo đồng thuận tích cực (Consensus Bullish)",
            "qwen_pred": "UP",
            "context": (
                "FPT Corporation reported consolidated revenue of 35,657 billion VND (+19.5% YoY) and pre-tax profit of "
                "6,337 billion VND (+20.4% YoY), beating management targets. Core growth was driven by global IT services "
                "and digital transformation surging +72% with major US cloud deals. Capital structure is safe with strong "
                "positive operating cash flows and zero litigation risks."
            ),
            "expected_outcome": "APPROVED",
        },
        {
            "case_id": "CASE-2",
            "stock": "DXG (Niên độ 2024)",
            "scenario": "Bẫy Ảo Giác (Hallucination): Qwen dự đoán UP nhưng Báo cáo chỉ ra gánh nặng nợ vay và giá giảm -19.5%",
            "qwen_pred": "UP",
            "context": (
                "Dat Xanh Group faces heavy financial headwinds with capitalized borrowing costs surging to 128.9 billion VND. "
                "Stock price declined -19.57% amid property market downturn. The company relies heavily on equity rights issuance "
                "and customer deposit financing to cover working capital cashflow deficits."
            ),
            "expected_outcome": "BLOCKED",
        },
        {
            "case_id": "CASE-3",
            "stock": "TIN ĐỒN DIỄN ĐÀN (Market Rumor)",
            "scenario": "Bẫy Đoán Mò: Tin đồn mạng xã hội không có số liệu tài chính kiểm chứng (Abstention Test)",
            "qwen_pred": "UP",
            "context": (
                "Rumors on social media stock forums claim big players are manipulating the stock to pump 5 consecutive "
                "ceiling sessions, urging retail investors to go all-in without any audited financial numbers."
            ),
            "expected_outcome": "ABSTAIN",
        },
        {
            "case_id": "CASE-4",
            "stock": "VNM (Tăng trưởng bão hòa)",
            "scenario": "Tín hiệu đi ngang: Qwen dự đoán UP nhưng doanh thu bão hòa, động lực yếu (Low Confidence)",
            "qwen_pred": "UP",
            "context": (
                "Vinamilk maintains stable domestic dairy market share, but annual revenue remains flat with negligible growth. "
                "High market penetration limits domestic expansion and raw material cost inflation suppresses margins."
            ),
            "expected_outcome": "WARNING",
        },
    ]

    # Warmup GPU
    _ = engine.evaluate(context="Warmup test.", queries=[guardrail_query])
    if device == "cuda":
        torch.cuda.synchronize()

    # 4. Thực thi thẩm định qua OpenJev Guardrail
    CONFIDENCE_SAFETY_THRESHOLD = 0.55

    print("=" * 90)
    print("🔍 KẾT QUẢ THẨM ĐỊNH CHI TIẾT CỦA OPENJEV GUARDRAIL TRÊN GPU RTX 4060")
    print("=" * 90)

    for c in sim_cases:
        stock = c["stock"]
        qwen_pred = c["qwen_pred"]
        ctx = c["context"]

        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        batch_res = engine.evaluate(context=ctx, queries=[guardrail_query])

        if device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        decision = batch_res.results[0]
        guardrail_verdict = decision.selected_id
        conf = decision.selected_probability
        is_abstention = decision.is_abstention

        # Phân loại trạng thái Guardrail
        if is_abstention:
            status = "🟠 ABSTENTION (TỪ CHỐI RA QUYẾT ĐỊNH)"
            action = "CHẶN LỆNH: OpenJev nhận diện dữ liệu là tin đồn hoặc thiếu số liệu tài chính xác thực."
            badge = "🛡️ [BẢO TOÀN VỐN - KHUYẾN NGHỊ ĐỨNG NGOÀI]"
        elif qwen_pred == "UP" and guardrail_verdict == "BEARISH_DOWN":
            status = "🔴 BÁO ĐỘNG ẢO GIÁC (HALLUCINATION DETECTED)"
            action = f"CHẶN ĐỨNG TÍN HIỆU: Qwen dự đoán UP nhưng thực tế dữ liệu chỉ ra rủi ro nợ và giảm giá (BEARISH_DOWN)."
            badge = "🚫 [CHẶN BẮN TÍN HIỆU SAI LỆCH RA DASHBOARD]"
        elif qwen_pred == "UP" and guardrail_verdict == "BULLISH_UP":
            if conf >= CONFIDENCE_SAFETY_THRESHOLD:
                status = "🟢 PHÊ DUYỆT AN TOÀN (DISPATCH APPROVED)"
                action = f"XÁC NHẬN: Nhãn dự đoán UP và bản chất tài chính BULLISH đồng thuận cao."
                badge = "🚀 [ĐỦ ĐIỀU KIỆN ĐƯA LÊN STREAMLIT DASHBOARD]"
            else:
                status = "🟡 CẢNH BÁO RỦI RO (LOW CONFIDENCE)"
                action = f"CẢNH BÁO: Tín hiệu đồng thuận nhưng độ tự tin hiệu chuẩn chỉ đạt {conf:.1%}."
                badge = "⚠️ [GẮN CỜ THẬN TRỌNG CHO NHÀ ĐẦU TƯ]"
        else:
            status = "🟡 CẢNH BÁO ĐI NGANG (NEUTRAL WARNING)"
            action = f"Dữ liệu phản ánh doanh thu bão hòa ({guardrail_verdict}), không đủ động lực tăng trưởng."
            badge = "⚠️ [KHÔNG NÊN GIẢI NGÂN]"

        print(f"\n[{c['case_id']}] MÃ: {stock}")
        print(f"Kịch bản          : {c['scenario']}")
        print(f"Qwen-0.5B Dự đoán : Nhãn = [{qwen_pred}]")
        print(f"Trích xuất BCTC   : \"{ctx[:110]}...\"")
        print(f"👉 OpenJev Phán   : \033[1;36m{guardrail_verdict}\033[0m (Độ tự tin hiệu chuẩn: \033[1;32m{conf * 100:.2f}%\033[0m)")
        print(f"⚡ GPU Latency     : \033[1;33m{lat_ms:.2f} ms\033[0m")
        print(f"🛡️ Guardrail Kết luận: {status}")
        print(f"🎯 Hành động xử lý : {action}\n   {badge}")
        print("-" * 90)

    print("\n✅ HOÀN THÀNH TOÀN BỘ 4 KỊCH BẢN THỰC NGHIỆM!")

if __name__ == "__main__":
    main()
