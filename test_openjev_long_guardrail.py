"""Comprehensive Real-world Simulation: OpenJev as a Calibrated Financial Guardrail

Tests OpenJev (ModernBERT-151M) on long, full-length financial explanations (~250-400 words each)
reflecting realistic multi-paragraph outputs from Qwen2.5-0.5B Self-Rationalization Head.
"""

from __future__ import annotations

import time
import torch
from rlcd import DecisionEngine, Choice, Option

def run_simulation():
    print("=" * 95)
    print("🛡️ KIỂM THỬ THỰC TẾ: OPENJEV FINANCIAL GUARDRAIL VỚI BÀI GIẢI TRÌNH DÀI (LONG CONTEXT)")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phần cứng GPU        : {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"Mô hình Guardrail    : OpenJev (heman10x/rlcd-modernbert-151m, ModernBERT-151M)")
    print(f"Cấu hình Context Max : 1024 tokens (hỗ trợ văn bản phân tích tài chính nhiều đoạn)")
    print("-" * 95)

    # 1. Khởi tạo Engine với max_length=1024 tokens
    t0 = time.perf_counter()
    engine = DecisionEngine(
        model_name_or_path="heman10x/rlcd-modernbert-151m",
        device=device,
        max_length=1024
    )
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"✅ Guardrail Engine nạp vào GPU thành công trong {init_ms:.1f} ms\n")

    # 2. Định nghĩa Schema phán quyết tài chính đa chiều
    guardrail_query = Choice(
        id="financial_rationalization_evaluation",
        question="What is the comprehensive financial health, operational momentum, and investment trajectory concluded in this detailed analysis?",
        options=[
            Option(
                id="BULLISH_UP",
                description="robust revenue expansion, surging operating cashflows, widening profit margins, and resilient capital structure supporting stock price appreciation",
            ),
            Option(
                id="BEARISH_DOWN",
                description="escalating debt burden, liquidity distress, declining margins, delayed projects, or operational headwinds indicating severe downside risk",
            ),
            Option(
                id="NEUTRAL_SIDEWAYS",
                description="mature business model with stagnant flat revenue growth, market saturation, and lack of clear upside or downside catalysts",
            ),
        ],
    )

    # 3. Bộ dữ liệu 4 kịch bản thực tế với bài giải trình dài (250 - 350 từ/bài)
    cases = [
        {
            "id": "CASE-1",
            "stock": "FPT (Tập đoàn FPT - FY2021)",
            "scenario": "Trường hợp chuẩn mực: Qwen dự đoán UP, Bài giải trình dài phân tích toàn diện các động lực tăng trưởng",
            "qwen_pred": "UP",
            "explanation": (
                "In fiscal year 2021, FPT Corporation achieved extraordinary business momentum, recording consolidated "
                "net revenue of 35,657 billion VND (+19.5% YoY) and pre-tax profit of 6,337 billion VND (+20.4% YoY), "
                "surpassing shareholder targets by 103% and 102% respectively. The Global IT Services division served as "
                "the primary growth engine, contributing 44% of consolidated pre-tax profit. International expansion was "
                "led by a remarkable 52% surge in United States revenue and 27% growth across the Asia-Pacific region. "
                "Strategic enterprise digital transformation initiatives surged 72%, heavily anchored by cloud computing "
                "deployments and the execution of 19 mega-contracts exceeding 5 million USD each. Simultaneously, the "
                "Telecommunications segment remained a reliable cash-generative pillar, expanding broadband subscribers "
                "by 16% and widening operating margins to 17.5%. On the balance sheet side, FPT maintained exceptional liquidity "
                "discipline throughout macro shocks, cycling over 16,000 billion VND in debt obligations with top commercial "
                "banks while preserving strong positive operating cash flows. The Supervisory Board reported zero material "
                "off-balance-sheet commitments, litigation disputes, or adverse post-balance-sheet adjustments. In conclusion, "
                "the confluence of double-digit earnings growth, expanding international market share, and balance sheet safety "
                "provides unequivocal fundamental support for an upward stock price trajectory."
            ),
        },
        {
            "id": "CASE-2",
            "stock": "DXG (Tập đoàn Đất Xanh - FY2024)",
            "scenario": "Bẫy Ảo Giác (Hallucination Alert): Qwen dự đoán UP nhưng bài văn dài lại phân tích rủi ro nợ vay và dòng tiền căng thẳng",
            "qwen_pred": "UP",
            "explanation": (
                "Dat Xanh Group concluded fiscal year 2024 amidst severe headwinds across the domestic real estate sector. "
                "While net after-tax profit reached 453 billion VND from handing over Opal Skyline apartments, the balance sheet "
                "reveals profound structural liquidity pressures. Total capitalized borrowing costs escalated sharply to 128.9 "
                "billion VND, driven by stalled progress and extended absorption timelines across flagship developments such as "
                "Gem Sky World and Gem Riverside. The company bears high floating interest rates ranging from 5% to 10.8% across "
                "commercial bank syndicates, forcing management to pledge project land-use rights and 25 billion VND in bank bonds "
                "to secure short-term refinancing. Facing persistent operating cashflow deficits, DXG had to rely on dilutive equity "
                "rights issuances to existing shareholders and substantial customer deposit advances to sustain ongoing project "
                "liabilities. Throughout 2024, the stock price plummeted -19.57%, reflecting investor skepticism regarding leverage "
                "rollover risks and prolonged regulatory hurdles. In summary, escalating interest burdens, debt refinancing reliance, "
                "and depressed market valuations highlight severe operational headwinds and substantial financial vulnerability."
            ),
        },
        {
            "id": "CASE-3",
            "stock": "SPECULATIVE_PENNY (Cổ phiếu Đầu cơ / Tin đồn)",
            "scenario": "Bẫy Đoán Mò (Abstention Test): Bài văn dài tràn ngập tin đồn hội nhóm mạng xã hội, hoàn toàn thiếu số liệu tài chính",
            "qwen_pred": "UP",
            "explanation": (
                "Confidential rumors circulating heavily across social media investment channels and insider telegram groups "
                "indicate that an aggressive market maker syndicate is quietly accumulating massive shares ahead of an impending "
                "corporate takeover. Unofficial sources claim an offshore fund will acquire a 51% controlling stake at double the "
                "current market price, triggering an anticipated series of five consecutive limit-up trading sessions. Multiple "
                "brokerage chatrooms are actively urging retail investors to execute aggressive full-margin buy orders before the "
                "public disclosure deadline. However, no audited financial statements, certified regulatory filings to the State "
                "Securities Commission, or verified disclosures from the Board of Directors exist to substantiate these takeover "
                "speculations, and company earnings history remains completely opaque with unverified revenue numbers."
            ),
        },
        {
            "id": "CASE-4",
            "stock": "VNM (Vinamilk - Tăng trưởng bão hòa)",
            "scenario": "Cảnh báo Đi ngang: Qwen dự đoán UP nhưng bài phân tích chỉ ra doanh nghiệp bão hòa, doanh thu đi ngang",
            "qwen_pred": "UP",
            "explanation": (
                "Vinamilk concluded the operating year with resilient operational performance, maintaining its commanding "
                "position as the domestic dairy market leader. However, annual top-line consolidated revenue experienced flat "
                "year-over-year momentum, advancing by an uninspiring +0.5% as high market penetration across core urban segments "
                "severely limits domestic volume expansion. Operating profit margins were compressed by persistent raw milk powder "
                "import cost inflation and intensified competition from foreign premium dairy brands. Capital expenditure was "
                "restrained to routine factory maintenance, with no transformative M&A transactions or new overseas category catalysts "
                "recorded. While the company maintains an unleveraged debt-free balance sheet and consistent cash dividend payouts, "
                "the lack of revenue catalysts and stagnant market demand suggest the stock will continue to trade in a narrow "
                "sideways consolidation channel without directional breakout potential."
            ),
        },
    ]

    # Warmup GPU
    _ = engine.evaluate(context="System warmup query.", queries=[guardrail_query])
    if device == "cuda":
        torch.cuda.synchronize()

    # 4. Chạy kiểm định Guardrail trên GPU
    print("=" * 95)
    print("🎯 KẾT QUẢ THẨM ĐỊNH THỰC TẾ CỦA OPENJEV GUARDRAIL VỚI BÀI VĂN DÀI")
    print("=" * 95)

    SAFETY_THRESHOLD = 0.55

    for c in cases:
        words = len(c["explanation"].split())
        chars = len(c["explanation"])

        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        res = engine.evaluate(context=c["explanation"], queries=[guardrail_query]).results[0]

        if device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        verdict = res.selected_id
        conf = res.selected_probability
        is_abstain = res.is_abstention

        # Xử lý Logic Phán Quyết An Toàn
        if is_abstain:
            guard_status = "🟠 ABSTENTION TRIGGERED (KÍCH HOẠT TỪ CHỐI RA LỆNH)"
            action = "CHẶN KHUYẾN NGHỊ: OpenJev phát hiện bài văn thiếu số liệu kiểm toán (Tin đồn không căn cứ)."
            badge = "🛡️ [BẢO TOÀN VỐN - KHUYẾN NGHỊ ĐỨNG NGOÀI THỊ TRƯỜNG]"
        elif c["qwen_pred"] == "UP" and verdict == "BEARISH_DOWN":
            guard_status = "🔴 BÁO ĐỘNG ẢO GIÁC (HALLUCINATION CONTRADICTION DETECTED)"
            action = f"CHẶN ĐỨNG: Qwen dự đoán UP nhưng toàn bộ bài văn giải trình chỉ ra rủi ro nợ và giảm giá (BEARISH_DOWN: {conf:.1%})."
            badge = "🚫 [CHẶN TÍN HIỆU SAI LỆCH - BẢO VỆ NHÀ ĐẦU TƯ]"
        elif c["qwen_pred"] == "UP" and verdict == "BULLISH_UP":
            if conf >= SAFETY_THRESHOLD:
                guard_status = "🟢 PHÊ DUYỆT AN TOÀN (SAFE TO DISPATCH)"
                action = f"XÁC NHẬN: Nhãn dự đoán UP và bài phân tích dài BULLISH đồng thuận toán học cao (Xác suất hiệu chuẩn: {conf:.1%})."
                badge = "🚀 [ĐỦ ĐIỀU KIỆN ĐƯA LÊN STREAMLIT DASHBOARD]"
            else:
                guard_status = "🟡 CẢNH BÁO RỦI RO (LOW CONFIDENCE WARNING)"
                action = f"Đồng thuận nhưng độ tự tin chỉ đạt {conf:.1%} (< {SAFETY_THRESHOLD:.1%})."
                badge = "⚠️ [GẮN CỜ THẬN TRỌNG CHO NHÀ ĐẦU TƯ]"
        else:
            guard_status = "🟡 CẢNH BÁO THỊ TRƯỜNG ĐI NGANG (NEUTRAL / SIDEWAYS)"
            action = f"Bài phân tích chỉ ra doanh nghiệp bão hòa, thiếu động lực tăng trưởng (NEUTRAL: {conf:.1%})."
            badge = "⚠️ [KHÔNG NÊN GIẢI NGÂN]"

        print(f"\n[{c['id']}] MÃ: {c['stock']}")
        print(f"Kịch bản            : {c['scenario']}")
        print(f"Độ dài bài giải trình: \033[1;35m{words} từ\033[0m (~{chars} ký tự)")
        print(f"Qwen-0.5B Dự đoán   : Nhãn = [\033[1;34m{c['qwen_pred']}\033[0m]")
        print(f"👉 OpenJev Thẩm định: \033[1;36m{verdict}\033[0m (Độ tự tin hiệu chuẩn: \033[1;32m{conf * 100:.2f}%\033[0m)")
        print(f"⚡ GPU Latency       : \033[1;33m{lat_ms:.2f} ms\033[0m")
        print(f"🛡️ Guardrail Kết luận: {guard_status}")
        print(f"🎯 Hành động xử lý   : {action}\n   {badge}")
        print("-" * 95)

    print("\n✅ HOÀN THÀNH XUẤT SẮC TOÀN BỘ BÀI TEST THỰC TẾ VỚI VĂN BẢN DÀI!")

if __name__ == "__main__":
    run_simulation()
