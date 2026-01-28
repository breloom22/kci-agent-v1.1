# 🐔 Idiotology KCI Agent

**Korean Chicken Index (KCI) AI Agent System - Research Grade v1.1**

치킨 가격으로 인플레이션을 예측하는 멀티에이전트 시스템

> "치킨이 오르면 물가가 오른다" - 이제 증명할 시간

---

## 📋 Overview

KCI Agent는 LangGraph 기반 멀티에이전트 시스템으로:
- **치킨 가격 데이터** 수집 (배달앱, 공공데이터)
- **치킨 지수(KCI)** 계산 및 CPI와 상관관계 분석
- **통계적 유의성 검정**으로 신호 검증
- **백테스트**로 투자 전략 성과 측정
- **리포트 생성**으로 의사결정 지원

```
┌─────────────────────────────────────────────────────┐
│              ORCHESTRATOR (Claude 4.5)              │
└────────────────────────┬────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│  DATA   │────▶│ DATA QUALITY │────▶│    INDEX    │
│  AGENT  │     │    GATE      │     │    AGENT    │
└─────────┘     └──────────────┘     └──────┬──────┘
                                            │
                                            ▼
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│ REPORT  │◀────│   BACKTEST   │◀────│  RESEARCH   │
│  AGENT  │     │    AGENT     │     │    GUARD    │
└─────────┘     └──────────────┘     └─────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone
git clone https://github.com/breloom22/kci-agent.git
cd kci-agent

# Install dependencies
pip install poetry
poetry install
```

### 2. Configuration

```bash
# 환경변수 설정
cp .env.example .env

# .env 파일 편집
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=sk-xxx
ECOS_API_KEY=your_key
```

### 3. Run

```bash
# Quick Test (Mock 데이터)
python -m src.main --test

# Full Pipeline
python -m src.main --start 2020-01-01 --end 2024-12-31
```

---

## 📊 V1.1 Features

### 데이터 품질 레이어
- ✅ 표준 메뉴 정의 (Canonical Menu)
- ✅ 이상치 탐지 (Z-score, 주간 변동률)
- ✅ 소스 간 정합성 체크
- ✅ 결측치 보간

### 통계 검증 게이트
- ✅ 정상성 검정 (ADF)
- ✅ 교차상관 분석
- ✅ 부트스트랩 유의성 검정
- ✅ 다중검정 보정 (FDR)

### 백테스트 강화
- ✅ 거래비용/슬리피지 반영
- ✅ Look-ahead bias 방지
- ✅ Walk-forward 검증
- ✅ 실패 케이스 기록

### 리포트 설명력
- ✅ 신호 근거 명시
- ✅ 데이터 품질 요약
- ✅ 실패 케이스 분석
- ✅ 벤치마크 대비
- ✅ 불확실성 표시

---

## 📁 Project Structure

```
idiotology-kci-agent/
├── src/
│   ├── agents/
│   │   ├── index_agent.py      # KCI 계산
│   │   ├── backtest_agent.py   # 백테스트
│   │   └── report_agent.py     # 리포트 생성
│   ├── gates/
│   │   ├── data_quality.py     # 데이터 품질 검증
│   │   └── research_guard.py   # 통계 유의성 검증
│   ├── tools/
│   │   ├── apis/
│   │   │   └── ecos.py         # 한국은행 API
│   │   └── scrapers/           # 배달앱 크롤러 (TODO)
│   ├── utils/
│   │   └── math.py             # 통계 함수
│   ├── config.py               # 설정 관리
│   ├── state.py                # State 정의
│   ├── graph.py                # LangGraph 파이프라인
│   └── main.py                 # CLI 진입점
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 처리된 데이터
├── tests/
├── notebooks/
├── pyproject.toml
└── README.md
```

---

## 📈 KCI Formula

```
KCI(t) = Σ wᵢ × (Pᵢ(t) / Pᵢ(base)) × 100
```

| 브랜드 | 대표 메뉴 | 가중치 |
|--------|----------|--------|
| BBQ | 황금올리브치킨 | 35% |
| 교촌 | 교촌오리지널 | 35% |
| BHC | 뿌링클 | 30% |

- 기준일: 2020-01-01 (= 100)
- 수집 주기: 주 1회 (일요일)

---

## 🔬 Research Methodology

### 핵심 가설
> 치킨 가격은 CPI보다 2개월 먼저 움직인다

### 검증 방법
1. **교차상관 분석**: -3 ~ +3개월 래그에서 최대 상관
2. **부트스트랩 검정**: n=1000, p < 0.05
3. **다중검정 보정**: Benjamini-Hochberg FDR

### 투자 전략
- Entry: KCI 주간 변화율 > 2% & 20MA 상향 돌파
- Exit: 14일 보유 또는 -5% 손절
- 타겟: KODEX 물가채권 ETF (A396510)

---

## 📋 Output Example

```
🐔 KCI (Korean Chicken Index) 리포트 v1.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 생성일시: 2025-01-28T10:30:00
📊 현재 KCI: 128.5 (주간 +2.3%, 월간 +3.1%)

═══════════════════════════════════════════
🔍 [1] 신호 근거
═══════════════════════════════════════════
• 신호: 🟢 LONG
• 신뢰도: 78%
• 트리거:
  - 주간 변화율 > 2%: +2.3% ✓
  - 통계적 유의성: p=0.012 ✓

═══════════════════════════════════════════
📈 [4] 벤치마크 대비
═══════════════════════════════════════════
• 전략 CAGR: 12.4%
• 벤치마크 CAGR: 3.2%
• 초과수익: +9.2%p
• Sharpe: 1.18

⚠️ 본 리포트는 투자 조언이 아닙니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🛠️ Development

### TODO (v1.2)
- [ ] 실제 배달앱 크롤러 구현
- [ ] Telegram 알림 연동
- [ ] 자동 스케줄링 (GitHub Actions)
- [ ] 웹 대시보드

### Contributing
PR 환영합니다!

---

## ⚠️ Disclaimer

이 프로젝트는 교육 및 연구 목적입니다. 실제 투자 결정에 사용하지 마세요.
과거 성과가 미래 수익을 보장하지 않습니다.

---

## 📜 License

MIT License
