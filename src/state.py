"""
KCI Agent State Definition (V1.1)

LangGraph State 스키마 정의
- 데이터 품질 레포트
- 통계 검증 레포트  
- 백테스트 레포트
- 에러 처리
"""

from typing import TypedDict, Optional, Literal, Any
from enum import Enum
from datetime import datetime
import pandas as pd


# ===== Enums =====

class ErrorType(str, Enum):
    """에러 타입 분류"""
    NONE = "none"
    COLLECTION_FAILED = "collection_failed"
    DATA_MISSING = "data_missing"
    VALIDATION_FAILED = "validation_failed"
    SIGNIFICANCE_FAILED = "significance_failed"
    BACKTEST_FAILED = "backtest_failed"


class SignalType(str, Enum):
    """거래 신호"""
    LONG = "LONG"
    FLAT = "FLAT"
    SHORT = "SHORT"


class GateStatus(str, Enum):
    """게이트 통과 상태"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


# ===== Report TypedDicts =====

class SourceConsistency(TypedDict):
    """소스 간 정합성 체크 결과"""
    status: str  # OK, ALERT, WARNING
    retail_change: float
    wholesale_change: float
    reason: Optional[str]


class DataQualityReport(TypedDict):
    """데이터 품질 리포트"""
    # 수집 메타
    collection_timestamp: str
    brands_collected: list[str]
    
    # 품질 지표
    missing_rate: dict[str, float]  # 브랜드별 결측률
    outlier_count: int
    outliers_detected: list[dict]  # 이상치 상세
    
    # 스키마 검증
    schema_valid: bool
    schema_errors: list[str]
    
    # 표본 변경 감지
    sample_change_detected: bool
    sample_changes: list[dict]
    
    # 소스 정합성
    source_consistency: SourceConsistency
    
    # 최종 판정
    overall_status: GateStatus
    failure_reasons: list[str]


class StationarityResult(TypedDict):
    """정상성 검정 결과"""
    kci_adf_statistic: float
    kci_p_value: float
    cpi_adf_statistic: float
    cpi_p_value: float
    passed: bool


class CrossCorrelationResult(TypedDict):
    """교차상관 결과"""
    best_lag: int  # 월 단위
    best_correlation: float
    all_correlations: dict[int, float]  # lag -> correlation


class SignificanceReport(TypedDict):
    """통계 유의성 리포트"""
    # 정상성 검정
    stationarity: StationarityResult
    
    # 교차상관
    cross_correlation: CrossCorrelationResult
    
    # 부트스트랩 검정
    bootstrap_p_value: float
    observed_correlation: float
    
    # 다중검정 보정
    raw_p_values: dict[int, float]
    corrected_p_values: dict[int, float]
    multiple_testing_method: str
    
    # 최종 판정
    final_pass: bool
    failure_reason: Optional[str]


class TradeRecord(TypedDict):
    """개별 거래 기록"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    exit_reason: str  # "target", "stop_loss", "time_limit"


class FailedCase(TypedDict):
    """실패 케이스 상세"""
    date: str
    loss_pct: float
    market_context: str  # 실패 원인


class WalkForwardFold(TypedDict):
    """Walk-forward 단일 Fold 결과"""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    test_return: float
    degradation: float


class WalkForwardResult(TypedDict):
    """Walk-forward 전체 결과"""
    folds: list[WalkForwardFold]
    avg_oos_sharpe: float
    avg_degradation: float
    stability_score: float  # OOS/IS Sharpe 비율


class BenchmarkComparison(TypedDict):
    """벤치마크 비교"""
    strategy_cagr: float
    benchmark_cagr: float
    excess_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_dd: float
    benchmark_max_dd: float


class BacktestReport(TypedDict):
    """백테스트 리포트"""
    # 거래 요약
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # 수익률 지표
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # 거래 통계
    avg_return: float
    avg_winning_return: float
    avg_losing_return: float
    avg_holding_days: float
    
    # 비용
    total_commission: float
    total_slippage: float
    
    # 상세 기록
    trades: list[TradeRecord]
    failed_cases: list[FailedCase]
    
    # Walk-forward
    walk_forward: WalkForwardResult
    
    # 벤치마크 대비
    benchmark_comparison: BenchmarkComparison
    
    # 신뢰구간
    return_ci_95: tuple[float, float]


class FinalReport(TypedDict):
    """최종 리포트"""
    # 메타
    generated_at: str
    version: str
    
    # 현재 신호
    current_signal: SignalType
    signal_confidence: float
    
    # KCI 현황
    current_kci: float
    kci_change_weekly: float
    kci_change_monthly: float
    
    # 각 섹션 데이터
    signal_rationale: dict  # 신호 근거
    data_quality_summary: dict  # 데이터 품질 요약
    failed_cases_summary: list[dict]  # 실패 케이스
    benchmark_summary: dict  # 벤치마크 대비
    uncertainty: dict  # 불확실성 정보
    
    # 면책 조항
    disclaimer: str


# ===== Main State =====

class KCIState(TypedDict):
    """KCI Agent 메인 State"""
    
    # ===== 입력 =====
    date_range: tuple[str, str]
    target_brands: list[str]
    run_mode: Literal["full", "data_only", "backtest_only"]
    
    # ===== 데이터 수집 =====
    raw_chicken_prices: Optional[Any]  # pd.DataFrame (직렬화 이슈로 Any)
    raw_cpi_data: Optional[Any]
    raw_wholesale_data: Optional[Any]
    collection_timestamp: Optional[str]
    
    # ===== 데이터 품질 (V1.1) =====
    cleaned_data: Optional[Any]  # 정제된 데이터
    data_quality_report: Optional[DataQualityReport]
    data_quality_pass: bool
    
    # ===== 지수 계산 =====
    kci_weekly: Optional[Any]  # pd.Series
    kci_monthly: Optional[Any]
    cpi_monthly: Optional[Any]
    
    # ===== 통계 검증 (V1.1) =====
    significance_report: Optional[SignificanceReport]
    research_guard_pass: bool
    
    # ===== 백테스트 =====
    backtest_report: Optional[BacktestReport]
    current_signal: Optional[SignalType]
    
    # ===== 에러 처리 (V1.1 확장) =====
    error_type: ErrorType
    error_message: Optional[str]
    error_details: Optional[dict]
    retry_count: int
    
    # ===== 최종 출력 =====
    final_report: Optional[FinalReport]


def create_initial_state(
    start_date: str = "2020-01-01",
    end_date: str = None,
    brands: list[str] = None,
    run_mode: str = "full"
) -> KCIState:
    """초기 State 생성"""
    from datetime import datetime
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if brands is None:
        brands = ["BBQ", "교촌", "BHC"]
    
    return KCIState(
        # 입력
        date_range=(start_date, end_date),
        target_brands=brands,
        run_mode=run_mode,
        
        # 데이터 수집
        raw_chicken_prices=None,
        raw_cpi_data=None,
        raw_wholesale_data=None,
        collection_timestamp=None,
        
        # 데이터 품질
        cleaned_data=None,
        data_quality_report=None,
        data_quality_pass=False,
        
        # 지수 계산
        kci_weekly=None,
        kci_monthly=None,
        cpi_monthly=None,
        
        # 통계 검증
        significance_report=None,
        research_guard_pass=False,
        
        # 백테스트
        backtest_report=None,
        current_signal=None,
        
        # 에러 처리
        error_type=ErrorType.NONE,
        error_message=None,
        error_details=None,
        retry_count=0,
        
        # 최종 출력
        final_report=None,
    )
