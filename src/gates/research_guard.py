"""
ResearchGuard Gate (V1.1)

통계 유의성 검증 게이트:
- 정상성 검정 (ADF)
- 교차상관 분석
- 부트스트랩 유의성 검정
- 다중검정 보정
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger

from src.config import SIGNIFICANCE_CONFIG
from src.state import (
    SignificanceReport,
    StationarityResult,
    CrossCorrelationResult,
)
from src.utils.math import (
    test_stationarity,
    make_stationary,
    cross_correlation,
    find_best_lag,
    bootstrap_correlation_test,
    multiple_testing_correction,
)


class ResearchGuard:
    """
    통계 유의성 검증 게이트
    
    통과 기준:
    1. 정상성: 차분 후 ADF p < 0.05
    2. 유의성: 부트스트랩 p < 0.05
    3. 다중검정: 보정 후 p < 0.05
    
    하나라도 실패하면 백테스트 진행 불가
    """
    
    def __init__(self, config=None):
        self.config = config or SIGNIFICANCE_CONFIG
    
    def validate(
        self,
        kci_monthly: pd.Series,
        cpi_monthly: pd.Series,
    ) -> SignificanceReport:
        """
        통계 유의성 검증 수행
        
        Args:
            kci_monthly: 월간 치킨지수
            cpi_monthly: 월간 CPI
        
        Returns:
            SignificanceReport
        """
        logger.info("통계 유의성 검증 시작")
        
        failure_reasons = []
        
        # ===== 1. 정상성 검정 =====
        logger.debug("Step 1: 정상성 검정 (ADF)")
        
        # 원본으로 먼저 테스트
        kci_adf_raw = test_stationarity(kci_monthly, self.config.adf_p_threshold)
        cpi_adf_raw = test_stationarity(cpi_monthly, self.config.adf_p_threshold)
        
        # 차분 적용
        kci_diff = make_stationary(kci_monthly, method="pct_change")
        cpi_diff = make_stationary(cpi_monthly, method="pct_change")
        
        kci_adf = test_stationarity(kci_diff, self.config.adf_p_threshold)
        cpi_adf = test_stationarity(cpi_diff, self.config.adf_p_threshold)
        
        stationarity_pass = kci_adf["is_stationary"] and cpi_adf["is_stationary"]
        
        if not stationarity_pass:
            if not kci_adf["is_stationary"]:
                failure_reasons.append(f"KCI 정상성 실패 (p={kci_adf['p_value']:.4f})")
            if not cpi_adf["is_stationary"]:
                failure_reasons.append(f"CPI 정상성 실패 (p={cpi_adf['p_value']:.4f})")
        
        stationarity_result = StationarityResult(
            kci_adf_statistic=kci_adf["statistic"],
            kci_p_value=kci_adf["p_value"],
            cpi_adf_statistic=cpi_adf["statistic"],
            cpi_p_value=cpi_adf["p_value"],
            passed=stationarity_pass,
        )
        
        logger.debug(f"정상성 검정: KCI p={kci_adf['p_value']:.4f}, CPI p={cpi_adf['p_value']:.4f}")
        
        # ===== 2. 교차상관 분석 =====
        logger.debug("Step 2: 교차상관 분석")
        
        lag_min, lag_max = self.config.lag_range
        correlations = cross_correlation(kci_diff, cpi_diff, max_lag=abs(lag_max))
        best_lag, best_corr = find_best_lag(correlations)
        
        cross_corr_result = CrossCorrelationResult(
            best_lag=best_lag,
            best_correlation=best_corr,
            all_correlations=correlations,
        )
        
        logger.info(f"최적 래그: {best_lag}개월, 상관계수: {best_corr:.4f}")
        
        # ===== 3. 부트스트랩 유의성 검정 =====
        logger.debug("Step 3: 부트스트랩 유의성 검정")
        
        bootstrap_result = bootstrap_correlation_test(
            kci_diff,
            cpi_diff,
            n_bootstrap=self.config.bootstrap_n,
            lag=best_lag,
        )
        
        bootstrap_pass = bootstrap_result["p_value"] < self.config.correlation_p_threshold
        
        if not bootstrap_pass:
            failure_reasons.append(
                f"부트스트랩 유의성 실패 (p={bootstrap_result['p_value']:.4f} >= {self.config.correlation_p_threshold})"
            )
        
        logger.debug(f"부트스트랩 p-value: {bootstrap_result['p_value']:.4f}")
        
        # ===== 4. 다중검정 보정 =====
        logger.debug("Step 4: 다중검정 보정")
        
        # 각 래그에 대한 p-value 계산
        raw_p_values = {}
        for lag in range(lag_min, lag_max + 1):
            lag_result = bootstrap_correlation_test(
                kci_diff, cpi_diff, 
                n_bootstrap=min(self.config.bootstrap_n, 500),  # 속도 위해 줄임
                lag=lag
            )
            raw_p_values[lag] = lag_result["p_value"]
        
        # 보정
        correction_result = multiple_testing_correction(
            raw_p_values,
            method=self.config.multiple_testing_method,
            alpha=self.config.correlation_p_threshold,
        )
        
        corrected_p_values = correction_result["corrected_p_values"]
        significant_lags = correction_result["significant_lags"]
        
        multiple_testing_pass = len(significant_lags) > 0
        
        if not multiple_testing_pass:
            failure_reasons.append(
                f"다중검정 보정 후 유의한 래그 없음 (method={self.config.multiple_testing_method})"
            )
        
        logger.debug(f"유의한 래그 (보정 후): {significant_lags}")
        
        # ===== 최종 판정 =====
        final_pass = stationarity_pass and bootstrap_pass and multiple_testing_pass
        
        report = SignificanceReport(
            stationarity=stationarity_result,
            cross_correlation=cross_corr_result,
            bootstrap_p_value=bootstrap_result["p_value"],
            observed_correlation=bootstrap_result["observed_corr"],
            raw_p_values=raw_p_values,
            corrected_p_values=corrected_p_values,
            multiple_testing_method=self.config.multiple_testing_method,
            final_pass=final_pass,
            failure_reason="; ".join(failure_reasons) if failure_reasons else None,
        )
        
        status = "PASS ✓" if final_pass else "FAIL ✗"
        logger.info(f"통계 유의성 검증 완료: {status}")
        
        if not final_pass:
            logger.warning(f"실패 사유: {failure_reasons}")
        
        return report
    
    def is_passed(self, report: SignificanceReport) -> bool:
        """게이트 통과 여부"""
        return report["final_pass"]
    
    def get_recommended_lag(self, report: SignificanceReport) -> int:
        """권장 래그 반환"""
        return report["cross_correlation"]["best_lag"]


class MockResearchGuard:
    """테스트용 Mock (항상 통과)"""
    
    def validate(
        self,
        kci_monthly: pd.Series,
        cpi_monthly: pd.Series,
    ) -> SignificanceReport:
        return SignificanceReport(
            stationarity=StationarityResult(
                kci_adf_statistic=-4.5,
                kci_p_value=0.001,
                cpi_adf_statistic=-3.8,
                cpi_p_value=0.005,
                passed=True,
            ),
            cross_correlation=CrossCorrelationResult(
                best_lag=2,
                best_correlation=0.65,
                all_correlations={-3: 0.2, -2: 0.3, -1: 0.4, 0: 0.5, 1: 0.55, 2: 0.65, 3: 0.5},
            ),
            bootstrap_p_value=0.012,
            observed_correlation=0.65,
            raw_p_values={-3: 0.3, -2: 0.15, -1: 0.08, 0: 0.05, 1: 0.03, 2: 0.012, 3: 0.06},
            corrected_p_values={-3: 0.5, -2: 0.3, -1: 0.15, 0: 0.1, 1: 0.06, 2: 0.035, 3: 0.12},
            multiple_testing_method="fdr_bh",
            final_pass=True,
            failure_reason=None,
        )
    
    def is_passed(self, report: SignificanceReport) -> bool:
        return True
