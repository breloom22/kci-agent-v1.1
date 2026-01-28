"""
Statistical Utilities for KCI Agent

- 정상성 검정
- 교차상관 분석
- 부트스트랩 검정
- 성과 지표 계산
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
from typing import Optional
from loguru import logger


# ===== 정상성 검정 =====

def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    ADF (Augmented Dickey-Fuller) 정상성 검정
    
    Args:
        series: 시계열 데이터
        alpha: 유의수준
    
    Returns:
        dict: {statistic, p_value, critical_values, is_stationary}
    """
    # NaN 제거
    clean_series = series.dropna()
    
    if len(clean_series) < 20:
        logger.warning(f"시계열 길이 부족: {len(clean_series)}")
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "critical_values": {},
            "is_stationary": False,
            "error": "insufficient_data"
        }
    
    result = adfuller(clean_series, autolag='AIC')
    
    return {
        "statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < alpha
    }


def make_stationary(series: pd.Series, method: str = "diff") -> pd.Series:
    """
    시계열 정상화
    
    Args:
        series: 원본 시계열
        method: "diff" (차분) 또는 "log_return" (로그 수익률)
    
    Returns:
        정상화된 시계열
    """
    if method == "diff":
        return series.diff().dropna()
    elif method == "log_return":
        return np.log(series / series.shift(1)).dropna()
    elif method == "pct_change":
        return series.pct_change().dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


# ===== 교차상관 분석 =====

def cross_correlation(
    x: pd.Series, 
    y: pd.Series, 
    max_lag: int = 6
) -> dict[int, float]:
    """
    교차상관 계산 (x가 y를 선행하는지 분석)
    
    Args:
        x: 선행 변수 (KCI)
        y: 후행 변수 (CPI)
        max_lag: 최대 래그 (양방향)
    
    Returns:
        dict: {lag: correlation}
        - lag > 0: x가 y를 선행
        - lag < 0: y가 x를 선행
    """
    # 인덱스 정렬
    common_idx = x.index.intersection(y.index)
    x_aligned = x.loc[common_idx].values
    y_aligned = y.loc[common_idx].values
    
    correlations = {}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # y가 선행 (y를 앞으로 이동)
            x_slice = x_aligned[-lag:]
            y_slice = y_aligned[:lag]
        elif lag > 0:
            # x가 선행 (x를 앞으로 이동)
            x_slice = x_aligned[:-lag]
            y_slice = y_aligned[lag:]
        else:
            x_slice = x_aligned
            y_slice = y_aligned
        
        if len(x_slice) > 3:
            corr, _ = stats.pearsonr(x_slice, y_slice)
            correlations[lag] = corr
        else:
            correlations[lag] = np.nan
    
    return correlations


def find_best_lag(correlations: dict[int, float]) -> tuple[int, float]:
    """
    최대 상관계수를 가지는 래그 찾기
    
    Returns:
        (best_lag, best_correlation)
    """
    valid_corrs = {k: v for k, v in correlations.items() if not np.isnan(v)}
    
    if not valid_corrs:
        return 0, 0.0
    
    # 절대값 기준 최대
    best_lag = max(valid_corrs, key=lambda k: abs(valid_corrs[k]))
    return best_lag, valid_corrs[best_lag]


# ===== 유의성 검정 =====

def bootstrap_correlation_test(
    x: pd.Series,
    y: pd.Series,
    n_bootstrap: int = 1000,
    lag: int = 0
) -> dict:
    """
    부트스트랩 상관계수 유의성 검정
    
    Args:
        x, y: 시계열
        n_bootstrap: 부트스트랩 반복 횟수
        lag: 적용할 래그
    
    Returns:
        dict: {observed_corr, p_value, null_distribution_stats}
    """
    # 래그 적용
    if lag > 0:
        x_aligned = x.iloc[:-lag].values
        y_aligned = y.iloc[lag:].values
    elif lag < 0:
        x_aligned = x.iloc[-lag:].values
        y_aligned = y.iloc[:lag].values
    else:
        x_aligned = x.values
        y_aligned = y.values
    
    # 관측 상관계수
    observed_corr, _ = stats.pearsonr(x_aligned, y_aligned)
    
    # 귀무분포 생성 (y를 셔플)
    null_distribution = []
    for _ in range(n_bootstrap):
        y_shuffled = np.random.permutation(y_aligned)
        null_corr, _ = stats.pearsonr(x_aligned, y_shuffled)
        null_distribution.append(null_corr)
    
    null_distribution = np.array(null_distribution)
    
    # 양측 검정 p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_corr))
    
    return {
        "observed_corr": observed_corr,
        "p_value": p_value,
        "null_mean": np.mean(null_distribution),
        "null_std": np.std(null_distribution),
        "ci_95": (np.percentile(null_distribution, 2.5), 
                  np.percentile(null_distribution, 97.5))
    }


def multiple_testing_correction(
    p_values: dict[int, float],
    method: str = "fdr_bh",
    alpha: float = 0.05
) -> dict:
    """
    다중검정 보정
    
    Args:
        p_values: {lag: p_value}
        method: "bonferroni", "fdr_bh" (Benjamini-Hochberg), "holm"
        alpha: 유의수준
    
    Returns:
        dict: {corrected_p_values, significant_lags}
    """
    lags = list(p_values.keys())
    pvals = list(p_values.values())
    
    reject, corrected_pvals, _, _ = multipletests(pvals, alpha=alpha, method=method)
    
    return {
        "corrected_p_values": dict(zip(lags, corrected_pvals)),
        "significant_lags": [lag for lag, sig in zip(lags, reject) if sig],
        "method": method,
        "alpha": alpha
    }


# ===== 성과 지표 =====

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.035,  # 연 3.5%
    periods_per_year: int = 52  # 주간
) -> float:
    """
    Sharpe Ratio 계산
    
    Args:
        returns: 수익률 시계열
        risk_free_rate: 연간 무위험 수익률
        periods_per_year: 연간 기간 수
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.035,
    periods_per_year: int = 52
) -> float:
    """
    Sortino Ratio (하방 리스크만 고려)
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum Drawdown 계산
    
    Args:
        cumulative_returns: 누적 수익률 (1 + return 형태)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calculate_cagr(
    total_return: float,
    years: float
) -> float:
    """
    CAGR (Compound Annual Growth Rate)
    """
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_win_rate(returns: pd.Series) -> float:
    """승률 계산"""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


# ===== Z-Score 정규화 =====

def zscore_normalize(series: pd.Series, window: int = None) -> pd.Series:
    """
    Z-score 정규화
    
    Args:
        series: 원본 시계열
        window: Rolling window (None이면 전체 기간)
    """
    if window:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
    else:
        mean = series.mean()
        std = series.std()
    
    return (series - mean) / std
