"""
Index Calculation Agent

KCI (Korean Chicken Index) 계산:
- 브랜드별 가중 지수 계산
- 주간 → 월간 집계
- CPI와 정렬
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from loguru import logger

from src.config import BRAND_CONFIGS, BASE_DATE
from src.state import KCIState


class IndexAgent:
    """
    KCI 계산 에이전트
    
    수식:
    KCI(t) = Σ wᵢ × (Pᵢ(t) / Pᵢ(base)) × 100
    
    - wᵢ: 브랜드 가중치 (BBQ 0.35, 교촌 0.35, BHC 0.30)
    - Pᵢ(base): 기준일(2020-01-01) 가격
    """
    
    def __init__(self):
        self.base_date = pd.to_datetime(BASE_DATE)
        self.weights = {brand: cfg.weight for brand, cfg in BRAND_CONFIGS.items()}
    
    def calculate_kci(
        self,
        chicken_prices: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """
        KCI 계산
        
        Args:
            chicken_prices: 정제된 치킨 가격 데이터
                columns: [date, brand, menu, price]
        
        Returns:
            (kci_weekly, kci_monthly)
        """
        logger.info("KCI 계산 시작")
        
        # 날짜 파싱
        df = chicken_prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # 브랜드별 피벗
        pivot = df.pivot_table(
            index="date",
            columns="brand",
            values="price",
            aggfunc="mean"
        )
        
        # 기준가 설정 (가장 초기 데이터)
        base_prices = {}
        for brand in self.weights.keys():
            if brand in pivot.columns:
                # 기준일에 가장 가까운 데이터
                base_prices[brand] = pivot[brand].iloc[0]
        
        logger.debug(f"기준가: {base_prices}")
        
        # 브랜드별 지수 계산
        for brand in self.weights.keys():
            if brand in pivot.columns and brand in base_prices:
                pivot[f"{brand}_idx"] = (pivot[brand] / base_prices[brand]) * 100
        
        # 가중 평균 KCI 계산
        kci_values = []
        for idx in pivot.index:
            weighted_sum = 0
            total_weight = 0
            
            for brand, weight in self.weights.items():
                col = f"{brand}_idx"
                if col in pivot.columns and not pd.isna(pivot.loc[idx, col]):
                    weighted_sum += weight * pivot.loc[idx, col]
                    total_weight += weight
            
            if total_weight > 0:
                # P0 Fix: 가중평균으로 정규화(결측이 생겨도 total_weight로 재정규화)
                kci_values.append(weighted_sum / total_weight)
            else:
                kci_values.append(np.nan)
        
        kci_weekly = pd.Series(kci_values, index=pivot.index, name="KCI")
        kci_weekly = kci_weekly.sort_index()
        
        # 월간 집계
        kci_monthly = kci_weekly.resample("ME").mean()
        kci_monthly.name = "KCI_monthly"
        
        logger.info(f"KCI 계산 완료: 주간 {len(kci_weekly)}건, 월간 {len(kci_monthly)}건")
        logger.debug(f"KCI 범위: {kci_weekly.min():.2f} ~ {kci_weekly.max():.2f}")
        
        return kci_weekly, kci_monthly
    
    def align_with_cpi(
        self,
        kci_monthly: pd.Series,
        cpi_monthly: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        KCI와 CPI 시계열 정렬
        
        Args:
            kci_monthly: 월간 KCI
            cpi_monthly: 월간 CPI
        
        Returns:
            (aligned_kci, aligned_cpi)
        """
        # 공통 인덱스 찾기
        kci_idx = pd.DatetimeIndex(kci_monthly.index)
        cpi_idx = pd.DatetimeIndex(cpi_monthly.index)
        
        common_idx = kci_idx.intersection(cpi_idx)
        
        if len(common_idx) == 0:
            logger.warning("KCI와 CPI 간 겹치는 기간 없음")
            return kci_monthly, cpi_monthly
        
        aligned_kci = kci_monthly.loc[common_idx]
        aligned_cpi = cpi_monthly.loc[common_idx]
        
        logger.info(f"시계열 정렬 완료: {len(common_idx)}개월 ({common_idx[0]} ~ {common_idx[-1]})")
        
        return aligned_kci, aligned_cpi
    
    def run(self, state: KCIState) -> dict:
        """
        LangGraph 노드 실행 함수
        
        Args:
            state: 현재 상태
        
        Returns:
            상태 업데이트 dict
        """
        try:
            # 정제된 데이터 사용
            cleaned_data = state.get("cleaned_data")
            
            if cleaned_data is None:
                logger.error("정제된 데이터 없음")
                return {
                    "error_type": "validation_failed",
                    "error_message": "cleaned_data is None",
                }
            
            # DataFrame으로 변환 (직렬화된 경우)
            if isinstance(cleaned_data, dict):
                cleaned_data = pd.DataFrame(cleaned_data)
            
            # KCI 계산
            kci_weekly, kci_monthly = self.calculate_kci(cleaned_data)
            
            # CPI 데이터 확인
            cpi_data = state.get("raw_cpi_data")
            
            if cpi_data is not None:
                if isinstance(cpi_data, dict):
                    cpi_data = pd.DataFrame(cpi_data)
                
                # CPI를 Series로 변환
                if "date" in cpi_data.columns and "value" in cpi_data.columns:
                    cpi_data["date"] = pd.to_datetime(cpi_data["date"])
                    cpi_monthly = cpi_data.set_index("date")["value"]
                else:
                    cpi_monthly = cpi_data.iloc[:, -1]  # 마지막 컬럼
                
                # 정렬
                kci_monthly, cpi_monthly = self.align_with_cpi(kci_monthly, cpi_monthly)
            else:
                cpi_monthly = None
            
            return {
                "kci_weekly": kci_weekly.to_dict(),
                "kci_monthly": kci_monthly.to_dict(),
                "cpi_monthly": cpi_monthly.to_dict() if cpi_monthly is not None else None,
            }
            
        except Exception as e:
            logger.error(f"Index Agent 에러: {e}")
            return {
                "error_type": "validation_failed",
                "error_message": str(e),
            }


def create_mock_chicken_data(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    테스트용 Mock 치킨 가격 데이터 생성
    """
    dates = pd.date_range(start_date, end_date, freq="W-SUN")
    
    data = []
    
    base_prices = {"BBQ": 20000, "교촌": 19000, "BHC": 18000}
    
    for date in dates:
        for brand, base in base_prices.items():
            # 연간 3% 상승 트렌드 + 노이즈
            years_passed = (date - dates[0]).days / 365
            trend = base * (1 + 0.03) ** years_passed
            noise = np.random.normal(0, base * 0.02)
            price = trend + noise
            
            # 간헐적 가격 인상 이벤트
            if np.random.random() < 0.02:
                price += 1000
            
            data.append({
                "date": date,
                "brand": brand,
                "menu": BRAND_CONFIGS[brand].canonical_menu,
                "price": round(price / 100) * 100,  # 100원 단위
            })
    
    return pd.DataFrame(data)
