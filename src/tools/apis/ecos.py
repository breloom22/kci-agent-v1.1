"""
한국은행 ECOS API Client

소비자물가지수(CPI) 및 관련 경제 통계 조회
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger

from src.config import get_settings


class EcosClient:
    """한국은행 ECOS API 클라이언트"""
    
    BASE_URL = "https://ecos.bok.or.kr/api"
    
    # 주요 통계표 코드
    STAT_CODES = {
        "cpi": "901Y009",           # 소비자물가지수 (2020=100)
        "cpi_food": "901Y009",      # 식료품 물가지수
        "cpi_dining_out": "901Y009",# 외식 물가지수
        "ppi": "901Y010",           # 생산자물가지수
    }
    
    # 품목 코드
    ITEM_CODES = {
        "cpi_total": "0",           # 총지수
        "cpi_food": "01",           # 식료품및비주류음료
        "cpi_dining_out": "0701",   # 외식
        "cpi_chicken": "070117",    # 치킨 (외식-치킨)
    }
    
    def __init__(self, api_key: str = None):
        settings = get_settings()
        self.api_key = api_key or settings.ecos_api_key
        
        if not self.api_key:
            logger.warning("ECOS API 키가 설정되지 않았습니다")
    
    def _request(
        self,
        service: str,
        stat_code: str,
        cycle: str,  # A:연간, Q:분기, M:월간
        start_date: str,
        end_date: str,
        item_code1: str = "?",
        item_code2: str = "?",
        item_code3: str = "?",
    ) -> dict:
        """API 요청"""
        
        url = (
            f"{self.BASE_URL}/{service}/{self.api_key}/json/kr/"
            f"1/1000/{stat_code}/{cycle}/{start_date}/{end_date}/"
            f"{item_code1}/{item_code2}/{item_code3}"
        )
        
        logger.debug(f"ECOS API 요청: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "RESULT" in data and data["RESULT"]["CODE"] != "INFO-000":
            error_msg = data["RESULT"]["MESSAGE"]
            logger.error(f"ECOS API 에러: {error_msg}")
            raise ValueError(f"ECOS API Error: {error_msg}")
        
        return data
    
    def get_cpi(
        self,
        start_date: str = "202001",
        end_date: str = None,
        item: str = "total"
    ) -> pd.DataFrame:
        """
        소비자물가지수 조회
        
        Args:
            start_date: 시작일 (YYYYMM 형식)
            end_date: 종료일 (YYYYMM 형식)
            item: "total", "food", "dining_out", "chicken"
        
        Returns:
            DataFrame with columns: [date, value, yoy_change]
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m")
        
        item_code = self.ITEM_CODES.get(f"cpi_{item}", "0")
        
        try:
            data = self._request(
                service="StatisticSearch",
                stat_code=self.STAT_CODES["cpi"],
                cycle="M",
                start_date=start_date,
                end_date=end_date,
                item_code1=item_code,
            )
            
            if "StatisticSearch" not in data:
                logger.warning("CPI 데이터 없음")
                return pd.DataFrame()
            
            rows = data["StatisticSearch"]["row"]
            
            df = pd.DataFrame(rows)
            df = df[["TIME", "DATA_VALUE"]].rename(columns={
                "TIME": "date",
                "DATA_VALUE": "value"
            })
            
            df["date"] = pd.to_datetime(df["date"], format="%Y%m")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            df = df.sort_values("date").reset_index(drop=True)
            
            # YoY 변화율 계산
            df["yoy_change"] = df["value"].pct_change(12) * 100
            
            logger.info(f"CPI 데이터 조회 완료: {len(df)}건 ({item})")
            
            return df
            
        except Exception as e:
            logger.error(f"CPI 조회 실패: {e}")
            raise
    
    def get_key_statistics(self) -> pd.DataFrame:
        """100대 주요 통계지표 조회"""
        
        url = f"{self.BASE_URL}/KeyStatisticList/{self.api_key}/json/kr/1/100"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "KeyStatisticList" not in data:
            return pd.DataFrame()
        
        rows = data["KeyStatisticList"]["row"]
        df = pd.DataFrame(rows)
        
        return df[["KEYSTAT_NAME", "DATA_VALUE", "UNIT_NAME", "CYCLE"]]


class MockEcosClient:
    """
    테스트/개발용 Mock 클라이언트
    실제 API 키 없이 샘플 데이터 반환
    """
    
    def get_cpi(
        self,
        start_date: str = "202001",
        end_date: str = None,
        item: str = "total"
    ) -> pd.DataFrame:
        """샘플 CPI 데이터 생성"""
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m")
        
        start = pd.to_datetime(start_date, format="%Y%m")
        end = pd.to_datetime(end_date, format="%Y%m")
        
        dates = pd.date_range(start, end, freq='ME')
        
        # 샘플 CPI (2020=100 기준, 연 3% 상승 트렌드 + 노이즈)
        n = len(dates)
        base = 100
        trend = np.linspace(0, n * 0.25, n)  # 월 0.25% 상승
        noise = np.random.normal(0, 0.5, n)
        values = base + trend + noise
        
        df = pd.DataFrame({
            "date": dates,
            "value": values
        })
        
        df["yoy_change"] = df["value"].pct_change(12) * 100
        
        logger.info(f"[Mock] CPI 샘플 데이터 생성: {len(df)}건")
        
        return df


def get_ecos_client(use_mock: bool = False) -> EcosClient:
    """ECOS 클라이언트 팩토리"""
    if use_mock:
        return MockEcosClient()
    return EcosClient()


# NumPy import for Mock
import numpy as np
