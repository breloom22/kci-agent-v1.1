"""
Configuration Management for KCI Agent System
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class BrandConfig(BaseModel):
    """브랜드별 설정"""
    name: str
    canonical_menu: str
    weight: float  # KCI 가중치
    store_id: str
    backup_store_ids: list[str] = []
    aliases: list[str] = []


class DataQualityConfig(BaseModel):
    """데이터 품질 설정"""
    # 이상치 탐지
    weekly_change_max: float = 0.20  # 주간 20% 초과 변동
    zscore_max: float = 3.0
    min_price: int = 10000  # 최소 1만원
    max_price: int = 50000  # 최대 5만원
    
    # 결측 허용치
    max_missing_rate: float = 0.10  # 10%
    max_outlier_rate: float = 0.05  # 5%


class SignificanceConfig(BaseModel):
    """통계 검정 설정"""
    adf_p_threshold: float = 0.05
    correlation_p_threshold: float = 0.05
    bootstrap_n: int = 1000
    lag_range: tuple[int, int] = (-3, 3)  # 월 단위
    multiple_testing_method: str = "fdr_bh"  # Benjamini-Hochberg


class BacktestConfig(BaseModel):
    """백테스트 설정"""
    # 거래 비용
    commission: float = 0.00015  # 0.015% 편도
    slippage: float = 0.0005     # 0.05%
    execution_delay: int = 1      # T+1
    
    # 전략 파라미터
    weekly_change_threshold: float = 0.02  # 2%
    ma_period: int = 20
    holding_days: int = 14
    stop_loss: float = -0.05  # -5%
    position_size: float = 0.10  # 10%
    
    # Walk-forward
    in_sample_months: int = 24
    out_sample_months: int = 6
    step_months: int = 6
    
    # 타깃 자산
    target_ticker: str = "396510"  # KODEX 물가채권
    benchmark_ticker: str = "069500"  # KODEX 200


class Settings(BaseSettings):
    """전역 설정"""
    # API Keys
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    ecos_api_key: str = ""
    data_go_kr_key: str = ""

    # KAMIS OpenAPI (aT)
    kamis_cert_key: str = ""
    kamis_cert_id: str = ""
    kamis_itemcategorycode: str = ""
    kamis_itemcode: str = ""
    kamis_kindcode: str = ""
    kamis_productrankcode: str = "04"
    kamis_countrycode: str = ""
    
    # Database
    database_path: str = "./data/kci.db"
    
    # Notifications
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ===== 브랜드 설정 (고정) =====
BRAND_CONFIGS = {
    "BBQ": BrandConfig(
        name="BBQ",
        canonical_menu="황금올리브치킨",
        weight=0.35,
        store_id="",  # .env에서 로드
        aliases=["황올", "황금올리브", "Golden Olive", "황금올리브 후라이드"]
    ),
    "교촌": BrandConfig(
        name="교촌",
        canonical_menu="교촌오리지널",
        weight=0.35,
        store_id="",
        aliases=["오리지날", "Original", "교촌 오리지널", "허니오리지널X"]
    ),
    "BHC": BrandConfig(
        name="BHC",
        canonical_menu="뿌링클",
        weight=0.30,
        store_id="",
        aliases=["뿌링클콤보X", "Puringkle", "뿌링클 치킨"]
    ),
}


# ===== 기본 설정 인스턴스 =====
DATA_QUALITY_CONFIG = DataQualityConfig()
SIGNIFICANCE_CONFIG = SignificanceConfig()
BACKTEST_CONFIG = BacktestConfig()


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()


# ===== 날짜 설정 =====
BASE_DATE = "2020-01-01"  # KCI 기준일
COLLECTION_DAY = 6  # 일요일 (0=월, 6=일)
COLLECTION_HOUR = 21  # 21:00 KST
