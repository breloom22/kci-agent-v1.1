"""
DataQuality Gate (V1.1)

데이터 품질 검증 게이트:
- 스키마 검증
- 이상치 탐지 및 처리
- 결측치 처리
- 표본 변경 감지
- 소스 간 정합성 체크
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

from src.config import DATA_QUALITY_CONFIG, BRAND_CONFIGS, get_settings, COLLECTION_DAY
from src.state import (
    DataQualityReport, 
    SourceConsistency, 
    GateStatus,
)


class DataQualityGate:
    """데이터 품질 검증 게이트"""
    
    # 필수 컬럼
    REQUIRED_COLUMNS = ["date", "brand", "menu", "price"]
    
    def __init__(self, config=None):
        self.config = config or DATA_QUALITY_CONFIG

        # P0 Fix: 실행 간 표본(메뉴) 변경 감지를 위해 스냅샷을 디스크에 저장
        settings = get_settings()
        self._menu_snapshot_path = (
            Path(settings.project_root) / "data" / "processed" / "menu_snapshot.json"
        )
        self._previous_menu_snapshot = self._load_menu_snapshot()

    def _load_menu_snapshot(self) -> dict:
        """이전 실행의 메뉴 스냅샷 로드 (brand -> sorted menus)"""
        try:
            if self._menu_snapshot_path.exists():
                with open(self._menu_snapshot_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload.get("menus_by_brand", {})
        except Exception as e:
            logger.warning(f"menu_snapshot 로드 실패: {e}")
        return {}

    def _save_menu_snapshot(self, menus_by_brand: dict) -> None:
        """메뉴 스냅샷 저장"""
        try:
            self._menu_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now().isoformat(),
                "menus_by_brand": menus_by_brand,
            }
            with open(self._menu_snapshot_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"menu_snapshot 저장 실패: {e}")
    
    def validate(
        self,
        chicken_prices: pd.DataFrame,
        wholesale_data: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        데이터 품질 검증 수행
        
        Args:
            chicken_prices: 치킨 가격 데이터
            wholesale_data: 도매가 데이터 (정합성 체크용)
        
        Returns:
            (정제된 데이터, 품질 리포트)
        """
        logger.info("데이터 품질 검증 시작")
        
        failure_reasons = []
        
        # 1. 스키마 검증
        schema_valid, schema_errors = self._validate_schema(chicken_prices)
        if not schema_valid:
            failure_reasons.extend(schema_errors)
        
        # 2. 결측치 분석
        missing_rate = self._calculate_missing_rate(chicken_prices)
        high_missing = [b for b, r in missing_rate.items() 
                       if r > self.config.max_missing_rate]
        if high_missing:
            failure_reasons.append(f"높은 결측률 브랜드: {high_missing}")
        
        # 3. 이상치 탐지
        outliers = self._detect_outliers(chicken_prices)
        outlier_rate = len(outliers) / len(chicken_prices) if len(chicken_prices) > 0 else 0
        if outlier_rate > self.config.max_outlier_rate:
            failure_reasons.append(f"이상치 비율 {outlier_rate:.1%} > {self.config.max_outlier_rate:.1%}")
        
        # 4. 표본 변경 감지
        sample_change, sample_changes = self._detect_sample_changes(chicken_prices)
        if sample_change:
            failure_reasons.append(f"표본 변경 감지: {len(sample_changes)}건")
        
        # 5. 소스 정합성 (도매가 있는 경우)
        source_consistency = self._check_source_consistency(chicken_prices, wholesale_data)
        if source_consistency["status"] == "ALERT":
            failure_reasons.append(f"소스 정합성 이상: {source_consistency['reason']}")
        
        # 6. 데이터 정제
        cleaned_data = self._clean_data(chicken_prices, outliers)
        
        # 최종 상태 결정
        if not schema_valid:
            overall_status = GateStatus.FAIL
        elif failure_reasons:
            overall_status = GateStatus.WARNING
        else:
            overall_status = GateStatus.PASS
        
        report = DataQualityReport(
            collection_timestamp=datetime.now().isoformat(),
            brands_collected=chicken_prices["brand"].unique().tolist() if "brand" in chicken_prices else [],
            missing_rate=missing_rate,
            outlier_count=len(outliers),
            outliers_detected=outliers,
            schema_valid=schema_valid,
            schema_errors=schema_errors,
            sample_change_detected=sample_change,
            sample_changes=sample_changes,
            source_consistency=source_consistency,
            overall_status=overall_status,
            failure_reasons=failure_reasons,
        )
        
        logger.info(f"데이터 품질 검증 완료: {overall_status.value}")
        
        return cleaned_data, report
    
    def _validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """스키마 검증"""
        errors = []
        
        # 필수 컬럼 확인
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"필수 컬럼 누락: {missing_cols}")
        
        # 데이터 타입 확인
        if "price" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["price"]):
                errors.append("price 컬럼이 숫자형이 아님")
        
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except:
                errors.append("date 컬럼 파싱 실패")
        
        # 브랜드 확인
        if "brand" in df.columns:
            valid_brands = set(BRAND_CONFIGS.keys())
            actual_brands = set(df["brand"].unique())
            unknown = actual_brands - valid_brands
            if unknown:
                errors.append(f"알 수 없는 브랜드: {unknown}")
        
        return len(errors) == 0, errors
    
    def _calculate_missing_rate(self, df: pd.DataFrame) -> dict[str, float]:
        """브랜드별 결측률 계산 ("행 누락"까지 포함)

        - 기존: price NaN 비율만 계산 → row 자체가 누락되면 결측률이 과소추정될 수 있음
        - 개선: 기대 날짜 그리드 대비 (non-null 관측치 / 기대 관측치) 기반 결측률 계산
        """
        if "brand" not in df.columns or "date" not in df.columns:
            return {}

        dfx = df.copy()
        dfx["date"] = pd.to_datetime(dfx["date"]).dt.normalize()

        unique_dates = pd.DatetimeIndex(sorted(dfx["date"].unique()))
        if len(unique_dates) == 0:
            return {b: 1.0 for b in BRAND_CONFIGS.keys()}

        inferred = pd.infer_freq(unique_dates)
        if inferred is None:
            # COLLECTION_DAY: 0=월 ... 6=일
            day_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
            inferred = f"W-{day_map.get(COLLECTION_DAY, 'SUN')}"

        expected_dates = pd.date_range(unique_dates.min(), unique_dates.max(), freq=inferred)
        expected_n = len(expected_dates) if len(expected_dates) > 0 else 1

        result: dict[str, float] = {}
        for brand in BRAND_CONFIGS.keys():
            brand_rows = dfx[dfx["brand"] == brand]
            if len(brand_rows) == 0:
                result[brand] = 1.0
                continue

            # 동일 날짜에 여러 행이 있으면 평균으로 집계
            brand_series = brand_rows.groupby("date")["price"].mean().reindex(expected_dates)
            non_null = int(brand_series.notna().sum())
            result[brand] = 1.0 - (non_null / expected_n)

        return result
    
    def _detect_outliers(self, df: pd.DataFrame) -> list[dict]:
        """이상치 탐지"""
        outliers = []
        
        if "price" not in df.columns or "brand" not in df.columns:
            return outliers
        
        for brand in df["brand"].unique():
            brand_data = df[df["brand"] == brand].copy()
            
            if len(brand_data) < 5:
                continue
            
            # Rule 1: 주간 변동률 체크
            brand_data = brand_data.sort_values("date")
            brand_data["weekly_change"] = brand_data["price"].pct_change()
            
            high_change = brand_data[
                brand_data["weekly_change"].abs() > self.config.weekly_change_max
            ]
            
            for _, row in high_change.iterrows():
                outliers.append({
                    "date": str(row.get("date", "")),
                    "brand": brand,
                    "price": row.get("price"),
                    "reason": f"주간 변동률 {row['weekly_change']:.1%} > {self.config.weekly_change_max:.1%}"
                })
            
            # Rule 2: Z-score 체크
            mean_price = brand_data["price"].mean()
            std_price = brand_data["price"].std()
            
            if std_price > 0:
                brand_data["zscore"] = (brand_data["price"] - mean_price) / std_price
                high_zscore = brand_data[brand_data["zscore"].abs() > self.config.zscore_max]
                
                for _, row in high_zscore.iterrows():
                    if not any(o["date"] == str(row.get("date", "")) and o["brand"] == brand 
                              for o in outliers):
                        outliers.append({
                            "date": str(row.get("date", "")),
                            "brand": brand,
                            "price": row.get("price"),
                            "reason": f"Z-score {row['zscore']:.2f} > {self.config.zscore_max}"
                        })
            
            # Rule 3: 범위 이탈
            out_of_range = brand_data[
                (brand_data["price"] < self.config.min_price) | 
                (brand_data["price"] > self.config.max_price)
            ]
            
            for _, row in out_of_range.iterrows():
                if not any(o["date"] == str(row.get("date", "")) and o["brand"] == brand 
                          for o in outliers):
                    outliers.append({
                        "date": str(row.get("date", "")),
                        "brand": brand,
                        "price": row.get("price"),
                        "reason": f"범위 이탈 ({self.config.min_price}~{self.config.max_price})"
                    })
        
        return outliers
    
    def _detect_sample_changes(self, df: pd.DataFrame) -> tuple[bool, list[dict]]:
        """표본 변경 감지 (메뉴/가게 변경) - 실행 간 비교 가능하도록 영속화"""
        changes: list[dict] = []

        if "brand" not in df.columns or "menu" not in df.columns:
            return False, changes

        prev_snapshot = {k: set(v) for k, v in (self._previous_menu_snapshot or {}).items()}
        current_snapshot: dict[str, list[str]] = {}

        for brand in df["brand"].unique():
            brand_data = df[df["brand"] == brand]
            current_menus = set(brand_data["menu"].dropna().astype(str).unique())
            current_snapshot[brand] = sorted(current_menus)

            prev_menus = prev_snapshot.get(brand, set())
            if prev_menus and current_menus != prev_menus:
                added = sorted(list(current_menus - prev_menus))
                removed = sorted(list(prev_menus - current_menus))
                if added or removed:
                    changes.append({
                        "brand": brand,
                        "added_menus": added,
                        "removed_menus": removed,
                    })

        # 스냅샷 저장 및 상태 갱신
        self._save_menu_snapshot(current_snapshot)
        self._previous_menu_snapshot = current_snapshot

        return len(changes) > 0, changes
    
    def _check_source_consistency(
        self,
        retail_data: pd.DataFrame,
        wholesale_data: Optional[pd.DataFrame]
    ) -> SourceConsistency:
        """소스 간 정합성 체크"""
        
        if wholesale_data is None or len(wholesale_data) == 0:
            return SourceConsistency(
                status="SKIP",
                retail_change=0.0,
                wholesale_change=0.0,
                reason="도매가 데이터 없음"
            )
        
        # 최근 변화율 계산
        if "price" in retail_data.columns and len(retail_data) > 1:
            retail_data_sorted = retail_data.sort_values("date")
            recent_retail = retail_data_sorted.groupby("brand")["price"].last().mean()
            prev_retail = retail_data_sorted.groupby("brand")["price"].nth(-2).mean()
            retail_change = (recent_retail / prev_retail - 1) if prev_retail > 0 else 0
        else:
            retail_change = 0
        
        if "price" in wholesale_data.columns and len(wholesale_data) > 1:
            wholesale_sorted = wholesale_data.sort_values("date")
            wholesale_change = wholesale_sorted["price"].pct_change().iloc[-1]
        else:
            wholesale_change = 0
        
        # 불일치 감지
        # 도매가 5% 상승인데 소매가 3% 하락 = ALERT
        if wholesale_change > 0.05 and retail_change < -0.03:
            return SourceConsistency(
                status="ALERT",
                retail_change=retail_change,
                wholesale_change=wholesale_change,
                reason="도매↑ 소매↓ 불일치"
            )
        
        # 도매가 5% 하락인데 소매가 5% 상승 = ALERT
        if wholesale_change < -0.05 and retail_change > 0.05:
            return SourceConsistency(
                status="ALERT",
                retail_change=retail_change,
                wholesale_change=wholesale_change,
                reason="도매↓ 소매↑ 불일치"
            )
        
        return SourceConsistency(
            status="OK",
            retail_change=retail_change,
            wholesale_change=wholesale_change,
            reason=None
        )
    
    def _clean_data(
        self,
        df: pd.DataFrame,
        outliers: list[dict]
    ) -> pd.DataFrame:
        """데이터 정제 (이상치 보간)"""
        cleaned = df.copy()
        
        # 이상치를 NaN으로 변환
        for outlier in outliers:
            mask = (
                (cleaned["date"].astype(str) == outlier["date"]) & 
                (cleaned["brand"] == outlier["brand"])
            )
            cleaned.loc[mask, "price"] = np.nan
        
        # 브랜드별 선형 보간
        if "brand" in cleaned.columns:
            for brand in cleaned["brand"].unique():
                mask = cleaned["brand"] == brand
                cleaned.loc[mask, "price"] = (
                    cleaned.loc[mask, "price"]
                    .interpolate(method="linear")
                )
        
        # 남은 NaN은 forward fill
        cleaned["price"] = cleaned["price"].ffill().bfill()
        
        return cleaned
    
    def is_passed(self, report: DataQualityReport) -> bool:
        """게이트 통과 여부"""
        return report["overall_status"] in [GateStatus.PASS, GateStatus.WARNING]
