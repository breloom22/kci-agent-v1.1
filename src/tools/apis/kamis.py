"""
KAMIS OpenAPI Client (aT 농수산물유통정보)

- periodRetailProductList: 기간별(일/주/월) 농축수산물 소매가격 조회
- 문서: KAMIS OpenAPI (인증키: cert_key / cert_id)

NOTE
- KAMIS는 응답이 JSON이라도 payload가 문자열로 들어오는 경우가 있어 방어적으로 파싱합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from loguru import logger

from src.config import get_settings


@dataclass
class KAMISParams:
    itemcategorycode: str
    itemcode: str
    kindcode: str
    productrankcode: str = "04"  # 중품(예시). 코드표 확인 필요
    countrycode: str = ""        # 국내/수입 등 (선택)


class KAMISClient:
    """KAMIS OpenAPI 클라이언트"""

    BASE_URL = "https://www.kamis.or.kr/service/price/xml.do"

    def __init__(self, cert_key: str | None = None, cert_id: str | None = None):
        settings = get_settings()
        self.cert_key = cert_key or settings.kamis_cert_key
        self.cert_id = cert_id or settings.kamis_cert_id

        if not self.cert_key or not self.cert_id:
            logger.warning("KAMIS cert_key/cert_id가 설정되지 않았습니다 (.env 확인)")

    def _request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cert_key or not self.cert_id:
            raise ValueError("KAMIS cert_key/cert_id is required")

        q = {
            "action": action,
            "p_cert_key": self.cert_key,
            "p_cert_id": self.cert_id,
            "p_returntype": "json",
            **params,
        }

        resp = requests.get(self.BASE_URL, params=q, timeout=30)
        resp.raise_for_status()

        # 응답이 JSON string일 수도, dict일 수도 있음
        try:
            data = resp.json()
        except Exception:
            data = json.loads(resp.text)

        return data

    def get_period_retail_prices(
        self,
        start_date: str,
        end_date: str,
        kamis_params: KAMISParams,
        convert_to_dataframe: bool = True,
    ) -> pd.DataFrame | Dict[str, Any]:
        """
        기간별 소매가격 조회

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            kamis_params: 품목/품종/등급 코드
        Returns:
            DataFrame columns: [date, price, unit, item_name, kind_name, rank_name, market_name]
        """

        payload = self._request(
            action="periodRetailProductList",
            params={
                "p_startday": start_date,
                "p_endday": end_date,
                "p_itemcategorycode": kamis_params.itemcategorycode,
                "p_itemcode": kamis_params.itemcode,
                "p_kindcode": kamis_params.kindcode,
                "p_productrankcode": kamis_params.productrankcode,
                "p_countrycode": kamis_params.countrycode,
            },
        )

        if not convert_to_dataframe:
            return payload

        # 안전 파싱
        try:
            data = payload.get("data", payload)
            items = data.get("item", []) if isinstance(data, dict) else []
        except Exception:
            items = []

        if not items:
            logger.warning("KAMIS: 결과 item이 비어있습니다. 코드/기간 확인 필요")
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for it in items:
            # KAMIS 필드명은 케이스가 종종 바뀔 수 있어 방어적으로 접근
            rows.append(
                {
                    "date": it.get("day") or it.get("date") or it.get("yyyymmdd"),
                    "price": it.get("dpr1") or it.get("price"),
                    "unit": it.get("unit"),
                    "item_name": it.get("item_name"),
                    "kind_name": it.get("kind_name"),
                    "rank_name": it.get("rank_name"),
                    "market_name": it.get("name") or it.get("market_name"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("-", "", regex=False)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        return df


def get_kamis_client(use_mock: bool = False) -> KAMISClient:
    """팩토리 (현재는 mock 미구현)"""
    if use_mock:
        raise NotImplementedError("KAMIS mock client not implemented")
    return KAMISClient()
