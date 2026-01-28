"""
KCI Agent Graph (V1.1)

LangGraph 기반 멀티에이전트 파이프라인:
Data Agent → DataQuality Gate → Index Agent → ResearchGuard → Backtest Agent → Report Agent
"""

from typing import Literal
from loguru import logger

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    logger.warning("langgraph not installed. Run: pip install langgraph")
    StateGraph = None
    END = "END"

from src.state import KCIState, ErrorType, GateStatus, create_initial_state
from src.gates.data_quality import DataQualityGate
from src.gates.research_guard import ResearchGuard
from src.agents.index_agent import IndexAgent
from src.agents.backtest_agent import BacktestAgent
from src.agents.report_agent import ReportAgent


# ===== Node Functions =====

def data_collection_node(state: KCIState) -> dict:
    """
    데이터 수집 노드 (Data Agent)
    
    실제 구현에서는:
    - 배달앱 크롤링
    - ECOS API 호출
    - aT API 호출
    """
    logger.info("=== Data Collection Node ===")
    
    from datetime import datetime
    import pandas as pd
    import numpy as np
    
    # Mock 데이터 생성 (실제로는 크롤링/API 호출)
    from src.agents.index_agent import create_mock_chicken_data
    from src.tools.apis.ecos import MockEcosClient
    
    start_date, end_date = state["date_range"]
    
    # 치킨 가격 (Mock)
    chicken_prices = create_mock_chicken_data(start_date, end_date)
    
    # CPI (Mock)
    ecos_client = MockEcosClient()
    cpi_data = ecos_client.get_cpi(
        start_date=start_date.replace("-", "")[:6],
        end_date=end_date.replace("-", "")[:6],
    )
    
    # 도매가 (Mock) - 간단히 None
    wholesale_data = None
    
    logger.info(f"데이터 수집 완료: 치킨 {len(chicken_prices)}건, CPI {len(cpi_data)}건")
    
    return {
        "raw_chicken_prices": chicken_prices.to_dict(orient="records"),
        "raw_cpi_data": cpi_data.to_dict(orient="records"),
        "raw_wholesale_data": wholesale_data,
        "collection_timestamp": datetime.now().isoformat(),
    }


def data_quality_node(state: KCIState) -> dict:
    """데이터 품질 검증 노드 (DataQuality Gate)"""
    logger.info("=== Data Quality Gate ===")
    
    import pandas as pd
    
    # 데이터 복원
    raw_chicken = state.get("raw_chicken_prices")
    if raw_chicken is None:
        return {
            "data_quality_pass": False,
            "error_type": ErrorType.DATA_MISSING,
            "error_message": "raw_chicken_prices is None",
        }
    
    chicken_df = pd.DataFrame(raw_chicken)
    
    # 도매가 (옵션)
    raw_wholesale = state.get("raw_wholesale_data")
    wholesale_df = pd.DataFrame(raw_wholesale) if raw_wholesale else None
    
    # 품질 검증
    gate = DataQualityGate()
    cleaned_data, report = gate.validate(chicken_df, wholesale_df)
    
    passed = gate.is_passed(report)
    
    logger.info(f"Data Quality Gate: {'PASS' if passed else 'FAIL'}")
    
    return {
        "cleaned_data": cleaned_data.to_dict(orient="records"),
        "data_quality_report": report,
        "data_quality_pass": passed,
    }


def index_calculation_node(state: KCIState) -> dict:
    """지수 계산 노드 (Index Agent)"""
    logger.info("=== Index Calculation Node ===")
    
    agent = IndexAgent()
    return agent.run(state)


def research_guard_node(state: KCIState) -> dict:
    """통계 유의성 검증 노드 (ResearchGuard Gate)"""
    logger.info("=== Research Guard Gate ===")
    
    import pandas as pd
    
    # KCI, CPI 데이터
    kci_data = state.get("kci_monthly")
    cpi_data = state.get("cpi_monthly")
    
    if kci_data is None or cpi_data is None:
        return {
            "research_guard_pass": False,
            "error_type": ErrorType.VALIDATION_FAILED,
            "error_message": "KCI or CPI data is None",
        }
    
    # dict → Series
    kci = pd.Series(kci_data)
    kci.index = pd.to_datetime(kci.index)
    
    cpi = pd.Series(cpi_data)
    cpi.index = pd.to_datetime(cpi.index)
    
    # 유의성 검증
    guard = ResearchGuard()
    report = guard.validate(kci, cpi)
    
    passed = guard.is_passed(report)
    
    logger.info(f"Research Guard: {'PASS' if passed else 'FAIL'}")
    
    return {
        "significance_report": report,
        "research_guard_pass": passed,
    }


def backtest_node(state: KCIState) -> dict:
    """백테스트 노드 (Backtest Agent)"""
    logger.info("=== Backtest Node ===")
    
    agent = BacktestAgent()
    return agent.run(state)


def report_node(state: KCIState) -> dict:
    """리포트 생성 노드 (Report Agent)"""
    logger.info("=== Report Node ===")
    
    agent = ReportAgent()
    return agent.run(state)


def error_handler_node(state: KCIState) -> dict:
    """에러 핸들링 노드"""
    logger.warning(f"=== Error Handler: {state.get('error_type')} ===")
    logger.warning(f"Message: {state.get('error_message')}")
    
    # 재시도 로직 (간단 버전)
    retry_count = state.get("retry_count", 0)
    
    if retry_count < 3:
        logger.info(f"Retry attempt {retry_count + 1}")
        return {"retry_count": retry_count + 1}
    
    return {"error_message": "Max retries exceeded"}


# ===== Router Functions =====

def route_after_data_quality(state: KCIState) -> Literal["index_agent", "error_handler"]:
    """데이터 품질 게이트 후 라우팅"""
    if state.get("data_quality_pass", False):
        return "index_agent"
    return "error_handler"


def route_after_research_guard(state: KCIState) -> Literal["backtest_agent", "report_agent"]:
    """리서치 가드 후 라우팅"""
    if state.get("research_guard_pass", False):
        return "backtest_agent"
    # 유의성 실패 시 백테스트 스킵하고 리포트로
    logger.warning("Research Guard failed - skipping backtest")
    return "report_agent"


# ===== Graph Builder =====

def build_kci_graph():
    """
    KCI 에이전트 그래프 빌드
    
    Flow:
    data_agent → data_quality_gate → index_agent → research_guard → backtest_agent → report_agent
                        ↓                                 ↓
                  error_handler                    report_agent (skip backtest)
    """
    if StateGraph is None:
        raise ImportError("langgraph is required. Install with: pip install langgraph")
    
    # 그래프 생성
    workflow = StateGraph(KCIState)
    
    # 노드 추가
    workflow.add_node("data_agent", data_collection_node)
    workflow.add_node("data_quality_gate", data_quality_node)
    workflow.add_node("index_agent", index_calculation_node)
    workflow.add_node("research_guard", research_guard_node)
    workflow.add_node("backtest_agent", backtest_node)
    workflow.add_node("report_agent", report_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # 엣지 정의
    workflow.add_edge("data_agent", "data_quality_gate")
    
    # 조건부 라우팅: Data Quality Gate
    workflow.add_conditional_edges(
        "data_quality_gate",
        route_after_data_quality,
        {
            "index_agent": "index_agent",
            "error_handler": "error_handler",
        }
    )
    
    workflow.add_edge("index_agent", "research_guard")
    
    # 조건부 라우팅: Research Guard
    workflow.add_conditional_edges(
        "research_guard",
        route_after_research_guard,
        {
            "backtest_agent": "backtest_agent",
            "report_agent": "report_agent",
        }
    )
    
    workflow.add_edge("backtest_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    workflow.add_edge("error_handler", END)
    
    # 시작점
    workflow.set_entry_point("data_agent")
    
    return workflow.compile()


def run_kci_pipeline(
    start_date: str = "2020-01-01",
    end_date: str = None,
    brands: list[str] = None,
) -> dict:
    """
    KCI 파이프라인 실행
    
    Args:
        start_date: 시작일
        end_date: 종료일 (None이면 오늘)
        brands: 타겟 브랜드 목록
    
    Returns:
        최종 상태 dict
    """
    from datetime import datetime
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 초기 상태
    initial_state = create_initial_state(
        start_date=start_date,
        end_date=end_date,
        brands=brands,
    )
    
    # 그래프 빌드 & 실행
    logger.info("=" * 60)
    logger.info("KCI Agent Pipeline 시작")
    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info("=" * 60)
    
    graph = build_kci_graph()
    final_state = graph.invoke(initial_state)
    
    logger.info("=" * 60)
    logger.info("KCI Agent Pipeline 완료")
    logger.info("=" * 60)
    
    return final_state


# ===== CLI Entry Point =====

if __name__ == "__main__":
    import sys
    
    # 로그 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    # 파이프라인 실행
    result = run_kci_pipeline(
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    
    # 결과 출력
    if result.get("final_report"):
        agent = ReportAgent()
        text_report = agent.format_text_report(result["final_report"])
        print("\n" + text_report)
    else:
        print("리포트 생성 실패")
        print(f"Error: {result.get('error_message')}")
