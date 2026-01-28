#!/usr/bin/env python3
"""
Idiotology KCI Agent - Main Entry Point

Usage:
    python -m src.main                     # 전체 파이프라인 실행
    python -m src.main --test              # Mock 데이터로 테스트
    python -m src.main --start 2022-01-01  # 특정 기간
"""

import sys
import argparse
from datetime import datetime
from loguru import logger

# 로그 설정
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
)


def run_full_pipeline(start_date: str, end_date: str):
    """전체 파이프라인 실행"""
    from src.graph import run_kci_pipeline
    from src.agents.report_agent import ReportAgent
    
    logger.info("🐔 Idiotology KCI Agent v1.1")
    logger.info(f"기간: {start_date} ~ {end_date}")
    
    # 파이프라인 실행
    result = run_kci_pipeline(
        start_date=start_date,
        end_date=end_date,
    )
    
    # 결과 출력
    if result.get("final_report"):
        agent = ReportAgent()
        text_report = agent.format_text_report(result["final_report"])
        print("\n" + text_report)
        
        # JSON 저장
        json_report = agent.format_json_report(result["final_report"])
        output_path = f"data/processed/kci_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            import os
            os.makedirs("data/processed", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_report)
            logger.info(f"리포트 저장: {output_path}")
        except Exception as e:
            logger.warning(f"리포트 저장 실패: {e}")
    else:
        logger.error("파이프라인 실패")
        logger.error(f"Error Type: {result.get('error_type')}")
        logger.error(f"Error Message: {result.get('error_message')}")
        return 1
    
    return 0


def run_quick_test():
    """빠른 테스트 (Mock 데이터)"""
    import pandas as pd
    import numpy as np
    
    logger.info("🧪 Quick Test Mode")
    
    # 1. Mock 치킨 가격 생성
    from src.agents.index_agent import create_mock_chicken_data, IndexAgent
    
    logger.info("1. Mock 데이터 생성...")
    chicken_data = create_mock_chicken_data("2020-01-01", "2024-12-31")
    logger.info(f"   치킨 가격: {len(chicken_data)}건")
    
    # 2. 데이터 품질 검증
    from src.gates.data_quality import DataQualityGate
    
    logger.info("2. 데이터 품질 검증...")
    dq_gate = DataQualityGate()
    cleaned_data, dq_report = dq_gate.validate(chicken_data)
    logger.info(f"   결과: {dq_report['overall_status']}")
    logger.info(f"   이상치: {dq_report['outlier_count']}건")
    
    # 3. KCI 계산
    logger.info("3. KCI 계산...")
    index_agent = IndexAgent()
    kci_weekly, kci_monthly = index_agent.calculate_kci(cleaned_data)
    logger.info(f"   KCI 범위: {kci_weekly.min():.1f} ~ {kci_weekly.max():.1f}")
    
    # 4. Mock CPI 생성
    from src.tools.apis.ecos import MockEcosClient
    
    logger.info("4. Mock CPI 생성...")
    ecos = MockEcosClient()
    cpi_data = ecos.get_cpi("202001", "202412")
    cpi_monthly = cpi_data.set_index("date")["value"]
    logger.info(f"   CPI 범위: {cpi_monthly.min():.1f} ~ {cpi_monthly.max():.1f}")
    
    # 5. 시계열 정렬
    kci_aligned, cpi_aligned = index_agent.align_with_cpi(kci_monthly, cpi_monthly)
    
    # 6. 유의성 검정
    from src.gates.research_guard import ResearchGuard
    
    logger.info("5. 유의성 검정...")
    guard = ResearchGuard()
    sig_report = guard.validate(kci_aligned, cpi_aligned)
    logger.info(f"   최적 래그: {sig_report['cross_correlation']['best_lag']}개월")
    logger.info(f"   상관계수: {sig_report['cross_correlation']['best_correlation']:.3f}")
    logger.info(f"   p-value: {sig_report['bootstrap_p_value']:.4f}")
    logger.info(f"   통과: {'✓' if sig_report['final_pass'] else '✗'}")
    
    # 7. 백테스트
    from src.agents.backtest_agent import BacktestEngine
    
    logger.info("6. 백테스트...")
    engine = BacktestEngine()
    
    # 가격 데이터 (KCI를 프록시로 사용)
    prices = kci_weekly / kci_weekly.iloc[0] * 10000
    
    bt_report = engine.run_full_backtest(kci_weekly, prices)
    logger.info(f"   총 거래: {bt_report['total_trades']}회")
    logger.info(f"   승률: {bt_report['win_rate']:.1%}")
    logger.info(f"   총 수익률: {bt_report['total_return']:.1%}")
    logger.info(f"   Sharpe: {bt_report['sharpe_ratio']:.2f}")
    logger.info(f"   Max DD: {bt_report['max_drawdown']:.1%}")
    
    # 8. Walk-forward 결과
    wf = bt_report['walk_forward']
    logger.info(f"   Walk-forward Folds: {len(wf['folds'])}")
    logger.info(f"   평균 OOS Sharpe: {wf['avg_oos_sharpe']:.2f}")
    logger.info(f"   안정성 점수: {wf['stability_score']:.1%}")
    
    logger.info("")
    logger.info("✅ Quick Test 완료!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Idiotology KCI Agent v1.1")
    parser.add_argument("--test", action="store_true", help="Run quick test with mock data")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")
    
    if args.test:
        return run_quick_test()
    else:
        return run_full_pipeline(args.start, args.end)


if __name__ == "__main__":
    sys.exit(main())
