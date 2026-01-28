"""
Report Agent (V1.1)

리포트 생성:
1. 신호 근거
2. 데이터 품질
3. 실패 케이스
4. 벤치마크 대비
5. 불확실성 & 면책
"""

import json
from datetime import datetime
from typing import Optional
from loguru import logger

from src.state import (
    KCIState,
    FinalReport,
    SignalType,
    DataQualityReport,
    SignificanceReport,
    BacktestReport,
    GateStatus,
)


class ReportAgent:
    """리포트 생성 에이전트"""
    
    VERSION = "1.1.0"
    
    DISCLAIMER = (
        "⚠️ 본 리포트는 교육/정보 목적이며 투자 조언이 아닙니다. "
        "과거 성과가 미래 수익을 보장하지 않습니다. "
        "투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
    )
    
    def generate_report(
        self,
        kci_current: float,
        kci_change_weekly: float,
        kci_change_monthly: float,
        current_signal: SignalType,
        data_quality: Optional[DataQualityReport],
        significance: Optional[SignificanceReport],
        backtest: Optional[BacktestReport],
    ) -> FinalReport:
        """최종 리포트 생성"""
        
        logger.info("리포트 생성 시작")
        
        # 1. 신호 근거
        signal_rationale = self._build_signal_rationale(
            current_signal, kci_change_weekly, kci_change_monthly, significance
        )
        
        # 2. 데이터 품질 요약
        data_quality_summary = self._build_data_quality_summary(data_quality)
        
        # 3. 실패 케이스
        failed_cases_summary = self._build_failed_cases_summary(backtest)
        
        # 4. 벤치마크 대비
        benchmark_summary = self._build_benchmark_summary(backtest)
        
        # 5. 불확실성
        uncertainty = self._build_uncertainty(backtest, significance)
        
        # 신뢰도 계산
        signal_confidence = self._calculate_confidence(
            data_quality, significance, backtest
        )
        
        report = FinalReport(
            generated_at=datetime.now().isoformat(),
            version=self.VERSION,
            current_signal=current_signal,
            signal_confidence=signal_confidence,
            current_kci=kci_current,
            kci_change_weekly=kci_change_weekly,
            kci_change_monthly=kci_change_monthly,
            signal_rationale=signal_rationale,
            data_quality_summary=data_quality_summary,
            failed_cases_summary=failed_cases_summary,
            benchmark_summary=benchmark_summary,
            uncertainty=uncertainty,
            disclaimer=self.DISCLAIMER,
        )
        
        logger.info(f"리포트 생성 완료: 신호={current_signal.value}, 신뢰도={signal_confidence:.1%}")
        
        return report
    
    def _build_signal_rationale(
        self,
        signal: SignalType,
        weekly_change: float,
        monthly_change: float,
        significance: Optional[SignificanceReport],
    ) -> dict:
        """신호 근거 구성"""
        
        rationale = {
            "signal": signal.value,
            "triggers": [],
            "statistics": {},
        }
        
        # 트리거 조건
        if weekly_change > 0.02:
            rationale["triggers"].append({
                "condition": "주간 변화율 > 2%",
                "value": f"{weekly_change:.1%}",
                "passed": True,
            })
        
        if monthly_change > 0.03:
            rationale["triggers"].append({
                "condition": "월간 변화율 > 3%",
                "value": f"{monthly_change:.1%}",
                "passed": True,
            })
        
        # 통계 검정 결과
        if significance:
            rationale["statistics"] = {
                "best_lag": f"{significance['cross_correlation']['best_lag']}개월",
                "correlation": f"{significance['cross_correlation']['best_correlation']:.3f}",
                "bootstrap_p_value": f"{significance['bootstrap_p_value']:.4f}",
                "significance_pass": significance["final_pass"],
            }
            
            if significance["final_pass"]:
                rationale["triggers"].append({
                    "condition": "통계적 유의성",
                    "value": f"p={significance['bootstrap_p_value']:.4f}",
                    "passed": True,
                })
        
        return rationale
    
    def _build_data_quality_summary(
        self,
        data_quality: Optional[DataQualityReport],
    ) -> dict:
        """데이터 품질 요약"""
        
        if data_quality is None:
            return {"status": "UNKNOWN", "details": "데이터 품질 정보 없음"}
        
        return {
            "status": data_quality["overall_status"].value if isinstance(data_quality["overall_status"], GateStatus) else data_quality["overall_status"],
            "collection_time": data_quality["collection_timestamp"],
            "brands_collected": data_quality["brands_collected"],
            "missing_rate": data_quality["missing_rate"],
            "outlier_count": data_quality["outlier_count"],
            "source_consistency": data_quality["source_consistency"]["status"],
            "issues": data_quality["failure_reasons"],
        }
    
    def _build_failed_cases_summary(
        self,
        backtest: Optional[BacktestReport],
    ) -> list[dict]:
        """실패 케이스 요약"""
        
        if backtest is None:
            return []
        
        failed_cases = backtest.get("failed_cases", [])
        
        return [
            {
                "date": case["date"],
                "loss": f"{case['loss_pct']:.1%}",
                "context": case["market_context"],
            }
            for case in failed_cases[:5]  # 최대 5개
        ]
    
    def _build_benchmark_summary(
        self,
        backtest: Optional[BacktestReport],
    ) -> dict:
        """벤치마크 대비 요약"""
        
        if backtest is None:
            return {"status": "NO_DATA"}
        
        comparison = backtest.get("benchmark_comparison", {})
        
        return {
            "strategy_cagr": f"{comparison.get('strategy_cagr', 0):.1%}",
            "benchmark_cagr": f"{comparison.get('benchmark_cagr', 0):.1%}",
            "excess_return": f"{comparison.get('excess_return', 0):.1%}",
            "strategy_sharpe": f"{comparison.get('strategy_sharpe', 0):.2f}",
            "benchmark_sharpe": f"{comparison.get('benchmark_sharpe', 0):.2f}",
            "strategy_max_dd": f"{comparison.get('strategy_max_dd', 0):.1%}",
            "win_rate": f"{backtest.get('win_rate', 0):.1%}",
            "total_trades": backtest.get("total_trades", 0),
        }
    
    def _build_uncertainty(
        self,
        backtest: Optional[BacktestReport],
        significance: Optional[SignificanceReport],
    ) -> dict:
        """불확실성 정보"""
        
        uncertainty = {
            "return_ci_95": None,
            "walk_forward_stability": None,
            "statistical_power": None,
        }
        
        if backtest:
            ci = backtest.get("return_ci_95", (0, 0))
            uncertainty["return_ci_95"] = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            
            wf = backtest.get("walk_forward", {})
            uncertainty["walk_forward_stability"] = f"{wf.get('stability_score', 0):.1%}"
            uncertainty["oos_sharpe"] = f"{wf.get('avg_oos_sharpe', 0):.2f}"
            uncertainty["degradation"] = f"{wf.get('avg_degradation', 0):.2f}"
        
        if significance:
            uncertainty["statistical_power"] = "HIGH" if significance["final_pass"] else "LOW"
        
        return uncertainty
    
    def _calculate_confidence(
        self,
        data_quality: Optional[DataQualityReport],
        significance: Optional[SignificanceReport],
        backtest: Optional[BacktestReport],
    ) -> float:
        """신호 신뢰도 계산 (0~1)"""
        
        score = 0.5  # 기본값
        
        # 데이터 품질 (30%)
        if data_quality:
            status = data_quality["overall_status"]
            if isinstance(status, GateStatus):
                if status == GateStatus.PASS:
                    score += 0.3
                elif status == GateStatus.WARNING:
                    score += 0.15
        
        # 통계 유의성 (40%)
        if significance and significance["final_pass"]:
            p_value = significance["bootstrap_p_value"]
            if p_value < 0.01:
                score += 0.4
            elif p_value < 0.05:
                score += 0.3
        
        # 백테스트 성과 (30%)
        if backtest:
            win_rate = backtest.get("win_rate", 0)
            sharpe = backtest.get("sharpe_ratio", 0)
            
            if win_rate > 0.6 and sharpe > 1.0:
                score += 0.3
            elif win_rate > 0.5 and sharpe > 0.5:
                score += 0.15
        
        return min(score, 1.0)
    
    def format_text_report(self, report: FinalReport) -> str:
        """텍스트 형식 리포트"""
        
        lines = [
            "🐔 KCI (Korean Chicken Index) 리포트 v1.1",
            "━" * 50,
            "",
            f"📅 생성일시: {report['generated_at']}",
            f"📊 현재 KCI: {report['current_kci']:.1f} (주간 {report['kci_change_weekly']:+.1%}, 월간 {report['kci_change_monthly']:+.1%})",
            "",
            "═" * 50,
            "🔍 [1] 신호 근거",
            "═" * 50,
        ]
        
        signal_icon = "🟢" if report["current_signal"] == SignalType.LONG else "⚪"
        lines.append(f"• 신호: {signal_icon} {report['current_signal'].value}")
        lines.append(f"• 신뢰도: {report['signal_confidence']:.0%}")
        
        rationale = report["signal_rationale"]
        if rationale.get("triggers"):
            lines.append("• 트리거:")
            for t in rationale["triggers"]:
                icon = "✓" if t["passed"] else "✗"
                lines.append(f"  - {t['condition']}: {t['value']} {icon}")
        
        if rationale.get("statistics"):
            stats = rationale["statistics"]
            lines.extend([
                "• 통계 검정:",
                f"  - 리드-래그: KCI → CPI {stats.get('best_lag', 'N/A')} 선행",
                f"  - 상관계수: {stats.get('correlation', 'N/A')}",
                f"  - p-value: {stats.get('bootstrap_p_value', 'N/A')}",
            ])
        
        lines.extend([
            "",
            "═" * 50,
            "📋 [2] 데이터 품질",
            "═" * 50,
        ])
        
        dq = report["data_quality_summary"]
        lines.extend([
            f"• 상태: {dq.get('status', 'N/A')}",
            f"• 수집 일시: {dq.get('collection_time', 'N/A')}",
            f"• 브랜드: {', '.join(dq.get('brands_collected', []))}",
            f"• 이상치: {dq.get('outlier_count', 0)}건",
        ])
        
        if dq.get("issues"):
            lines.append("• 이슈: " + "; ".join(dq["issues"]))
        
        lines.extend([
            "",
            "═" * 50,
            "⚠️ [3] 실패 케이스",
            "═" * 50,
        ])
        
        failed = report["failed_cases_summary"]
        if failed:
            for case in failed:
                lines.append(f"• {case['date']}: {case['loss']} ({case['context']})")
        else:
            lines.append("• 기록된 실패 케이스 없음")
        
        lines.extend([
            "",
            "═" * 50,
            "📈 [4] 벤치마크 대비",
            "═" * 50,
        ])
        
        bm = report["benchmark_summary"]
        lines.extend([
            f"• 전략 CAGR: {bm.get('strategy_cagr', 'N/A')}",
            f"• 벤치마크 CAGR: {bm.get('benchmark_cagr', 'N/A')}",
            f"• 초과수익: {bm.get('excess_return', 'N/A')}",
            f"• Sharpe: {bm.get('strategy_sharpe', 'N/A')} (벤치마크: {bm.get('benchmark_sharpe', 'N/A')})",
            f"• 승률: {bm.get('win_rate', 'N/A')} ({bm.get('total_trades', 0)}거래)",
        ])
        
        lines.extend([
            "",
            "═" * 50,
            "📊 [5] 불확실성 & 면책",
            "═" * 50,
        ])
        
        unc = report["uncertainty"]
        lines.extend([
            f"• 수익률 95% 신뢰구간: {unc.get('return_ci_95', 'N/A')}",
            f"• Walk-forward 안정성: {unc.get('walk_forward_stability', 'N/A')}",
            f"• OOS Sharpe: {unc.get('oos_sharpe', 'N/A')}",
            "",
            report["disclaimer"],
            "━" * 50,
        ])
        
        return "\n".join(lines)
    
    def format_json_report(self, report: FinalReport) -> str:
        """JSON 형식 리포트"""
        
        # SignalType enum을 문자열로 변환
        report_dict = dict(report)
        if isinstance(report_dict.get("current_signal"), SignalType):
            report_dict["current_signal"] = report_dict["current_signal"].value
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)
    
    def run(self, state: dict) -> dict:
        """LangGraph 노드 실행"""
        try:
            # KCI 현황
            kci_monthly = state.get("kci_monthly", {})
            if isinstance(kci_monthly, dict) and kci_monthly:
                values = list(kci_monthly.values())
                kci_current = values[-1] if values else 100.0
                kci_weekly = (values[-1] / values[-2] - 1) if len(values) > 1 else 0.0
                kci_monthly_change = (values[-1] / values[-5] - 1) if len(values) > 4 else 0.0
            else:
                kci_current, kci_weekly, kci_monthly_change = 100.0, 0.0, 0.0
            
            # 현재 신호
            signal_str = state.get("current_signal", "FLAT")
            current_signal = SignalType(signal_str) if isinstance(signal_str, str) else signal_str
            
            # 리포트 생성
            report = self.generate_report(
                kci_current=kci_current,
                kci_change_weekly=kci_weekly,
                kci_change_monthly=kci_monthly_change,
                current_signal=current_signal,
                data_quality=state.get("data_quality_report"),
                significance=state.get("significance_report"),
                backtest=state.get("backtest_report"),
            )
            
            return {"final_report": report}
            
        except Exception as e:
            logger.error(f"Report Agent 에러: {e}")
            return {"error_message": str(e)}
