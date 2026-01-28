"""
Backtest Agent (V1.1)

- 거래비용/슬리피지 포함
- Look-ahead bias 방지
- Walk-forward 검증
- 실패 케이스 기록
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from src.config import BACKTEST_CONFIG
from src.state import (
    BacktestReport,
    TradeRecord,
    FailedCase,
    WalkForwardResult,
    WalkForwardFold,
    BenchmarkComparison,
)
from src.utils.math import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_win_rate,
)


class BacktestEngine:
    """
    백테스트 엔진 (V1.1)
    
    핵심 원칙:
    1. 미래 정보 누수 방지 (t 시점에서 t-1까지 데이터만 사용)
    2. T+1 체결 (신호 다음날 시가)
    3. 거래비용 반영
    """
    
    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG
    
    def generate_signals(
        self,
        kci: pd.Series,
    ) -> pd.Series:
        """
        신호 생성 (Look-ahead bias 방지)
        
        Entry 조건:
        - 주간 변화율 > threshold
        - 20MA 상향 돌파
        """
        signals = pd.Series(0, index=kci.index)
        ma_period = self.config.ma_period
        
        if len(kci) < ma_period + 5:
            logger.warning(f"데이터 부족: {len(kci)} < {ma_period + 5}")
            return signals
        
        for t in range(ma_period + 1, len(kci)):
            # t 시점에서는 t-1까지의 데이터만 사용 가능
            available = kci.iloc[:t]
            
            if len(available) < 5:
                continue
            
            # 조건 1: 변화율
            recent_change = available.iloc[-1] / available.iloc[-5] - 1
            
            # 조건 2: MA 상향 돌파
            current_ma = available.iloc[-ma_period:].mean()
            prev_ma = available.iloc[-(ma_period+1):-1].mean()
            
            price_above_ma = available.iloc[-1] > current_ma
            was_below_ma = available.iloc[-2] <= prev_ma
            
            if (recent_change > self.config.weekly_change_threshold and 
                price_above_ma and was_below_ma):
                signals.iloc[t] = 1
        
        logger.info(f"신호 생성 완료: {(signals == 1).sum()}개 진입 신호")
        return signals
    
    def execute_backtest(
        self,
        kci: pd.Series,
        signals: pd.Series,
        prices: pd.Series,
    ) -> dict:
        """백테스트 실행"""
        trades = []
        position = 0
        entry_price = 0.0
        entry_date = None
        daily_returns = []
        
        for t in range(1, len(signals)):
            signal = signals.iloc[t-1]
            today = signals.index[t]
            today_price = prices.iloc[t] if t < len(prices) else prices.iloc[-1]
            
            # 진입
            if signal == 1 and position == 0:
                entry_price = today_price * (1 + self.config.slippage) * (1 + self.config.commission)
                entry_date = today
                position = 1
                logger.debug(f"진입: {today} @ {entry_price:.0f}")
            
            # 포지션 보유 중
            if position == 1:
                holding_days = (today - entry_date).days
                current_return = today_price / entry_price - 1
                daily_returns.append(current_return / max(holding_days, 1))
                
                exit_reason = None
                if holding_days >= self.config.holding_days:
                    exit_reason = "time_limit"
                if current_return <= self.config.stop_loss:
                    exit_reason = "stop_loss"
                
                if exit_reason:
                    exit_price = today_price * (1 - self.config.slippage) * (1 - self.config.commission)
                    trade_return = exit_price / entry_price - 1
                    
                    trades.append(TradeRecord(
                        entry_date=entry_date.strftime("%Y-%m-%d"),
                        exit_date=today.strftime("%Y-%m-%d"),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=trade_return,
                        holding_days=holding_days,
                        exit_reason=exit_reason,
                    ))
                    
                    logger.debug(f"청산: {today} ({exit_reason}), 수익: {trade_return:.2%}")
                    position = 0
        
        return {"trades": trades, "daily_returns": daily_returns}
    
    def calculate_metrics(
        self,
        trades: list[TradeRecord],
        daily_returns: list[float],
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """성과 지표 계산"""
        if not trades:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0.0, "total_return": 0.0, "cagr": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0,
                "avg_return": 0.0, "avg_winning_return": 0.0, "avg_losing_return": 0.0,
                "avg_holding_days": 0.0,
            }
        
        returns = [t["return_pct"] for t in trades]
        returns_series = pd.Series(returns)
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]
        
        cumulative = (1 + returns_series).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        years = (end_date - start_date).days / 365
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades) if trades else 0,
            "total_return": total_return,
            "cagr": calculate_cagr(total_return, years),
            "sharpe_ratio": calculate_sharpe_ratio(returns_series, periods_per_year=12),
            "sortino_ratio": calculate_sortino_ratio(returns_series, periods_per_year=12),
            "max_drawdown": calculate_max_drawdown(cumulative),
            "avg_return": np.mean(returns) if returns else 0,
            "avg_winning_return": np.mean(winning) if winning else 0,
            "avg_losing_return": np.mean(losing) if losing else 0,
            "avg_holding_days": np.mean([t["holding_days"] for t in trades]) if trades else 0,
        }
    
    def identify_failed_cases(self, trades: list[TradeRecord]) -> list[FailedCase]:
        """실패 케이스 식별"""
        return [
            FailedCase(
                date=t["entry_date"],
                loss_pct=t["return_pct"],
                market_context=f"청산 사유: {t['exit_reason']}",
            )
            for t in trades if t["return_pct"] < -0.02
        ]
    
    def walk_forward_validation(
        self,
        kci: pd.Series,
        prices: pd.Series,
    ) -> WalkForwardResult:
        """Walk-forward 검증"""
        folds = []
        total_months = len(kci)
        in_months = self.config.in_sample_months
        out_months = self.config.out_sample_months
        step = self.config.step_months
        
        current_start = 0
        fold_num = 1
        
        while current_start + in_months + out_months <= total_months:
            train_end = current_start + in_months
            test_end = train_end + out_months
            
            # In-sample
            kci_train = kci.iloc[current_start:train_end]
            prices_train = prices.iloc[current_start:train_end]
            signals_train = self.generate_signals(kci_train)
            train_result = self.execute_backtest(kci_train, signals_train, prices_train)
            train_metrics = self.calculate_metrics(
                train_result["trades"], train_result["daily_returns"],
                kci_train.index[0], kci_train.index[-1]
            )
            
            # Out-of-sample
            kci_test = kci.iloc[train_end:test_end]
            prices_test = prices.iloc[train_end:test_end]
            signals_test = self.generate_signals(kci.iloc[:test_end]).iloc[train_end:test_end]
            test_result = self.execute_backtest(kci_test, signals_test, prices_test)
            test_metrics = self.calculate_metrics(
                test_result["trades"], test_result["daily_returns"],
                kci_test.index[0] if len(kci_test) > 0 else kci_train.index[-1],
                kci_test.index[-1] if len(kci_test) > 0 else kci_train.index[-1]
            )
            
            folds.append(WalkForwardFold(
                fold=fold_num,
                train_start=kci_train.index[0].strftime("%Y-%m-%d"),
                train_end=kci_train.index[-1].strftime("%Y-%m-%d"),
                test_start=kci_test.index[0].strftime("%Y-%m-%d") if len(kci_test) > 0 else "",
                test_end=kci_test.index[-1].strftime("%Y-%m-%d") if len(kci_test) > 0 else "",
                train_sharpe=train_metrics["sharpe_ratio"],
                test_sharpe=test_metrics["sharpe_ratio"],
                test_return=test_metrics["total_return"],
                degradation=train_metrics["sharpe_ratio"] - test_metrics["sharpe_ratio"],
            ))
            
            current_start += step
            fold_num += 1
        
        if folds:
            avg_oos_sharpe = np.mean([f["test_sharpe"] for f in folds])
            avg_degradation = np.mean([f["degradation"] for f in folds])
            is_sharpes = [f["train_sharpe"] for f in folds if f["train_sharpe"] != 0]
            oos_sharpes = [f["test_sharpe"] for f in folds]
            stability = np.mean(oos_sharpes) / np.mean(is_sharpes) if is_sharpes and np.mean(is_sharpes) != 0 else 0
        else:
            avg_oos_sharpe, avg_degradation, stability = 0.0, 0.0, 0.0
        
        return WalkForwardResult(
            folds=folds,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_degradation=avg_degradation,
            stability_score=stability,
        )
    
    def compare_benchmark(
        self,
        strategy_returns: list[float],
        benchmark_prices: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> BenchmarkComparison:
        """벤치마크 대비 성과"""
        years = (end_date - start_date).days / 365
        
        if strategy_returns:
            strategy_cumulative = (1 + pd.Series(strategy_returns)).cumprod()
            strategy_total = strategy_cumulative.iloc[-1] - 1
            strategy_cagr = calculate_cagr(strategy_total, years)
            strategy_sharpe = calculate_sharpe_ratio(pd.Series(strategy_returns), periods_per_year=12)
            strategy_max_dd = calculate_max_drawdown(strategy_cumulative)
        else:
            strategy_cagr, strategy_sharpe, strategy_max_dd = 0.0, 0.0, 0.0
        
        if len(benchmark_prices) > 1:
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_total = benchmark_cumulative.iloc[-1] - 1 if len(benchmark_cumulative) > 0 else 0
            benchmark_cagr = calculate_cagr(benchmark_total, years)
            benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns, periods_per_year=12)
            benchmark_max_dd = calculate_max_drawdown(benchmark_cumulative)
        else:
            benchmark_cagr, benchmark_sharpe, benchmark_max_dd = 0.0, 0.0, 0.0
        
        return BenchmarkComparison(
            strategy_cagr=strategy_cagr,
            benchmark_cagr=benchmark_cagr,
            excess_return=strategy_cagr - benchmark_cagr,
            strategy_sharpe=strategy_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            strategy_max_dd=strategy_max_dd,
            benchmark_max_dd=benchmark_max_dd,
        )
    
    def run_full_backtest(
        self,
        kci: pd.Series,
        prices: pd.Series,
        benchmark_prices: pd.Series = None,
    ) -> BacktestReport:
        """전체 백테스트 실행"""
        logger.info("전체 백테스트 시작")
        
        signals = self.generate_signals(kci)
        result = self.execute_backtest(kci, signals, prices)
        trades = result["trades"]
        
        start_date = kci.index[0]
        end_date = kci.index[-1]
        metrics = self.calculate_metrics(trades, result["daily_returns"], start_date, end_date)
        
        failed_cases = self.identify_failed_cases(trades)
        walk_forward = self.walk_forward_validation(kci, prices)
        
        if benchmark_prices is None:
            benchmark_prices = prices
        
        strategy_returns = [t["return_pct"] for t in trades]
        benchmark_comparison = self.compare_benchmark(strategy_returns, benchmark_prices, start_date, end_date)
        
        # 신뢰구간
        if strategy_returns:
            bootstrap_returns = [np.mean(np.random.choice(strategy_returns, len(strategy_returns), replace=True)) for _ in range(1000)]
            ci_95 = (np.percentile(bootstrap_returns, 2.5), np.percentile(bootstrap_returns, 97.5))
        else:
            ci_95 = (0.0, 0.0)
        
        report = BacktestReport(
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            win_rate=metrics["win_rate"],
            total_return=metrics["total_return"],
            cagr=metrics["cagr"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            avg_return=metrics["avg_return"],
            avg_winning_return=metrics["avg_winning_return"],
            avg_losing_return=metrics["avg_losing_return"],
            avg_holding_days=metrics["avg_holding_days"],
            total_commission=len(trades) * 2 * self.config.commission,
            total_slippage=len(trades) * 2 * self.config.slippage,
            trades=trades,
            failed_cases=failed_cases,
            walk_forward=walk_forward,
            benchmark_comparison=benchmark_comparison,
            return_ci_95=ci_95,
        )
        
        logger.info(f"백테스트 완료: {metrics['total_trades']}거래, 승률 {metrics['win_rate']:.1%}, Sharpe {metrics['sharpe_ratio']:.2f}")
        return report


class BacktestAgent:
    """백테스트 에이전트 (LangGraph 노드용)"""
    
    def __init__(self, config=None):
        self.engine = BacktestEngine(config)
    
    def run(self, state: dict) -> dict:
        """LangGraph 노드 실행"""
        try:
            kci_data = state.get("kci_monthly") or state.get("kci_weekly")
            
            if kci_data is None:
                return {"error_type": "validation_failed", "error_message": "KCI data is None"}
            
            if isinstance(kci_data, dict):
                kci = pd.Series(kci_data)
                kci.index = pd.to_datetime(kci.index)
            else:
                kci = kci_data
            
            prices = kci / kci.iloc[0] * 10000
            report = self.engine.run_full_backtest(kci, prices)
            
            signals = self.engine.generate_signals(kci)
            current_signal = "LONG" if len(signals) > 0 and signals.iloc[-1] == 1 else "FLAT"
            
            return {"backtest_report": report, "current_signal": current_signal}
            
        except Exception as e:
            logger.error(f"Backtest Agent 에러: {e}")
            return {"error_type": "backtest_failed", "error_message": str(e)}
