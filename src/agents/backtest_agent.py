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
        - "주간" 변화율 > threshold (데이터 주기가 주간이면 1 period, 일간이면 5 period)
        - MA 상향 돌파
        """
        signals = pd.Series(0, index=kci.index)
        ma_period = self.config.ma_period
        
        if len(kci) < ma_period + 5:
            logger.warning(f"데이터 부족: {len(kci)} < {ma_period + 5}")
            return signals
        
        freq = pd.infer_freq(pd.DatetimeIndex(kci.index))
        # 주간이면 1주 변화, 일간이면 5영업일 변화로 근사
        change_lookback = 1
        if freq and freq.startswith("D"):
            change_lookback = 5
        elif freq and freq.startswith("W"):
            change_lookback = 1
        elif freq and freq.startswith("M"):
            change_lookback = 1
        
        for t in range(ma_period + 1, len(kci)):
            # t 시점에서는 t-1까지의 데이터만 사용 가능
            available = kci.iloc[:t]
            if len(available) < change_lookback + 2:
                continue
            
            # 조건 1: 변화율 (lookback)
            recent_change = available.iloc[-1] / available.iloc[-(change_lookback + 1)] - 1
            
            # 조건 2: MA 상향 돌파
            current_ma = available.iloc[-ma_period:].mean()
            prev_ma = available.iloc[-(ma_period + 1):-1].mean()
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
        """백테스트 실행
        
        Returns:
            dict: {trades, returns, equity_curve}
        """
        trades: list[TradeRecord] = []
        
        # 정렬/정합성
        prices = prices.reindex(kci.index).ffill().bfill()
        signals = signals.reindex(kci.index).fillna(0).astype(int)
        
        # 실행 지연(기본 1 period): signal(t) → execute(t+delay)로 처리
        exec_signals = signals.shift(self.config.execution_delay).fillna(0).astype(int)
        
        equity = 1.0
        cash = 1.0
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_date = None
        
        equity_curve = []
        returns = []
        prev_equity = equity
        
        for t, today in enumerate(kci.index):
            price = float(prices.loc[today])
            signal_today = int(exec_signals.loc[today])
            
            # ===== 진입 (오늘 체결) =====
            if signal_today == 1 and position == 0:
                invest = self.config.position_size * cash
                fill_price = price * (1 + self.config.slippage) * (1 + self.config.commission)
                if fill_price > 0 and invest > 0:
                    shares = invest / fill_price
                    cash = cash - invest
                    entry_price = fill_price
                    entry_date = today
                    position = 1
                    logger.debug(f"진입: {today} @ {entry_price:.0f} (shares={shares:.6f})")
            
            # ===== 포지션 보유/청산 판단 =====
            exit_reason = None
            if position == 1 and entry_date is not None:
                holding_days = (today - entry_date).days
                current_return = price / entry_price - 1 if entry_price > 0 else 0.0
                
                if holding_days >= self.config.holding_days:
                    exit_reason = "time_limit"
                if current_return <= self.config.stop_loss:
                    exit_reason = "stop_loss"
                
                if exit_reason:
                    exit_price = price * (1 - self.config.slippage) * (1 - self.config.commission)
                    proceeds = shares * exit_price
                    cash = cash + proceeds
                    trade_return = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                    
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
                    # reset
                    shares = 0.0
                    position = 0
                    entry_price = 0.0
                    entry_date = None
            
            # ===== 포트폴리오 가치 (마크투마켓) =====
            equity = cash + shares * price
            equity_curve.append(equity)
            
            if t == 0:
                returns.append(0.0)
            else:
                r = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
                returns.append(r)
            prev_equity = equity
        
        equity_curve = pd.Series(equity_curve, index=kci.index, name="equity")
        returns = pd.Series(returns, index=kci.index, name="strategy_return")
        
        return {"trades": trades, "returns": returns, "equity_curve": equity_curve}

    def calculate_metrics(
        self,
        trades: list[TradeRecord],
        returns: pd.Series,
        equity_curve: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """성과 지표 계산 (포트폴리오 기준)"""
        # 기본값
        if returns is None or len(returns) < 2:
            returns = pd.Series([0.0], index=[start_date])
        if equity_curve is None or len(equity_curve) == 0:
            equity_curve = pd.Series([1.0], index=[start_date])
        
        years = (end_date - start_date).days / 365
        total_return = float(equity_curve.iloc[-1] - 1.0)
        
        # 주기 추정 → Sharpe/Sortino 스케일링
        freq = pd.infer_freq(pd.DatetimeIndex(returns.index))
        if freq and freq.startswith("D"):
            ppy = 252
        elif freq and freq.startswith("W"):
            ppy = 52
        elif freq and freq.startswith("M"):
            ppy = 12
        else:
            ppy = 52
        
        # Trade 통계
        trade_returns = [t["return_pct"] for t in trades] if trades else []
        winning = [r for r in trade_returns if r > 0]
        losing = [r for r in trade_returns if r <= 0]
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades) if trades else 0.0,
            "total_return": total_return,
            "cagr": calculate_cagr(total_return, years),
            "sharpe_ratio": calculate_sharpe_ratio(returns.dropna(), periods_per_year=ppy),
            "sortino_ratio": calculate_sortino_ratio(returns.dropna(), periods_per_year=ppy),
            "max_drawdown": float(calculate_max_drawdown(equity_curve)),
            "avg_return": float(np.mean(trade_returns)) if trade_returns else 0.0,
            "avg_winning_return": float(np.mean(winning)) if winning else 0.0,
            "avg_losing_return": float(np.mean(losing)) if losing else 0.0,
            "avg_holding_days": float(np.mean([t["holding_days"] for t in trades])) if trades else 0.0,
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
        """Walk-forward 검증
        
        config는 "months" 단위를 사용하므로, 입력 시계열 주기에 맞춰 period로 변환한다.
        - Weekly: 1 month ≈ 4 weeks
        - Daily: 1 month ≈ 21 trading days
        - Monthly: 1 month = 1 period
        """
        folds = []
        total_periods = len(kci)
        
        freq = pd.infer_freq(pd.DatetimeIndex(kci.index))
        if freq and freq.startswith("D"):
            per_month = 21
        elif freq and freq.startswith("W"):
            per_month = 4
        elif freq and freq.startswith("M"):
            per_month = 1
        else:
            per_month = 4
        
        in_p = int(self.config.in_sample_months * per_month)
        out_p = int(self.config.out_sample_months * per_month)
        step_p = int(self.config.step_months * per_month)
        in_p = max(in_p, 10)
        out_p = max(out_p, 4)
        step_p = max(step_p, out_p)
        
        current_start = 0
        fold_num = 1
        
        while current_start + in_p + out_p <= total_periods:
            train_end = current_start + in_p
            test_end = train_end + out_p
            
            # In-sample
            kci_train = kci.iloc[current_start:train_end]
            prices_train = prices.iloc[current_start:train_end]
            signals_train = self.generate_signals(kci_train)
            train_result = self.execute_backtest(kci_train, signals_train, prices_train)
            train_metrics = self.calculate_metrics(
                train_result["trades"], train_result["returns"], train_result["equity_curve"],
                kci_train.index[0], kci_train.index[-1]
            )
            
            # Out-of-sample
            kci_test = kci.iloc[train_end:test_end]
            prices_test = prices.iloc[train_end:test_end]
            # 신호는 train_end 이전 정보만으로 생성해야 하므로, 전체를 넣고 OOS 구간만 슬라이스
            signals_full = self.generate_signals(kci.iloc[:test_end])
            signals_test = signals_full.iloc[train_end:test_end]
            test_result = self.execute_backtest(kci_test, signals_test, prices_test)
            test_metrics = self.calculate_metrics(
                test_result["trades"], test_result["returns"], test_result["equity_curve"],
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
            
            current_start += step_p
            fold_num += 1
        
        if folds:
            avg_oos_sharpe = float(np.mean([f["test_sharpe"] for f in folds]))
            avg_degradation = float(np.mean([f["degradation"] for f in folds]))
            is_sharpes = [f["train_sharpe"] for f in folds if f["train_sharpe"] != 0]
            oos_sharpes = [f["test_sharpe"] for f in folds]
            stability = float(np.mean(oos_sharpes) / np.mean(is_sharpes)) if is_sharpes and np.mean(is_sharpes) != 0 else 0.0
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
        strategy_returns: pd.Series,
        benchmark_prices: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> BenchmarkComparison:
        """벤치마크 대비 성과 (포트폴리오 수익률 기준)"""
        years = (end_date - start_date).days / 365
        
        # 주기 추정
        freq = pd.infer_freq(pd.DatetimeIndex(strategy_returns.index))
        if freq and freq.startswith("D"):
            ppy = 252
        elif freq and freq.startswith("W"):
            ppy = 52
        elif freq and freq.startswith("M"):
            ppy = 12
        else:
            ppy = 52
        
        # Strategy
        sr = strategy_returns.dropna()
        strat_cum = (1 + sr).cumprod()
        strat_total = float(strat_cum.iloc[-1] - 1) if len(strat_cum) > 0 else 0.0
        strategy_cagr = calculate_cagr(strat_total, years)
        strategy_sharpe = calculate_sharpe_ratio(sr, periods_per_year=ppy) if len(sr) > 1 else 0.0
        strategy_max_dd = float(calculate_max_drawdown(strat_cum)) if len(strat_cum) > 0 else 0.0
        
        # Benchmark
        benchmark_prices = benchmark_prices.reindex(strategy_returns.index).ffill().bfill()
        if len(benchmark_prices) > 1:
            br = benchmark_prices.pct_change().dropna()
            bench_cum = (1 + br).cumprod()
            bench_total = float(bench_cum.iloc[-1] - 1) if len(bench_cum) > 0 else 0.0
            benchmark_cagr = calculate_cagr(bench_total, years)
            benchmark_sharpe = calculate_sharpe_ratio(br, periods_per_year=ppy) if len(br) > 1 else 0.0
            benchmark_max_dd = float(calculate_max_drawdown(bench_cum)) if len(bench_cum) > 0 else 0.0
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
        returns = result["returns"]
        equity_curve = result["equity_curve"]
        
        start_date = kci.index[0]
        end_date = kci.index[-1]
        metrics = self.calculate_metrics(trades, returns, equity_curve, start_date, end_date)
        
        failed_cases = self.identify_failed_cases(trades)
        walk_forward = self.walk_forward_validation(kci, prices.reindex(kci.index).ffill().bfill())
        
        if benchmark_prices is None:
            benchmark_prices = prices
        benchmark_prices = benchmark_prices.reindex(kci.index).ffill().bfill()
        
        benchmark_comparison = self.compare_benchmark(returns, benchmark_prices, start_date, end_date)
        
        # 신뢰구간 (period 수익률 평균의 부트스트랩)
        if len(returns.dropna()) > 1:
            r = returns.dropna().values
            bootstrap_means = [float(np.mean(np.random.choice(r, len(r), replace=True))) for _ in range(1000)]
            ci_95 = (float(np.percentile(bootstrap_means, 2.5)), float(np.percentile(bootstrap_means, 97.5)))
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
        
        logger.info(
            f"백테스트 완료: {metrics['total_trades']}거래, 승률 {metrics['win_rate']:.1%}, Sharpe {metrics['sharpe_ratio']:.2f}"
        )
        return report

class BacktestAgent:
    """백테스트 에이전트 (LangGraph 노드용)"""
    
    def __init__(self, config=None):
        self.engine = BacktestEngine(config)
    
    def run(self, state: dict) -> dict:
        """LangGraph 노드 실행"""
        try:
            # ===== KCI 입력 (P0 Fix: backtest는 주간 KCI를 기본으로 사용) =====
            kci_weekly_data = state.get("kci_weekly")
            kci_monthly_data = state.get("kci_monthly")
            
            kci_data = kci_weekly_data if kci_weekly_data is not None else kci_monthly_data
            if kci_data is None:
                return {"error_type": "validation_failed", "error_message": "KCI data is None"}
            
            if isinstance(kci_data, dict):
                kci = pd.Series(kci_data)
                kci.index = pd.to_datetime(kci.index)
            else:
                kci = kci_data
            
            # monthly fallback 경고
            if kci_weekly_data is None and kci_monthly_data is not None:
                logger.warning("kci_weekly가 없어 kci_monthly로 백테스트를 수행합니다. (전략 파라미터/해석에 주의)")
            
            # ===== 타겟 자산 가격 (P0 Fix: KCI를 가격 프록시로 쓰는 것을 기본값에서 제거) =====
            # state에 target_prices(Series/dict)가 있으면 그걸 사용. 없으면 프록시로 fallback하되 명시적으로 표시
            price_source = "KCI_PROXY"
            prices_data = state.get("target_prices") or state.get("target_asset_prices")
            benchmark_data = state.get("benchmark_prices")
            
            if prices_data is not None:
                if isinstance(prices_data, dict):
                    prices = pd.Series(prices_data)
                    prices.index = pd.to_datetime(prices.index)
                else:
                    prices = prices_data
                price_source = "TARGET_ASSET"
            else:
                prices = kci / kci.iloc[0] * 10000
            
            if benchmark_data is not None:
                if isinstance(benchmark_data, dict):
                    benchmark_prices = pd.Series(benchmark_data)
                    benchmark_prices.index = pd.to_datetime(benchmark_prices.index)
                else:
                    benchmark_prices = benchmark_data
            else:
                benchmark_prices = prices
            
            report = self.engine.run_full_backtest(kci, prices, benchmark_prices=benchmark_prices)
            
            signals = self.engine.generate_signals(kci)
            current_signal = "LONG" if len(signals) > 0 and signals.iloc[-1] == 1 else "FLAT"
            
            return {"backtest_report": report, "current_signal": current_signal, "price_source": price_source}
            
        except Exception as e:
            logger.error(f"Backtest Agent 에러: {e}")
            return {"error_type": "backtest_failed", "error_message": str(e)}
