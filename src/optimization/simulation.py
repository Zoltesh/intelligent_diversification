"""
Production-grade, strictly sequential portfolio backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import polars as pl

getcontext().prec = 28

INITIAL_CAPITAL = Decimal("10000")
TRADING_FEE_RATE = Decimal("0.001")
PCT_PLACES = Decimal("0.000001")
WEIGHT_PLACES = Decimal("0.000001")
MONEY_PLACES = Decimal("0.01")
WEIGHT_EPS = Decimal("0.0000001")


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _fmt_decimal(value: Decimal, places: Decimal) -> str:
    return str(value.quantize(places, rounding=ROUND_HALF_UP))


def _safe_div(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return numerator / denominator


@dataclass(frozen=True)
class PricePoint:
    open: Decimal
    close: Decimal


class PortfolioSimulator:
    def __init__(
        self,
        weights_path: Path,
        data_dir: Path,
        initial_capital: Decimal = INITIAL_CAPITAL,
        fee_rate: Decimal = TRADING_FEE_RATE,
    ) -> None:
        self.weights_path = weights_path
        self.data_dir = data_dir
        self.initial_capital = _to_decimal(initial_capital)
        self.fee_rate = _to_decimal(fee_rate)

        self.weights_schedule = self._load_weights()
        self.symbols = sorted(self._discover_symbols())
        self.prices_by_symbol = self._load_market_data()

        self.min_ts, self.max_ts = self._get_simulation_window()
        self._filter_prices_to_window()

        self.current_units: Dict[str, Decimal] = {sym: Decimal("0") for sym in self.symbols}
        self.held_cash = self.initial_capital
        self.last_target_weights: Dict[str, Decimal] | None = None

    def _discover_symbols(self) -> List[str]:
        files = sorted(self.data_dir.glob("*_feature_df.csv"))
        if not files:
            raise FileNotFoundError(
                f"No engineered feature files found in {self.data_dir}"
            )
        return [f.name.replace("_feature_df.csv", "") for f in files]

    def _load_weights(self) -> List[Dict[str, object]]:
        with self.weights_path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("weekly_weights.json must be a non-empty list.")
        data = sorted(data, key=lambda item: item["timestamp"])
        return data

    def _get_simulation_window(self) -> Tuple[int, int]:
        timestamps = [entry["timestamp"] for entry in self.weights_schedule]
        return min(timestamps), max(timestamps)

    def _load_market_data(self) -> Dict[str, Dict[int, PricePoint]]:
        prices_by_symbol: Dict[str, Dict[int, PricePoint]] = {}
        for symbol in self.symbols:
            csv_path = self.data_dir / f"{symbol}_feature_df.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing data file: {csv_path}")

            # CRITICAL FIX: Force prices to be read as Strings to avoid float conversion errors
            # before they reach Decimal.
            df = pl.read_csv(
                csv_path,
                columns=["timestamp", "open", "close"],
                schema_overrides={"open": pl.String, "close": pl.String},
            )
            df = df.drop_nulls(subset=["timestamp", "open", "close"])

            price_map: Dict[int, PricePoint] = {}
            for row in df.iter_rows(named=True):
                ts = int(row["timestamp"])
                price_map[ts] = PricePoint(
                    open=_to_decimal(row["open"]),
                    close=_to_decimal(row["close"]),
                )
            prices_by_symbol[symbol] = price_map
        return prices_by_symbol

    def _filter_prices_to_window(self) -> None:
        for symbol, price_map in self.prices_by_symbol.items():
            filtered = {
                ts: price
                for ts, price in price_map.items()
                if self.min_ts <= ts <= self.max_ts
            }
            if not filtered:
                raise ValueError(f"No price data for {symbol} in window.")
            self.prices_by_symbol[symbol] = filtered

    def _get_prices(self, timestamp: int) -> Dict[str, PricePoint]:
        prices: Dict[str, PricePoint] = {}
        missing = []
        for symbol in self.symbols:
            price = self.prices_by_symbol[symbol].get(timestamp)
            if price is None:
                missing.append(symbol)
            else:
                prices[symbol] = price
        if missing:
            raise ValueError(
                f"Missing prices for timestamp {timestamp}: {', '.join(missing)}"
            )
        return prices

    def _compute_equity(self, prices: Dict[str, PricePoint]) -> Decimal:
        equity = self.held_cash
        for symbol in self.symbols:
            equity += self.current_units[symbol] * prices[symbol].open
        return equity

    def _compute_weights(self, prices: Dict[str, PricePoint], equity: Decimal) -> Dict[str, Decimal]:
        weights: Dict[str, Decimal] = {}
        for symbol in self.symbols:
            position_value = self.current_units[symbol] * prices[symbol].open
            weights[symbol] = _safe_div(position_value, equity)
        return weights

    def _rebalance(
        self,
        prices: Dict[str, PricePoint],
        target_weights: Dict[str, Decimal],
        equity: Decimal,
    ) -> Tuple[Decimal, Dict[str, Decimal]]:
        trade_values: Dict[str, Decimal] = {}
        for symbol in self.symbols:
            current_value = self.current_units[symbol] * prices[symbol].open
            target_value = equity * target_weights.get(symbol, Decimal("0"))
            trade_values[symbol] = target_value - current_value

        total_trade_value = sum(abs(v) for v in trade_values.values())
        fees = total_trade_value * self.fee_rate
        equity_after_fees = equity - fees

        new_units: Dict[str, Decimal] = {}
        for symbol in self.symbols:
            target_value = equity_after_fees * target_weights.get(symbol, Decimal("0"))
            price = prices[symbol].open
            new_units[symbol] = _safe_div(target_value, price)

        self.current_units = new_units
        self.held_cash = equity_after_fees - sum(
            self.current_units[symbol] * prices[symbol].open for symbol in self.symbols
        )

        return fees, trade_values

    def _weights_unchanged(self, target_weights: Dict[str, Decimal]) -> bool:
        if self.last_target_weights is None:
            return False
        for symbol in self.symbols:
            prev_weight = self.last_target_weights.get(symbol, Decimal("0"))
            next_weight = target_weights.get(symbol, Decimal("0"))
            if abs(prev_weight - next_weight) > WEIGHT_EPS:
                return False
        return True

    def run(self) -> Dict[str, object]:
        weekly_metrics: List[Dict[str, object]] = []
        net_returns: List[Decimal] = []
        total_fees_paid = Decimal("0")
        peak_value = self.initial_capital
        max_drawdown = Decimal("0")

        schedule_len = len(self.weights_schedule)
        for week_number, entry in enumerate(self.weights_schedule, start=1):
            timestamp = int(entry["timestamp"])
            weights_raw = entry["weights"]
            target_weights = {
                symbol: _to_decimal(weights_raw.get(symbol, 0))
                for symbol in self.symbols
            }

            prices = self._get_prices(timestamp)
            equity_before = self._compute_equity(prices)
            drifted_weights = self._compute_weights(prices, equity_before)

            if self._weights_unchanged(target_weights):
                fees = Decimal("0")
                trade_values = {symbol: Decimal("0") for symbol in self.symbols}
                start_value = equity_before
            else:
                fees, trade_values = self._rebalance(prices, target_weights, equity_before)
                total_fees_paid += fees
                start_value = self._compute_equity(prices)
                self.last_target_weights = target_weights

            if week_number < schedule_len:
                next_timestamp = int(self.weights_schedule[week_number]["timestamp"])
                next_prices = self._get_prices(next_timestamp)
                end_prices = {symbol: next_prices[symbol].open for symbol in self.symbols}
            else:
                end_prices = {symbol: prices[symbol].close for symbol in self.symbols}

            end_value_assets = {
                symbol: self.current_units[symbol] * end_prices[symbol]
                for symbol in self.symbols
            }
            end_value = self.held_cash + sum(end_value_assets.values())

            gross_return_pct = _safe_div(end_value + fees - equity_before, equity_before)
            net_return_pct = _safe_div(end_value - equity_before, equity_before)
            net_returns.append(net_return_pct)

            turnover_pct = _safe_div(sum(abs(v) for v in trade_values.values()), equity_before)

            peak_value = max(peak_value, end_value)
            drawdown_pct = _safe_div(end_value - peak_value, peak_value)
            max_drawdown = min(max_drawdown, drawdown_pct)

            assets_metrics: Dict[str, Dict[str, str]] = {}
            for symbol in self.symbols:
                target_weight = target_weights.get(symbol, Decimal("0"))
                drifted_weight = drifted_weights.get(symbol, Decimal("0"))
                allocated_value = start_value * target_weight
                end_value_asset = end_value_assets[symbol]
                pnl_usd = end_value_asset - allocated_value
                return_pct = _safe_div(
                    end_prices[symbol] - prices[symbol].open, prices[symbol].open
                )
                contribution_pct = return_pct * target_weight

                assets_metrics[symbol] = {
                    "target_weight": _fmt_decimal(target_weight, WEIGHT_PLACES),
                    "drifted_weight": _fmt_decimal(drifted_weight, WEIGHT_PLACES),
                    "allocated_value": _fmt_decimal(allocated_value, MONEY_PLACES),
                    "pnl_usd": _fmt_decimal(pnl_usd, MONEY_PLACES),
                    "return_pct": _fmt_decimal(return_pct, PCT_PLACES),
                    "contribution_pct": _fmt_decimal(contribution_pct, PCT_PLACES),
                }

            weekly_metrics.append(
                {
                    "timestamp": timestamp,
                    "week_number": week_number,
                    "portfolio": {
                        "start_value": _fmt_decimal(equity_before, MONEY_PLACES),
                        "end_value": _fmt_decimal(end_value, MONEY_PLACES),
                        "gross_return_pct": _fmt_decimal(gross_return_pct, PCT_PLACES),
                        "net_return_pct": _fmt_decimal(net_return_pct, PCT_PLACES),
                        "total_fees": _fmt_decimal(fees, MONEY_PLACES),
                        "turnover_pct": _fmt_decimal(turnover_pct, PCT_PLACES),
                        "drawdown_pct": _fmt_decimal(drawdown_pct, PCT_PLACES),
                    },
                    "assets": assets_metrics,
                }
            )

        summary = self._build_summary(
            weekly_metrics=weekly_metrics,
            net_returns=net_returns,
            total_fees_paid=total_fees_paid,
        )

        return {"weekly_metrics": weekly_metrics, "summary": summary}

    def _build_summary(
        self,
        weekly_metrics: List[Dict[str, object]],
        net_returns: Iterable[Decimal],
        total_fees_paid: Decimal,
    ) -> Dict[str, str]:
        if not weekly_metrics:
            raise ValueError("No weekly metrics produced.")

        final_balance = _to_decimal(weekly_metrics[-1]["portfolio"]["end_value"])
        total_return_pct = _safe_div(final_balance - self.initial_capital, self.initial_capital)

        num_weeks = Decimal(str(len(weekly_metrics)))
        years = _safe_div(num_weeks, Decimal("52"))
        if years > 0:
            cagr = _to_decimal(
                float(final_balance / self.initial_capital) ** (1 / float(years)) - 1
            )
        else:
            cagr = Decimal("0")

        net_returns_list = [float(r) for r in net_returns]
        if net_returns_list:
            mean_return = sum(net_returns_list) / len(net_returns_list)
            variance = sum((r - mean_return) ** 2 for r in net_returns_list) / len(
                net_returns_list
            )
            std_dev = sqrt(variance)
            sharpe_ratio = (
                (mean_return / std_dev) * sqrt(52) if std_dev > 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        max_drawdown_pct = min(
            _to_decimal(w["portfolio"]["drawdown_pct"]) for w in weekly_metrics
        )

        return {
            "total_return_pct": _fmt_decimal(total_return_pct, PCT_PLACES),
            "cagr": _fmt_decimal(cagr, PCT_PLACES),
            "max_drawdown_pct": _fmt_decimal(max_drawdown_pct, PCT_PLACES),
            "sharpe_ratio": _fmt_decimal(_to_decimal(sharpe_ratio), PCT_PLACES),
            "total_fees_paid": _fmt_decimal(total_fees_paid, MONEY_PLACES),
            "final_balance": _fmt_decimal(final_balance, MONEY_PLACES),
        }


def main(weights_filename: str) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    weights_path = base_dir / "results" / f"{weights_filename}.json"
    data_dir = base_dir / "engineered_features"
    output_path = base_dir / "results" / f"weekly_backtest_metrics_{weights_filename}.json"

    simulator = PortfolioSimulator(weights_path=weights_path, data_dir=data_dir)
    results = simulator.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main("weekly_weights")
