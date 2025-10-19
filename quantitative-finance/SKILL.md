---
name: quantitative-finance
description: Expert guidance for quantitative finance, algorithmic trading, systematic trading strategies, quantitative research, and trading systems development. Use when developing trading algorithms, backtesting strategies, building trading infrastructure, conducting quantitative research, or implementing production trading systems.
---

# Quantitative Finance & Trading

Expert guidance for senior quantitative developers, researchers, and systematic traders building sophisticated trading systems and conducting quantitative research.

## Core Competencies

### Quantitative Developer
- Trading system architecture and implementation
- High-performance computing for trading
- Order management and execution systems
- Market data processing and storage
- Production system deployment and monitoring

### Quantitative Researcher
- Alpha research and strategy development
- Statistical analysis and hypothesis testing
- Factor modeling and analysis
- Backtesting frameworks and methodology
- Research reproducibility and documentation

### Systematic Trader
- Strategy implementation and optimization
- Risk management and position sizing
- Portfolio construction and rebalancing
- Performance attribution and analysis
- Market regime detection

### Quantitative Trader
- Trade execution optimization
- Market microstructure analysis
- Transaction cost analysis
- Slippage and market impact modeling
- Real-time decision making

## Quantitative Research Framework

### Research Workflow
```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class AlphaResearch:
    """Framework for systematic alpha research."""
    
    def __init__(self, universe: List[str], start_date: str, end_date: str):
        self.universe = universe
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data = None
        self.signals = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean market data."""
        # Load price, volume, fundamental data
        data = self._fetch_data()
        data = self._clean_data(data)
        data = self._calculate_returns(data)
        self.data = data
        return data
    
    def generate_alpha(self, params: Dict) -> pd.DataFrame:
        """Generate alpha signals based on strategy logic."""
        signals = pd.DataFrame(index=self.data.index)
        
        # Example: Mean reversion strategy
        lookback = params.get('lookback', 20)
        zscore = params.get('zscore_threshold', 2.0)
        
        for ticker in self.universe:
            price = self.data[ticker]['close']
            ma = price.rolling(lookback).mean()
            std = price.rolling(lookback).std()
            zscore_val = (price - ma) / std
            
            signals[ticker] = -np.sign(zscore_val) * (np.abs(zscore_val) > zscore)
        
        self.signals = signals
        return signals
    
    def backtest(self, signals: pd.DataFrame, 
                 transaction_costs: float = 0.001) -> Dict:
        """Backtest strategy with realistic assumptions."""
        returns = self.data.xs('returns', level=1, axis=1)
        
        # Position changes (trades)
        positions = signals.shift(1).fillna(0)
        trades = positions.diff().fillna(positions)
        
        # Calculate returns
        strategy_returns = (positions * returns).sum(axis=1)
        
        # Apply transaction costs
        tc = np.abs(trades).sum(axis=1) * transaction_costs
        strategy_returns = strategy_returns - tc
        
        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns)
        
        return {
            'returns': strategy_returns,
            'positions': positions,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics."""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).sum() / len(returns)
        }
```

### Strategy Development Process

1. **Hypothesis Formation**
   - Economic intuition
   - Market observations
   - Academic research
   - Data mining (with proper statistical controls)

2. **Data Collection & Cleaning**
   - Survivorship bias elimination
   - Point-in-time data
   - Corporate actions adjustment
   - Data quality checks

3. **Feature Engineering**
   - Technical indicators
   - Fundamental ratios
   - Alternative data
   - Cross-sectional features

4. **Backtesting**
   - Out-of-sample testing
   - Walk-forward analysis
   - Monte Carlo simulation
   - Sensitivity analysis

5. **Risk Management**
   - Position sizing
   - Stop losses
   - Portfolio constraints
   - Correlation analysis

## Trading System Architecture

### High-Performance Trading System
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import asyncio
from datetime import datetime
import numpy as np

@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[float] = None
    timestamp: datetime = None
    order_id: Optional[str] = None
    status: str = 'pending'

@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class TradingStrategy(ABC):
    """Base class for trading strategies."""
    
    @abstractmethod
    async def on_market_data(self, data: Dict) -> List[Order]:
        """Process market data and generate orders."""
        pass
    
    @abstractmethod
    async def on_order_update(self, order: Order) -> None:
        """Handle order status updates."""
        pass
    
    @abstractmethod
    async def on_position_update(self, position: Position) -> None:
        """Handle position updates."""
        pass

class OrderManagementSystem:
    """Order management and execution system."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.fills: List[Dict] = []
        
    async def submit_order(self, order: Order) -> str:
        """Submit order to exchange."""
        order.timestamp = datetime.now()
        order.order_id = self._generate_order_id()
        
        # Validate order
        if not self._validate_order(order):
            order.status = 'rejected'
            return order.order_id
        
        # Risk checks
        if not self._risk_check(order):
            order.status = 'rejected'
            return order.order_id
        
        # Submit to exchange
        order.status = 'submitted'
        self.orders[order.order_id] = order
        
        # Simulate async execution
        asyncio.create_task(self._execute_order(order))
        
        return order.order_id
    
    async def _execute_order(self, order: Order) -> None:
        """Simulate order execution."""
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Fill order
        fill_price = order.price if order.order_type == 'limit' else self._get_market_price(order.symbol)
        
        fill = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': fill_price,
            'timestamp': datetime.now()
        }
        
        self.fills.append(fill)
        order.status = 'filled'
        
        # Update positions
        self._update_position(fill)
    
    def _update_position(self, fill: Dict) -> None:
        """Update position based on fill."""
        symbol = fill['symbol']
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=fill['quantity'],
                entry_price=fill['price'],
                current_price=fill['price'],
                unrealized_pnl=0.0
            )
        else:
            pos = self.positions[symbol]
            # Update position (simplified)
            total_cost = pos.quantity * pos.entry_price + fill['quantity'] * fill['price']
            pos.quantity += fill['quantity']
            pos.entry_price = total_cost / pos.quantity if pos.quantity != 0 else 0
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            return False
        if order.order_type == 'limit' and order.price is None:
            return False
        return True
    
    def _risk_check(self, order: Order) -> bool:
        """Pre-trade risk checks."""
        # Check position limits
        # Check notional limits
        # Check leverage
        # Check buying power
        return True
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        # In production, fetch from market data feed
        return 100.0

class RiskManager:
    """Real-time risk management."""
    
    def __init__(self, max_position_size: float, max_portfolio_var: float):
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        
    def check_position_limit(self, symbol: str, quantity: int, 
                            current_position: int) -> bool:
        """Check if new position would exceed limits."""
        new_position = current_position + quantity
        return abs(new_position) <= self.max_position_size
    
    def check_portfolio_risk(self, positions: Dict[str, Position]) -> bool:
        """Check portfolio-level risk metrics."""
        # Calculate portfolio VaR
        portfolio_var = self._calculate_portfolio_var(positions)
        return portfolio_var <= self.max_portfolio_var
    
    def calculate_position_size(self, signal_strength: float, 
                               volatility: float, 
                               account_value: float) -> int:
        """Calculate optimal position size using Kelly criterion."""
        # Simplified Kelly criterion
        win_rate = 0.55  # Historical win rate
        win_loss_ratio = 1.5  # Average win / average loss
        
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        
        # Adjust for signal strength and volatility
        position_size = kelly_fraction * account_value * signal_strength / volatility
        
        return int(position_size)
    
    def _calculate_portfolio_var(self, positions: Dict[str, Position], 
                                 confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk."""
        # Simplified VaR calculation
        # In production, use historical simulation or Monte Carlo
        
        portfolio_value = sum(p.quantity * p.current_price for p in positions.values())
        portfolio_volatility = 0.02  # Daily volatility
        
        z_score = 1.645 if confidence_level == 0.95 else 2.326
        var = portfolio_value * portfolio_volatility * z_score
        
        return var
```

## Statistical Methods for Trading

### Time Series Analysis
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

class TimeSeriesAnalysis:
    """Time series analysis for trading strategies."""
    
    @staticmethod
    def test_stationarity(series: pd.Series) -> Dict:
        """Augmented Dickey-Fuller test for stationarity."""
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    @staticmethod
    def calculate_half_life(series: pd.Series) -> float:
        """Calculate mean reversion half-life."""
        lag = series.shift(1)
        delta = series - lag
        
        # Run OLS regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(lag.dropna().values.reshape(-1, 1), 
                 delta.dropna().values)
        
        lambda_coef = model.coef_[0]
        half_life = -np.log(2) / lambda_coef
        
        return half_life
    
    @staticmethod
    def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> Dict:
        """Fit GARCH model for volatility forecasting."""
        model = arch_model(returns, vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        
        # Forecast next period volatility
        forecast = result.forecast(horizon=1)
        next_vol = np.sqrt(forecast.variance.values[-1, 0])
        
        return {
            'model': result,
            'forecast_volatility': next_vol,
            'params': result.params
        }
    
    @staticmethod
    def cointegration_test(series1: pd.Series, series2: pd.Series) -> Dict:
        """Test for cointegration between two series."""
        from statsmodels.tsa.stattools import coint
        
        score, pvalue, _ = coint(series1, series2)
        
        return {
            'test_statistic': score,
            'p_value': pvalue,
            'is_cointegrated': pvalue < 0.05
        }
```

### Factor Models
```python
import pandas as pd
import numpy as np
from typing import List, Dict

class FactorModel:
    """Multi-factor model for return attribution."""
    
    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_returns = None
        self.factor_loadings = None
        
    def fit(self, returns: pd.DataFrame, factor_data: pd.DataFrame) -> None:
        """Fit factor model to historical data."""
        from sklearn.linear_model import LinearRegression
        
        self.factor_returns = factor_data[self.factors]
        self.factor_loadings = pd.DataFrame(index=returns.columns, 
                                           columns=self.factors)
        
        for asset in returns.columns:
            model = LinearRegression()
            model.fit(self.factor_returns.values, returns[asset].values)
            self.factor_loadings.loc[asset] = model.coef_
    
    def calculate_factor_returns(self, date: str) -> pd.Series:
        """Calculate factor returns for a given date."""
        return self.factor_returns.loc[date]
    
    def attribute_returns(self, portfolio_weights: pd.Series, 
                         date: str) -> Dict:
        """Attribute portfolio returns to factors."""
        factor_rets = self.calculate_factor_returns(date)
        
        # Calculate factor contribution
        factor_contrib = {}
        for factor in self.factors:
            loadings = self.factor_loadings[factor]
            weighted_loadings = (loadings * portfolio_weights).sum()
            factor_contrib[factor] = weighted_loadings * factor_rets[factor]
        
        return factor_contrib
```

## Machine Learning for Trading

### Feature Engineering for ML
```python
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineering:
    """Feature engineering for ML trading strategies."""
    
    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        features = df.copy()
        
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'sma_{window}'] = df['close'].rolling(window).mean()
            features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Volatility
        features['volatility_20'] = df['returns'].rolling(20).std()
        features['atr_14'] = self._calculate_atr(df, 14)
        
        # Momentum
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['macd'] = self._calculate_macd(df['close'])
        
        # Volume
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        return features
    
    @staticmethod
    def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        features = df.copy()
        
        # Bid-ask spread
        features['spread'] = df['ask'] - df['bid']
        features['spread_bps'] = features['spread'] / df['mid'] * 10000
        
        # Order flow imbalance
        features['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        
        # VWAP
        features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_distance'] = (df['close'] - features['vwap']) / features['vwap']
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, 
                       fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr

class MLTradingModel:
    """Machine learning model for trading."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2) -> Dict:
        """Train ML model with proper validation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train with cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)
        
        # Get feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores),
            'feature_importance': self.feature_importance
        }
    
    def predict_signal(self, X: pd.DataFrame) -> np.ndarray:
        """Generate trading signals."""
        probabilities = self.model.predict_proba(X)
        # Convert probabilities to signals (-1, 0, 1)
        signals = np.where(probabilities[:, 1] > 0.55, 1,
                          np.where(probabilities[:, 1] < 0.45, -1, 0))
        return signals
```

## Backtesting Framework

### Professional Backtesting Engine
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 1000000
    commission: float = 0.001  # 10 bps
    slippage: float = 0.0005   # 5 bps
    margin_requirement: float = 0.25
    position_limit: float = 0.1  # 10% per position
    
class Backtester:
    """Professional backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Dict] = []
        self.portfolio_values: List[float] = []
        self.positions: Dict[str, int] = {}
        
    def run(self, strategy: Callable, data: pd.DataFrame) -> Dict:
        """Run backtest."""
        cash = self.config.initial_capital
        portfolio_value = cash
        
        for date, row in data.iterrows():
            # Generate signals
            signals = strategy(date, row, self.positions)
            
            # Execute trades
            for symbol, signal in signals.items():
                if signal != 0:
                    trade = self._execute_trade(
                        symbol, signal, row[symbol]['close'], 
                        date, cash, portfolio_value
                    )
                    
                    if trade:
                        self.trades.append(trade)
                        cash -= trade['cost']
            
            # Update portfolio value
            positions_value = sum(
                self.positions.get(symbol, 0) * row[symbol]['close']
                for symbol in self.positions
            )
            portfolio_value = cash + positions_value
            self.portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            'trades': pd.DataFrame(self.trades),
            'portfolio_values': pd.Series(self.portfolio_values, index=data.index),
            'metrics': metrics
        }
    
    def _execute_trade(self, symbol: str, signal: int, price: float,
                      date: pd.Timestamp, cash: float, 
                      portfolio_value: float) -> Dict:
        """Execute trade with realistic assumptions."""
        # Calculate position size
        max_position_value = portfolio_value * self.config.position_limit
        
        if signal > 0:  # Buy
            shares = int(max_position_value / price)
        else:  # Sell
            shares = -self.positions.get(symbol, 0)
        
        if shares == 0:
            return None
        
        # Apply slippage
        execution_price = price * (1 + self.config.slippage * np.sign(shares))
        
        # Calculate costs
        notional = abs(shares * execution_price)
        commission = notional * self.config.commission
        total_cost = shares * execution_price + commission
        
        # Check if we have enough cash
        if total_cost > cash:
            return None
        
        # Update position
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        
        return {
            'date': date,
            'symbol': symbol,
            'shares': shares,
            'price': execution_price,
            'commission': commission,
            'cost': total_cost
        }
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        # Basic metrics
        total_return = (self.portfolio_values[-1] / self.config.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            avg_trade_size = trades_df['cost'].abs().mean()
            total_commission = trades_df['commission'].sum()
        else:
            avg_trade_size = 0
            total_commission = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'num_trades': len(self.trades),
            'avg_trade_size': avg_trade_size,
            'total_commission': total_commission,
            'win_rate': (returns > 0).sum() / len(returns)
        }
```

## Market Microstructure

### Order Book Analysis
```python
import pandas as pd
import numpy as np
from typing import Dict, List

class OrderBook:
    """Limit order book analysis."""
    
    def __init__(self):
        self.bids: List[Dict] = []
        self.asks: List[Dict] = []
        
    def update(self, bids: List[Dict], asks: List[Dict]) -> None:
        """Update order book."""
        self.bids = sorted(bids, key=lambda x: x['price'], reverse=True)
        self.asks = sorted(asks, key=lambda x: x['price'])
    
    def get_best_bid_ask(self) -> tuple:
        """Get best bid and ask."""
        best_bid = self.bids[0]['price'] if self.bids else None
        best_ask = self.asks[0]['price'] if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> float:
        """Calculate mid price."""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> float:
        """Calculate bid-ask spread."""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def calculate_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum(b['size'] for b in self.bids[:levels])
        ask_volume = sum(a['size'] for a in self.asks[:levels])
        
        if bid_volume + ask_volume > 0:
            return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return 0
    
    def estimate_market_impact(self, order_size: int, 
                              side: str) -> float:
        """Estimate market impact of an order."""
        if side == 'buy':
            levels = self.asks
        else:
            levels = self.bids
        
        remaining_size = order_size
        total_cost = 0
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            fill_size = min(remaining_size, level['size'])
            total_cost += fill_size * level['price']
            remaining_size -= fill_size
        
        if remaining_size > 0:
            # Order cannot be fully filled
            return float('inf')
        
        avg_price = total_cost / order_size
        mid_price = self.get_mid_price()
        
        return (avg_price - mid_price) / mid_price

class ExecutionOptimizer:
    """Optimal execution strategies."""
    
    @staticmethod
    def twap_schedule(total_size: int, duration_minutes: int,
                     interval_minutes: int = 1) -> List[int]:
        """Time-Weighted Average Price execution schedule."""
        num_intervals = duration_minutes // interval_minutes
        size_per_interval = total_size // num_intervals
        
        schedule = [size_per_interval] * num_intervals
        # Add remainder to last interval
        schedule[-1] += total_size - sum(schedule)
        
        return schedule
    
    @staticmethod
    def vwap_schedule(total_size: int, 
                     historical_volume_profile: pd.Series) -> List[int]:
        """Volume-Weighted Average Price execution schedule."""
        total_hist_volume = historical_volume_profile.sum()
        
        schedule = []
        for vol in historical_volume_profile:
            size = int(total_size * (vol / total_hist_volume))
            schedule.append(size)
        
        # Adjust for rounding
        schedule[-1] += total_size - sum(schedule)
        
        return schedule
    
    @staticmethod
    def implementation_shortfall(arrival_price: float, 
                                execution_prices: List[float],
                                sizes: List[int],
                                commission: float = 0.001) -> Dict:
        """Calculate implementation shortfall."""
        total_size = sum(sizes)
        total_cost = sum(p * s for p, s in zip(execution_prices, sizes))
        avg_price = total_cost / total_size
        
        # Slippage
        slippage = (avg_price - arrival_price) / arrival_price
        
        # Commission
        commission_cost = total_cost * commission
        
        # Total shortfall
        total_shortfall = (avg_price - arrival_price) * total_size + commission_cost
        
        return {
            'avg_execution_price': avg_price,
            'slippage': slippage,
            'commission_cost': commission_cost,
            'total_shortfall': total_shortfall,
            'shortfall_bps': (total_shortfall / (arrival_price * total_size)) * 10000
        }
```

## Portfolio Construction & Optimization

### Mean-Variance Optimization
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional

class PortfolioOptimizer:
    """Portfolio construction and optimization."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(returns.columns)
        
    def max_sharpe_ratio(self, risk_free_rate: float = 0.02) -> Dict:
        """Maximize Sharpe ratio."""
        def neg_sharpe(weights):
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            neg_sharpe, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_return = np.sum(self.mean_returns * optimal_weights) * 252
        portfolio_vol = np.sqrt(
            np.dot(optimal_weights.T, np.dot(self.cov_matrix * 252, optimal_weights))
        )
        
        return {
            'weights': pd.Series(optimal_weights, index=self.returns.columns),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe': (portfolio_return - risk_free_rate) / portfolio_vol
        }
    
    def min_variance(self) -> Dict:
        """Minimum variance portfolio."""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            portfolio_variance, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_return = np.sum(self.mean_returns * optimal_weights) * 252
        portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))
        
        return {
            'weights': pd.Series(optimal_weights, index=self.returns.columns),
            'return': portfolio_return,
            'volatility': portfolio_vol
        }
    
    def risk_parity(self) -> Dict:
        """Risk parity portfolio."""
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def risk_parity_objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = portfolio_vol / self.num_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            risk_parity_objective, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_return = np.sum(self.mean_returns * optimal_weights) * 252
        portfolio_vol = np.sqrt(
            np.dot(optimal_weights.T, np.dot(self.cov_matrix * 252, optimal_weights))
        )
        
        return {
            'weights': pd.Series(optimal_weights, index=self.returns.columns),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'risk_contributions': risk_contribution(optimal_weights)
        }
    
    def black_litterman(self, market_caps: pd.Series, 
                       views: Optional[Dict] = None) -> Dict:
        """Black-Litterman portfolio optimization."""
        # Calculate market implied returns
        market_weights = market_caps / market_caps.sum()
        risk_aversion = 2.5
        
        market_returns = risk_aversion * np.dot(
            self.cov_matrix * 252, market_weights
        )
        
        if views is None:
            # No views, return market portfolio
            return {
                'weights': market_weights,
                'expected_returns': market_returns
            }
        
        # Incorporate views (simplified)
        # In production, implement full Black-Litterman model
        return {
            'weights': market_weights,
            'expected_returns': market_returns
        }
```

## Production Deployment Best Practices

### Monitoring and Alerting
```python
import logging
from typing import Dict, Callable
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

class TradingSystemMonitor:
    """Monitor trading system health."""
    
    def __init__(self, alert_email: str):
        self.alert_email = alert_email
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict = {}
        
    def check_system_health(self) -> bool:
        """Perform system health checks."""
        checks = [
            self._check_market_data_feed(),
            self._check_order_execution(),
            self._check_risk_limits(),
            self._check_position_reconciliation()
        ]
        
        return all(checks)
    
    def _check_market_data_feed(self) -> bool:
        """Check if market data feed is active."""
        # Check last update timestamp
        # Check data quality
        # Check for stale data
        return True
    
    def _check_order_execution(self) -> bool:
        """Check order execution system."""
        # Check pending orders
        # Check order latency
        # Check fill rates
        return True
    
    def _check_risk_limits(self) -> bool:
        """Check risk limit breaches."""
        # Check position limits
        # Check VaR limits
        # Check drawdown limits
        return True
    
    def _check_position_reconciliation(self) -> bool:
        """Reconcile positions with broker."""
        # Compare internal positions with broker
        # Check for discrepancies
        return True
    
    def send_alert(self, message: str, severity: str = 'WARNING') -> None:
        """Send alert notification."""
        self.logger.log(
            logging.WARNING if severity == 'WARNING' else logging.ERROR,
            message
        )
        
        # Send email alert
        self._send_email_alert(message, severity)
    
    def _send_email_alert(self, message: str, severity: str) -> None:
        """Send email alert."""
        subject = f"[{severity}] Trading System Alert"
        msg = MIMEText(f"{message}\n\nTimestamp: {datetime.now()}")
        msg['Subject'] = subject
        msg['From'] = 'trading-system@example.com'
        msg['To'] = self.alert_email
        
        # In production, configure SMTP server
        # smtplib.SMTP().send_message(msg)
```

## Best Practices

### Research Best Practices
- [ ] Document all hypotheses before testing
- [ ] Use out-of-sample testing
- [ ] Account for survivorship bias
- [ ] Adjust for corporate actions
- [ ] Use point-in-time data
- [ ] Control for multiple testing
- [ ] Implement proper cross-validation
- [ ] Calculate realistic transaction costs
- [ ] Test across different market regimes
- [ ] Perform Monte Carlo simulations

### Development Best Practices
- [ ] Version control all code
- [ ] Implement comprehensive testing
- [ ] Use type hints and documentation
- [ ] Implement proper error handling
- [ ] Log all trading decisions
- [ ] Separate research and production code
- [ ] Use configuration files
- [ ] Implement rollback capabilities
- [ ] Monitor system performance
- [ ] Implement circuit breakers

### Risk Management Best Practices
- [ ] Define maximum position sizes
- [ ] Implement portfolio-level risk limits
- [ ] Use stop losses
- [ ] Monitor correlation risk
- [ ] Calculate VaR daily
- [ ] Stress test strategies
- [ ] Implement position limits per sector
- [ ] Monitor leverage
- [ ] Track drawdowns
- [ ] Implement risk-adjusted position sizing

### Production Best Practices
- [ ] Implement redundancy
- [ ] Use separate dev/staging/prod environments
- [ ] Automated deployment pipeline
- [ ] Real-time monitoring
- [ ] Automated alerts
- [ ] Daily reconciliation
- [ ] Disaster recovery plan
- [ ] Performance benchmarking
- [ ] Regular system audits
- [ ] Documentation maintenance

## Tools and Libraries

### Essential Python Libraries
- **NumPy/Pandas**: Data manipulation
- **SciPy/Statsmodels**: Statistical analysis
- **Scikit-learn**: Machine learning
- **XGBoost/LightGBM**: Gradient boosting
- **PyTorch/TensorFlow**: Deep learning
- **Zipline/Backtrader**: Backtesting
- **QuantLib**: Quantitative finance
- **ccxt**: Cryptocurrency exchange integration
- **alpaca-trade-api**: Stock trading API

### Data Sources
- **Bloomberg Terminal**: Professional market data
- **Refinitiv**: Financial data and analytics
- **Interactive Brokers**: Trading and market data
- **Alpaca**: Commission-free stock trading
- **Polygon.io**: Real-time market data
- **Quandl**: Alternative data

### Infrastructure
- **PostgreSQL/TimescaleDB**: Time-series data storage
- **Redis**: Real-time data caching
- **Kafka**: Message streaming
- **Airflow**: Workflow orchestration
- **Grafana**: Monitoring dashboards
- **Prometheus**: Metrics collection

## Common Pitfalls

### Overfitting
```python
# Wrong - optimizing on same data used for testing
best_params = grid_search(data, params)
backtest(data, best_params)

# Correct - walk-forward optimization
for train_period, test_period in walk_forward_splits(data):
    params = optimize(train_period)
    results.append(backtest(test_period, params))
```

### Look-Ahead Bias
```python
# Wrong - using future data
df['signal'] = df['close'].shift(-1) > df['close']

# Correct - only use past data
df['signal'] = df['close'].shift(1) > df['close'].shift(2)
```

### Survivorship Bias
```python
# Wrong - only using currently listed stocks
universe = get_current_stock_universe()

# Correct - use point-in-time universe
universe = get_historical_universe(date)
```

### Transaction Costs
```python
# Wrong - ignoring costs
returns = signal * price_change

# Correct - include realistic costs
returns = signal * price_change - abs(signal.diff()) * (commission + slippage)
```

## References

### Books
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Quantitative Trading" by Ernest Chan
- "Algorithmic Trading" by Jeffrey Bacidore
- "Inside the Black Box" by Rishi K. Narang
- "Trading and Exchanges" by Larry Harris

### Academic Papers
- "The Econometrics of Financial Markets" - Campbell, Lo, MacKinlay
- "Market Microstructure" - O'Hara
- "Optimal Execution" - Almgren & Chriss
- "The Cross-Section of Expected Stock Returns" - Fama & French

### Industry Standards
- FIX Protocol for trading messages
- ISO 20022 for financial messaging
- MiFID II compliance requirements
- SEC Rule 15c3-5 (Market Access Rule)
