"""
System configuration: operation mode, network endpoints, trading rules,
risk limits, and executor parameters.

Two modes:
  test       - Binance Testnet + Arbitrum/Ethereum (dry-run)
  production - Binance Mainnet + Arbitrum One (live execution)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class OperationMode(str, Enum):
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True)
class NetworkPreset:
    name: str
    chain_id: int
    default_rpc: str
    router_address: str
    factory_address: str
    weth_address: str
    usdc_address: str


ETHEREUM_MAINNET = NetworkPreset(
    name="Ethereum Mainnet", chain_id=1,
    default_rpc="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
    router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    factory_address="0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
    weth_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    usdc_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
)

ARBITRUM_ONE = NetworkPreset(
    name="Arbitrum One", chain_id=42161,
    default_rpc="https://arb1.arbitrum.io/rpc",
    router_address="0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24",
    factory_address="0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9",
    weth_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    usdc_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
)

SUSHISWAP_ARB = NetworkPreset(
    name="SushiSwap Arbitrum", chain_id=42161,
    default_rpc="https://arb1.arbitrum.io/rpc",
    router_address="0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
    factory_address="0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
    weth_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    usdc_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
)


@dataclass
class BinanceTradingRules:
    pair: str = "ETH/USDC"
    min_notional_usd: float = 5.0
    lot_size_step: float = 0.0001
    price_tick: float = 0.01
    min_quantity: float = 0.0001

    def round_quantity(self, qty: float) -> float:
        return math.floor(qty / self.lot_size_step) * self.lot_size_step

    def round_price(self, price: float) -> float:
        return round(price / self.price_tick) * self.price_tick

    def validate_order(self, quantity: float, price: float) -> tuple[bool, str]:
        if quantity < self.min_quantity:
            return False, f"Quantity {quantity:.8f} below min {self.min_quantity:.8f}"
        notional = quantity * price
        if notional < self.min_notional_usd:
            return False, f"Notional ${notional:.2f} below min ${self.min_notional_usd:.2f}"
        return True, "OK"


@dataclass
class DEXConfig:
    rpc_url: str = ""
    ws_url: str = ""
    chain_id: int = 42161
    router_address: str = ARBITRUM_ONE.router_address
    factory_address: str = ARBITRUM_ONE.factory_address
    weth_address: str = ARBITRUM_ONE.weth_address
    usdc_address: str = ARBITRUM_ONE.usdc_address
    pool_addresses: list[str] = field(default_factory=list)
    gas_limit_swap: int = 500_000
    gas_limit_approval: int = 100_000
    slippage_tolerance_bps: Decimal = Decimal("50")
    deadline_seconds: int = 300
    dry_run: bool = True


@dataclass
class CEXConfig:
    api_key: str = ""
    secret: str = ""
    sandbox: bool = True
    enable_rate_limit: bool = True
    default_type: str = "spot"
    trading_rules: BinanceTradingRules = field(default_factory=BinanceTradingRules)


@dataclass
class ExecutorSettings:
    leg1_timeout: Decimal = Decimal("5")
    leg2_timeout: Decimal = Decimal("60")
    min_fill_ratio: Decimal = Decimal("0.80")
    var_confidence: Decimal = Decimal("0.95")
    vol_per_sqrt_second: Decimal = Decimal("0.0002")
    use_dex_first: bool = True
    max_recovery_attempts: int = 2
    unwind_timeout: Decimal = Decimal("30")
    unwind_slippage_bps: Decimal = Decimal("100")


@dataclass
class RiskConfig:
    max_trade_usd: float = 20.0
    max_trade_pct: float = 0.20
    max_position_per_token: float = 30.0
    max_open_positions: int = 1
    max_loss_per_trade: float = 5.0
    max_daily_loss: float = 15.0
    max_drawdown_pct: float = 0.20
    max_trades_per_hour: int = 20
    consecutive_loss_limit: int = 3
    max_spread_bps: float = 500.0
    max_signal_age_seconds: float = 5.0
    initial_capital: float = 100.0


@dataclass
class SystemConfig:
    mode: OperationMode = OperationMode.TEST
    dex: DEXConfig = field(default_factory=DEXConfig)
    cex: CEXConfig = field(default_factory=CEXConfig)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal_ttl_seconds: Decimal = Decimal("5")
    cooldown_seconds: Decimal = Decimal("2")
    kelly_fraction: Decimal = Decimal("0.25")
    max_position_usd: Decimal = Decimal("10000")
    cex_taker_bps: Decimal = Decimal("10")
    dex_swap_bps: Decimal = Decimal("30")
    gas_cost_usd: Decimal = Decimal("0.10")
    trading_pair: str = "ETH/USDC"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    log_dir: str = "logs"
    dry_run: bool = True

    @classmethod
    def from_env(cls) -> "SystemConfig":
        mode_str = os.getenv("OPERATION_MODE", "test").lower()
        mode = OperationMode.PRODUCTION if mode_str == "production" else OperationMode.TEST
        sandbox = mode == OperationMode.TEST
        dry_run = os.getenv("DRY_RUN", "true").lower() not in ("false", "0", "no")
        use_arb = os.getenv("USE_ARBITRUM", "true").lower() != "false"
        preset = ARBITRUM_ONE if use_arb else ETHEREUM_MAINNET

        dex = DEXConfig(
            rpc_url=os.getenv("ARBITRUM_RPC_URL", os.getenv("ETH_RPC_URL", preset.default_rpc)),
            ws_url=os.getenv("ARBITRUM_WS_URL", os.getenv("ETH_WS_URL", "")),
            chain_id=int(os.getenv("CHAIN_ID", str(preset.chain_id))),
            router_address=os.getenv("UNISWAP_V2_ROUTER", preset.router_address),
            factory_address=os.getenv("UNISWAP_V2_FACTORY", preset.factory_address),
            weth_address=os.getenv("WETH_ADDRESS", preset.weth_address),
            usdc_address=os.getenv("USDC_ADDRESS", preset.usdc_address),
            pool_addresses=[a.strip() for a in os.getenv("POOL_ADDRESSES", "").split(",") if a.strip()],
            gas_limit_swap=int(os.getenv("GAS_LIMIT_SWAP", "500000")),
            gas_limit_approval=int(os.getenv("GAS_LIMIT_APPROVAL", "100000")),
            slippage_tolerance_bps=Decimal(os.getenv("SLIPPAGE_TOLERANCE_BPS", "50")),
            deadline_seconds=int(os.getenv("TX_DEADLINE_SECONDS", "300")),
            dry_run=dry_run
        )
        cex = CEXConfig(
            api_key=os.getenv("BINANCE_TESTNET_API_KEY" if sandbox else "BINANCE_API_KEY", ""),
            secret=os.getenv("BINANCE_TESTNET_SECRET" if sandbox else "BINANCE_SECRET", ""),
            sandbox=sandbox, enable_rate_limit=True,
            trading_rules=BinanceTradingRules(
                pair=os.getenv("TRADING_PAIR", "ETH/USDC"),
                min_notional_usd=float(os.getenv("MIN_NOTIONAL_USD", "5.0")),
                lot_size_step=float(os.getenv("LOT_SIZE_STEP", "0.0001")),
                price_tick=float(os.getenv("PRICE_TICK", "0.01"))
            ),
        )
        executor = ExecutorSettings(
            leg1_timeout=Decimal(os.getenv("LEG1_TIMEOUT", "5")),
            leg2_timeout=Decimal(os.getenv("LEG2_TIMEOUT", "60")),
            min_fill_ratio=Decimal(os.getenv("MIN_FILL_RATIO", "0.80")),
            var_confidence=Decimal(os.getenv("VAR_CONFIDENCE", "0.95")),
            vol_per_sqrt_second=Decimal(os.getenv("VOL_PER_SQRT_SECOND", "0.0002")),
            use_dex_first=os.getenv("USE_DEX_FIRST", "true").lower() == "true",
            max_recovery_attempts=int(os.getenv("MAX_RECOVERY_ATTEMPTS", "2")),
            unwind_timeout=Decimal(os.getenv("UNWIND_TIMEOUT", "30")),
            unwind_slippage_bps=Decimal(os.getenv("UNWIND_SLIPPAGE_BPS", "100"))
        )
        risk = RiskConfig(
            max_trade_usd=float(os.getenv("MAX_TRADE_USD", "20.0")),
            max_trade_pct=float(os.getenv("MAX_TRADE_PCT", "0.20")),
            max_position_per_token=float(os.getenv("MAX_POSITION_PER_TOKEN", "30.0")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "1")),
            max_loss_per_trade=float(os.getenv("MAX_LOSS_PER_TRADE", "5.0")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "15.0")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.20")),
            max_trades_per_hour=int(os.getenv("MAX_TRADES_PER_HOUR", "20")),
            consecutive_loss_limit=int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3")),
            max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", "500.0")),
            max_signal_age_seconds=float(os.getenv("MAX_SIGNAL_AGE_SECONDS", "5.0")),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "100.0"))
        )
        return cls(
            mode=mode, dex=dex, cex=cex, executor=executor, risk=risk,
            signal_ttl_seconds=Decimal(os.getenv("SIGNAL_TTL_SECONDS", "5")),
            cooldown_seconds=Decimal(os.getenv("COOLDOWN_SECONDS", "2")),
            kelly_fraction=Decimal(os.getenv("KELLY_FRACTION", "0.25")),
            max_position_usd=Decimal(os.getenv("MAX_POSITION_USD", "10000")),
            cex_taker_bps=Decimal(os.getenv("CEX_TAKER_BPS", "10")),
            dex_swap_bps=Decimal(os.getenv("DEX_SWAP_BPS", "30")),
            gas_cost_usd=Decimal(os.getenv("GAS_COST_USD", "0.10")),
            trading_pair=os.getenv("TRADING_PAIR", "ETH/USDC"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            log_dir=os.getenv("LOG_DIR", "logs"),
            dry_run=dry_run
        )

    @property
    def is_test(self) -> bool:
        return self.mode == OperationMode.TEST

    @property
    def is_production(self) -> bool:
        return self.mode == OperationMode.PRODUCTION

    @property
    def network_preset(self) -> NetworkPreset:
        return ARBITRUM_ONE if self.dex.chain_id == 42161 else ETHEREUM_MAINNET
