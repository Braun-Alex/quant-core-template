"""
System operation mode configuration.

Two modes:
  - test:       Binance Testnet and Ethereum Mainnet (build transactions, DO NOT broadcast)
  - production: Binance Mainnet and Ethereum Mainnet (full execution)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class OperationMode(str, Enum):
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class DEXConfig:
    """DEX configuration."""
    rpc_url: str = ""
    ws_url: str = ""
    # Uniswap V2 router - same on mainnet for both modes
    router_address: str = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    # Known pool addresses (pair_address -> True)
    pool_addresses: list[str] = field(default_factory=list)
    gas_limit_swap: int = 250_000
    gas_limit_approval: int = 60_000
    slippage_tolerance_bps: Decimal = Decimal("50")   # 0.5 %
    deadline_seconds: int = 300
    # In test mode we build but do NOT send the tx
    dry_run: bool = True


@dataclass
class CEXConfig:
    """CEX (Binance) configuration."""
    api_key: str = ""
    secret: str = ""
    sandbox: bool = True   # True → Testnet
    enable_rate_limit: bool = True
    default_type: str = "spot"


@dataclass
class ExecutorSettings:
    """Executor tuning knobs (all overridable via env / config dict)."""
    leg1_timeout: Decimal = Decimal("5")
    leg2_timeout: Decimal = Decimal("60")
    min_fill_ratio: Decimal = Decimal("0.80")
    var_confidence: Decimal = Decimal("0.95")
    vol_per_sqrt_second: Decimal = Decimal("0.0002")
    use_dex_first: bool = True
    max_recovery_attempts: int = 2
    # Unwind settings
    unwind_timeout: Decimal = Decimal("30")
    unwind_slippage_bps: Decimal = Decimal("100")   # 1 % extra tolerance for unwind


@dataclass
class SystemConfig:
    mode: OperationMode = OperationMode.TEST
    dex: DEXConfig = field(default_factory=DEXConfig)
    cex: CEXConfig = field(default_factory=CEXConfig)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    openai_api_key: str = ""

    # Signal generator
    signal_ttl_seconds: Decimal = Decimal("5")
    cooldown_seconds: Decimal = Decimal("2")
    kelly_fraction: Decimal = Decimal("0.25")
    max_position_usd: Decimal = Decimal("10000")
    cex_taker_bps: Decimal = Decimal("10")
    dex_swap_bps: Decimal = Decimal("30")
    gas_cost_usd: Decimal = Decimal("5")

    @classmethod
    def from_env(cls) -> "SystemConfig":
        mode_str = os.getenv("OPERATION_MODE", "test").lower()
        mode = OperationMode.TEST if mode_str == "test" else OperationMode.PRODUCTION

        sandbox = mode == OperationMode.TEST

        dex = DEXConfig(
            rpc_url=os.getenv("ETH_RPC_URL", ""),
            ws_url=os.getenv("ETH_WS_URL", ""),
            router_address=os.getenv(
                "UNISWAP_V2_ROUTER",
                "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
            ),
            pool_addresses=[
                a.strip()
                for a in os.getenv("POOL_ADDRESSES", "").split(",")
                if a.strip()
            ],
            gas_limit_swap=int(os.getenv("GAS_LIMIT_SWAP", "250000")),
            gas_limit_approval=int(os.getenv("GAS_LIMIT_APPROVAL", "60000")),
            slippage_tolerance_bps=Decimal(os.getenv("SLIPPAGE_TOLERANCE_BPS", "50")),
            deadline_seconds=int(os.getenv("TX_DEADLINE_SECONDS", "300")),
            dry_run=(mode == OperationMode.TEST)
        )

        cex = CEXConfig(
            api_key=os.getenv("BINANCE_TESTNET_API_KEY" if sandbox else "BINANCE_API_KEY", ""),
            secret=os.getenv("BINANCE_TESTNET_SECRET" if sandbox else "BINANCE_SECRET", ""),
            sandbox=sandbox,
            enable_rate_limit=True
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

        return cls(
            mode=mode,
            dex=dex,
            cex=cex,
            executor=executor,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            signal_ttl_seconds=Decimal(os.getenv("SIGNAL_TTL_SECONDS", "5")),
            cooldown_seconds=Decimal(os.getenv("COOLDOWN_SECONDS", "2")),
            kelly_fraction=Decimal(os.getenv("KELLY_FRACTION", "0.25")),
            max_position_usd=Decimal(os.getenv("MAX_POSITION_USD", "10000")),
            cex_taker_bps=Decimal(os.getenv("CEX_TAKER_BPS", "10")),
            dex_swap_bps=Decimal(os.getenv("DEX_SWAP_BPS", "30")),
            gas_cost_usd=Decimal(os.getenv("GAS_COST_USD", "5"))
        )

    @property
    def is_test(self) -> bool:
        return self.mode == OperationMode.TEST

    @property
    def is_production(self) -> bool:
        return self.mode == OperationMode.PRODUCTION
