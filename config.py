# Configuration settings for the Multi-Symbol Trading Bot

import os
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    # Continue with default values

# Broker API Configuration (Exness)
API_KEY = os.getenv('EXNESS_API_KEY', '')
API_SECRET = os.getenv('EXNESS_API_SECRET', '')
ACCOUNT_ID = os.getenv('EXNESS_ACCOUNT_ID', '')

# MT5 Account Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')

# Multi-Symbol Trading Configuration
SYMBOLS = {
    'XAUUSDm': {  # Gold
        'enabled': False,
        'risk_percent': 1.0,
        'tp_ratio': 2.5,  # Reduced from 3.0 for better execution
        'max_trades': 1,
        'min_rr_ratio': 2.0,  # Higher minimum for swing trading
        'fixed_lot_size': 0.01,  # Set to specific value (e.g., 0.01) to use fixed lots, None for dynamic
        'trailing_settings': {
            'start_ratio': 1.5,  # Start trailing later for swing trading
            'trail_step': 0.8,   # Larger trail steps for swing trading
            'trail_tp': True,    # Enable TP trailing
            'trail_sl': True,    # Enable SL trailing
            'breakeven_ratio': 1.2,  # Move to breakeven later for swing trading
        },
        'volatility_adj': True,  # Adjust position size based on volatility
        'correlation_filter': True,  # Enable correlation filtering
        'scalping_mode': False,  # Disable scalping optimizations for swing trading
    },
    'BTCUSDm': {  # Bitcoin
        'enabled': True,
        'risk_percent': 0.5,  # Reduced risk for BTC
        'tp_ratio': 2.5,  # Higher targets for swing trading
        'max_trades': 1,
        'min_rr_ratio': 2.0,  # Higher minimum for swing trading
        'fixed_lot_size': 0.01,  # Set to specific value (e.g., 0.01) to use fixed lots, None for dynamic
        'trailing_settings': {
            'start_ratio': 1.5,  # Start trailing later for swing trading
            'trail_step': 0.8,  # Larger steps for swing trading
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,  # Later breakeven for swing trading
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,  # Disable scalping optimizations for swing trading
    },
    'USOILm': {  # US Oil
        'enabled': False,
        'risk_percent': 1.2,
        'tp_ratio': 2.0,  # Reduced for better execution
        'max_trades': 1,
        'min_rr_ratio': 1.2,  # Reduced from 1.3
        'trailing_settings': {
            'start_ratio': 1.0,
            'trail_step': 0.4,
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 0.7,
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,
    },
    'ETHUSDm': {  # Ethereum
        'enabled': True,
        'risk_percent': 0.5,
        'tp_ratio': 2.5,
        'max_trades': 2,
        'min_rr_ratio': 2.0,
        'fixed_lot_size': 0.1,
        'trailing_settings': {
            'start_ratio': 1.5,
            'trail_step': 0.8,
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,
        },
    },
    'EURUSDm': {  # Euro
        'enabled': False,
        'risk_percent': 0.5,
        'tp_ratio': 2.5,  # Higher targets for swing trading
        'max_trades': 1,
        'min_rr_ratio': 2.0,  # Higher minimum for swing trading
        'fixed_lot_size': 0.01,  # Set to specific value (e.g., 0.01) to use fixed lots, None for dynamic
        'trailing_settings': {
            'start_ratio': 1.5,  # Start trailing later for swing trading
            'trail_step': 0.8,   # Larger steps for swing trading
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,  # Later breakeven for swing trading
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,  # Disable scalping optimizations for swing trading
    },
    'XAGUSDm': {  # Silver
        'enabled': False,
        'risk_percent': 0.5,
        'tp_ratio': 2.5,
        'max_trades': 1,
        'min_rr_ratio': 2.0,
        'fixed_lot_size': 0.01,
        'trailing_settings': {
            'start_ratio': 1.5,
            'trail_step': 0.8,
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,
    },
    'GBPJPYm': {  # British Pound
        'enabled': False,
        'risk_percent': 0.5,
        'tp_ratio': 2.5,
        'max_trades': 1,
        'min_rr_ratio': 2.0,
        'fixed_lot_size': 0.01,
        'trailing_settings': {
            'start_ratio': 1.5,
            'trail_step': 0.8,
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,
    },
    'GBPUSDm': {  # British Pound
        'enabled': False,
        'risk_percent': 0.5,
        'tp_ratio': 2.5,
        'max_trades': 1,
        'min_rr_ratio': 2.0,
        'fixed_lot_size': 0.01,
        'trailing_settings': {
            'start_ratio': 1.5,
            'trail_step': 0.8,
            'trail_tp': True,
            'trail_sl': True,
            'breakeven_ratio': 1.2,
        },
        'volatility_adj': True,
        'correlation_filter': True,
        'scalping_mode': False,
    },
}

# Enhanced Risk Management
RISK_MANAGEMENT = {
    'max_total_risk': 2.0,  # Reduced maximum total risk across all positions
    'max_correlated_risk': 1.5,  # Maximum risk for correlated instruments
    'correlation_threshold': 0.7,  # Correlation threshold for filtering
    'volatility_lookback': 20,  # ATR periods for volatility calculation
    'dynamic_sizing': True,  # Enable dynamic position sizing
    'max_drawdown_stop': 8.0,  # Reduced stop trading at 8% drawdown
    'daily_loss_limit': 3.0,  # Reduced daily loss limit percentage
    'consecutive_loss_limit': 3,  # Stop after 3 consecutive losses
    'max_daily_trades': 10,  # Maximum trades per day
    'min_quality_score': 0.4,  # Minimum quality score for trade execution (0.0-1.0)
}

# Advanced Trailing Configuration
TRAILING_SETTINGS = {
    'algorithm': 'enhanced_atr',  # 'simple', 'atr', 'enhanced_atr', 'parabolic'
    'atr_multiplier': 2.0,
    'min_trail_distance': 0.001,  # Minimum trail distance
    'trail_frequency': 30,  # Trail update frequency in seconds
    'use_swing_levels': True,  # Use swing highs/lows for trailing
    'consolidation_filter': True,  # Avoid trailing in consolidation
}

# Legacy single symbol support (for backward compatibility)
SYMBOL = 'XAUUSDm'  # Primary symbol
RISK_PERCENT = 1.0  # Default risk per trade
TP_RATIO = 2.0      # Default take profit ratio
SL_PADDING = 5      # Additional pips for stop loss

# Multi-Timeframe Configuration
TIMEFRAMES = {
    'primary': '5m',      # Primary timeframe for signals
    'confirmation': ['1h', '4h'],  # Higher timeframes for confirmation
    'precision': ['5m', '1m'],     # Lower timeframes for precise entry
}

# ICT/SMC Strategy Configuration
ICT_SETTINGS = {
    'session_times': {
        'london': {'start': '08:00', 'end': '17:00'},
        'new_york': {'start': '13:00', 'end': '22:00'},
        'asian': {'start': '00:00', 'end': '09:00'}
    },
    'power_of_three': {
        'accumulation': '20:00-02:00',  # Asian session
        'manipulation': '02:00-05:00',  # Pre-London
        'distribution': '08:00-17:00'   # London/NY overlap
    },
    'order_block_lookback': 20,
    'liquidity_sweep_tolerance': 0.0001,  # 1 pip for gold
    'fvg_min_size': 0.0005,  # Minimum FVG size (0.5 pips)
    'premium_discount_levels': [0.618, 0.705, 0.79, 0.886],  # Fibonacci levels
}

# DeepSeek AI Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-chat'  # DeepSeek's main chat model
DEEPSEEK_MAX_TOKENS = 500
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_ENABLED = False  # Set to False to temporarily disable DeepSeek AI (e.g., during payment issues)

# AI Analysis Configuration
AI_SETTINGS = {
    'enable_deepseek': DEEPSEEK_ENABLED,  # Enable DeepSeek AI integration (respects DEEPSEEK_ENABLED setting)
    'confidence_threshold': 0.6,  # Higher threshold for swing trading quality
    'market_condition_weight': 0.4,  # Balanced for swing trading
    'technical_analysis_weight': 0.5,  # Balanced technical weight for swing trading
    'sentiment_analysis_weight': 0.1,  # Keep sentiment weight low
    'min_confluence_factors': 2,  # Require more confluence factors for swing trading
    'multi_symbol_analysis': True,  # Enable cross-symbol analysis
    'scalping_mode': False,  # Disable scalping optimizations for swing trading
    'accept_lower_confidence': False,  # Require higher confidence for swing trading
}

# Webhook Configuration
WEBHOOK_PORT = 5000
WEBHOOK_HOST = '0.0.0.0'
WEBHOOK_PATH = '/webhook'

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Database Configuration
DB_PATH = 'trades.db'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'

# Enhanced Scheduler Configuration (Optimized for Swing Trading)
SCHEDULER_INTERVALS = {
    'signal_check': 300,    # Less frequent signal checks for swing trading (every 5 minutes)
    'trade_monitor': 120,   # Patient monitoring for swing trading (every 2 minutes)
    'market_analysis': 180, # Slower market analysis updates for swing trading
    'correlation_update': 600,  # Update correlations every 10 minutes
    'risk_check': 180,      # Less frequent risk checks for swing trading (every 3 minutes)
}

# Trade Quality Filters
TRADE_QUALITY = {
    'min_atr_multiplier': 1.5,      # Minimum ATR multiple for stop loss
    'max_spread_multiplier': 2.0,   # Max spread as multiple of ATR
    'min_volume_filter': True,      # Filter low volume periods
    'trend_confirmation': True,     # Require trend confirmation
    'session_filter': False,         # Only trade during active sessions
    'news_avoidance_hours': 1,      # Avoid trading 1 hour around major news
}