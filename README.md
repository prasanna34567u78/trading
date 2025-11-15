# Simple Bot Trading System

A multi-symbol trading bot using MT5Executor with advanced SMC/ICT strategies and AI-enhanced trade validation.

## Features

- **MT5Executor Integration**: Uses enhanced MT5Executor for robust connection management
- **Multi-Symbol Support**: Trade multiple symbols simultaneously 
- **SMC/ICT Strategies**: Smart Money Concepts and ICT trading strategies
- **AI Trade Validation**: Integration with AI models and DeepSeek API for enhanced decision-making
- **Quality Filtering**: Advanced technical indicator confluence analysis
- **Trailing Stops**: Multiple trailing stop algorithms (ATR, Parabolic, Swing levels)
- **Risk Management**: Automatic position sizing and risk control with correlation analysis

## Requirements

- Python 3.8 or higher
- MetaTrader 5 terminal
- Valid MT5 account with API access

## Installation

### Automatic Installation:

1. Run `install.bat` to set up the environment automatically
2. The script will create a virtual environment and install all dependencies
3. Add your `config.py` file with MT5 credentials

### Manual Installation:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `config.py` file in the main directory with the following structure:

```python
# MT5 Account Settings
MT5_LOGIN = 'your_login_number'
MT5_PASSWORD = 'your_password'
MT5_SERVER = 'your_broker_server'

# Symbol Configuration
SYMBOLS = {
    'XAUUSD': {
        'enabled': True,
        'risk_percent': 1.0,  # Risk per trade (% of account)
        'tp_ratio': 2.0,      # Take profit ratio
        'trailing_settings': {
            'start_ratio': 1.0,
            'trail_step': 0.5,
            'breakeven_ratio': 0.8
        }
    },
    # Add more symbols as needed
}

# Trading Parameters
RISK_PERCENT = 1.0
TP_RATIO = 2.0

# Trailing settings
TRAILING_SETTINGS = {
    'algorithm': 'enhanced_atr',  # Options: 'simple', 'atr', 'enhanced_atr', 'parabolic'
    'atr_multiplier': 2.0,
    'min_trail_distance': 0.001,
    'use_swing_levels': True
}

# Risk management settings
RISK_MANAGEMENT = {
    'min_quality_score': 0.4
}

# DeepSeek AI settings (optional)
DEEPSEEK_API_KEY = ''  # Your DeepSeek API key
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-chat'
DEEPSEEK_MAX_TOKENS = 1000
DEEPSEEK_TEMPERATURE = 0.7
AI_SETTINGS = {
    'enable_deepseek': False  # Set to True to enable AI features
}
```

## Usage

1. **Start MetaTrader 5**: Ensure your MT5 terminal is running and logged in to your trading account
2. **Enable Expert Advisors**: In MT5, go to Tools → Options → Expert Advisors → Check "Allow automated trading"
3. **Run the bot**:
   ```bash
   python simple_bot.py
   ```

## Important Notes

### MT5 Setup Requirements:
- Ensure "Allow automated trading" is enabled in MT5 settings
- Make sure MT5 terminal is running and logged in before starting the bot
- Run both the bot and MT5 terminal as Administrator to avoid permission issues

### Connection Management:
- The bot uses MT5Executor for robust connection handling with automatic reconnection
- Each symbol has its own executor instance for better isolation

### Multi-Instance Setup:
- For running multiple bot instances, you'll need separate MT5 terminals
- Each bot instance should connect to a different MT5 terminal

## Files Structure

- `simple_bot.py`: Main bot implementation using MT5Executor
- `mt5_executor.py`: Enhanced MT5 connection management with auto-reconnection
- `strategy.py`: SMC/ICT trading strategies and AI integration
- `ai_analyzer.py`: AI model and DeepSeek integration
- `indicators.py`: Technical indicators and SMC/ICT indicators
- `requirements.txt`: Python package dependencies
- `install.bat`: Installation script
- `config.py`: Configuration file (not included, must be created by user)

## Troubleshooting

### IPC Timeout Errors:
- Ensure MT5 terminal is fully loaded before starting the bot
- Close other applications that might interfere
- Run both MT5 and the bot as Administrator

### Connection Issues:
- Verify your MT5 credentials in config.py
- Ensure you're using the correct server name
- Check that your MT5 account supports API trading

### Permission Issues:
- Run the bot from a folder outside Program Files if experiencing permission issues
- Ensure proper antivirus/firewall settings allow the connection

## Security

- Never commit your `config.py` file with credentials to version control
- Keep your MT5 credentials secure
- Consider using environment variables for sensitive data

## License

This software is provided as-is for educational purposes. Use at your own risk. Trading involves substantial risk and may not be suitable for all investors."# trading"  
