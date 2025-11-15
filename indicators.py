# Custom SMC/ICT indicators and Quality Enhancement indicators for the Gold Trading Bot

import numpy as np
import pandas as pd
import ta


class SMCIndicators:
    """
    Class containing Smart Money Concepts (SMC) and ICT indicators
    """
    
    @staticmethod
    def identify_swing_points(df, window=5):
        """
        Identify swing high and swing low points
        
        Args:
            df: DataFrame with OHLC data
            window: Window size for identifying swings
            
        Returns:
            DataFrame with swing high and swing low columns
        """
        df = df.copy()
        
        # Initialize swing columns
        df['swing_high'] = False
        df['swing_low'] = False
        
        # Identify swing highs
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] > df['high'].iloc[i-window:i]) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+1:i+window+1]):
                df.loc[df.index[i], 'swing_high'] = True
        
        # Identify swing lows
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] < df['low'].iloc[i-window:i]) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+1:i+window+1]):
                df.loc[df.index[i], 'swing_low'] = True
        
        return df
    
    @staticmethod
    def identify_market_structure(df):
        """
        Identify market structure (higher highs, lower lows, etc.)
        
        Args:
            df: DataFrame with swing points identified
            
        Returns:
            DataFrame with market structure columns
        """
        df = df.copy()
        
        # Initialize market structure columns
        df['higher_high'] = False
        df['lower_high'] = False
        df['higher_low'] = False
        df['lower_low'] = False
        df['bos_bullish'] = False  # Break of structure (bullish)
        df['bos_bearish'] = False  # Break of structure (bearish)
        
        # Find consecutive swing highs and lows
        swing_highs = df[df['swing_high']].index
        swing_lows = df[df['swing_low']].index
        
        # Analyze swing highs
        for i in range(1, len(swing_highs)):
            current_idx = swing_highs[i]
            prev_idx = swing_highs[i-1]
            
            if df.loc[current_idx, 'high'] > df.loc[prev_idx, 'high']:
                df.loc[current_idx, 'higher_high'] = True
            else:
                df.loc[current_idx, 'lower_high'] = True
        
        # Analyze swing lows
        for i in range(1, len(swing_lows)):
            current_idx = swing_lows[i]
            prev_idx = swing_lows[i-1]
            
            if df.loc[current_idx, 'low'] > df.loc[prev_idx, 'low']:
                df.loc[current_idx, 'higher_low'] = True
            else:
                df.loc[current_idx, 'lower_low'] = True
        
        # Identify break of structure
        for i in range(1, len(df)):
            # Bullish BOS: price breaks above a significant swing high after making lower lows
            if df.iloc[i-1]['lower_low'] and df.iloc[i]['close'] > df.iloc[i-1]['high']:
                df.loc[df.index[i], 'bos_bullish'] = True
            
            # Bearish BOS: price breaks below a significant swing low after making higher highs
            if df.iloc[i-1]['higher_high'] and df.iloc[i]['close'] < df.iloc[i-1]['low']:
                df.loc[df.index[i], 'bos_bearish'] = True
        
        return df
    
    @staticmethod
    def identify_order_blocks(df, window=5):
        """
        Identify bullish and bearish order blocks
        
        Args:
            df: DataFrame with OHLC data
            window: Window size for order block identification
            
        Returns:
            DataFrame with order block columns
        """
        df = df.copy()
        
        # Initialize order block columns
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        df['bullish_ob_high'] = np.nan
        df['bullish_ob_low'] = np.nan
        df['bearish_ob_high'] = np.nan
        df['bearish_ob_low'] = np.nan
        
        # Identify bullish order blocks (last down candle before a strong move up)
        for i in range(window, len(df) - window):
            # Check for a strong bullish move
            if df.iloc[i+1:i+window+1]['close'].max() > df.iloc[i]['high'] * 1.005:  # 0.5% move up
                # Look for the last bearish candle
                for j in range(i, i-window, -1):
                    if df.iloc[j]['close'] < df.iloc[j]['open']:  # Bearish candle
                        df.loc[df.index[j], 'bullish_ob'] = True
                        df.loc[df.index[j], 'bullish_ob_high'] = df.iloc[j]['high']
                        df.loc[df.index[j], 'bullish_ob_low'] = df.iloc[j]['low']
                        break
        
        # Identify bearish order blocks (last up candle before a strong move down)
        for i in range(window, len(df) - window):
            # Check for a strong bearish move
            if df.iloc[i+1:i+window+1]['close'].min() < df.iloc[i]['low'] * 0.995:  # 0.5% move down
                # Look for the last bullish candle
                for j in range(i, i-window, -1):
                    if df.iloc[j]['close'] > df.iloc[j]['open']:  # Bullish candle
                        df.loc[df.index[j], 'bearish_ob'] = True
                        df.loc[df.index[j], 'bearish_ob_high'] = df.iloc[j]['high']
                        df.loc[df.index[j], 'bearish_ob_low'] = df.iloc[j]['low']
                        break
        
        return df
    
    @staticmethod
    def identify_fair_value_gaps(df):
        """
        Identify fair value gaps (FVGs)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with FVG columns
        """
        df = df.copy()
        
        # Initialize FVG columns
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        df['bullish_fvg_high'] = np.nan
        df['bullish_fvg_low'] = np.nan
        df['bearish_fvg_high'] = np.nan
        df['bearish_fvg_low'] = np.nan
        
        # Identify bullish FVGs (gap up)
        for i in range(2, len(df)):
            if df.iloc[i-2]['high'] < df.iloc[i]['low']:  # Gap up
                df.loc[df.index[i-1], 'bullish_fvg'] = True
                df.loc[df.index[i-1], 'bullish_fvg_high'] = df.iloc[i]['low']
                df.loc[df.index[i-1], 'bullish_fvg_low'] = df.iloc[i-2]['high']
        
        # Identify bearish FVGs (gap down)
        for i in range(2, len(df)):
            if df.iloc[i-2]['low'] > df.iloc[i]['high']:  # Gap down
                df.loc[df.index[i-1], 'bearish_fvg'] = True
                df.loc[df.index[i-1], 'bearish_fvg_high'] = df.iloc[i-2]['low']
                df.loc[df.index[i-1], 'bearish_fvg_low'] = df.iloc[i]['high']
        
        return df
    
    @staticmethod
    def identify_liquidity_levels(df, window=10):
        """
        Identify liquidity levels (clusters of swing highs/lows)
        
        Args:
            df: DataFrame with swing points identified
            window: Window size for liquidity level identification
            
        Returns:
            DataFrame with liquidity level columns
        """
        df = df.copy()
        
        # Initialize liquidity level columns
        df['liquidity_high'] = False
        df['liquidity_low'] = False
        
        # Find clusters of swing highs (liquidity above)
        swing_highs = df[df['swing_high']].index
        for i in range(len(swing_highs) - 1):
            current_high = df.loc[swing_highs[i], 'high']
            next_high = df.loc[swing_highs[i+1], 'high']
            
            # If swing highs are within 0.2% of each other, mark as liquidity level
            if abs(current_high - next_high) / current_high < 0.002:
                df.loc[swing_highs[i], 'liquidity_high'] = True
                df.loc[swing_highs[i+1], 'liquidity_high'] = True
        
        # Find clusters of swing lows (liquidity below)
        swing_lows = df[df['swing_low']].index
        for i in range(len(swing_lows) - 1):
            current_low = df.loc[swing_lows[i], 'low']
            next_low = df.loc[swing_lows[i+1], 'low']
            
            # If swing lows are within 0.2% of each other, mark as liquidity level
            if abs(current_low - next_low) / current_low < 0.002:
                df.loc[swing_lows[i], 'liquidity_low'] = True
                df.loc[swing_lows[i+1], 'liquidity_low'] = True
        
        return df
    
    @staticmethod
    def identify_liquidity_sweep(df):
        """
        Identify liquidity sweeps (price taking out liquidity levels and reversing)
        
        Args:
            df: DataFrame with liquidity levels identified
            
        Returns:
            DataFrame with liquidity sweep columns
        """
        df = df.copy()
        
        # Initialize liquidity sweep columns
        df['sweep_high'] = False
        df['sweep_low'] = False
        
        # Identify high sweeps (price takes out liquidity high and reverses down)
        liquidity_highs = df[df['liquidity_high']].index
        for idx in liquidity_highs:
            i = df.index.get_loc(idx)
            if i + 3 < len(df):  # Ensure we have enough bars after the liquidity level
                liq_price = df.loc[idx, 'high']
                
                # Check if price exceeds the liquidity level
                if df.iloc[i+1:i+3]['high'].max() > liq_price:
                    # Check if price reverses down after taking out liquidity
                    if df.iloc[i+3]['close'] < df.iloc[i+2]['low']:
                        df.loc[df.index[i+2], 'sweep_high'] = True
        
        # Identify low sweeps (price takes out liquidity low and reverses up)
        liquidity_lows = df[df['liquidity_low']].index
        for idx in liquidity_lows:
            i = df.index.get_loc(idx)
            if i + 3 < len(df):  # Ensure we have enough bars after the liquidity level
                liq_price = df.loc[idx, 'low']
                
                # Check if price exceeds the liquidity level
                if df.iloc[i+1:i+3]['low'].min() < liq_price:
                    # Check if price reverses up after taking out liquidity
                    if df.iloc[i+3]['close'] > df.iloc[i+2]['high']:
                        df.loc[df.index[i+2], 'sweep_low'] = True
        
        return df
    
    @staticmethod
    def identify_displacement(df, threshold=0.001):
        """
        Identify displacement moves (strong momentum moves)
        """
        df = df.copy()
        
        # Initialize displacement columns
        df['bullish_displacement'] = False
        df['bearish_displacement'] = False
        df['displacement_strength'] = 0.0
        
        # Calculate candle body percentage
        df['body_pct'] = abs(df['close'] - df['open']) / df['open']
        
        # Identify displacement candles
        for i in range(1, len(df)):
            body_pct = df.iloc[i]['body_pct']
            prev_body_pct = df.iloc[i-1]['body_pct']
            
            # Strong bullish displacement
            if (df.iloc[i]['close'] > df.iloc[i]['open'] and 
                body_pct > threshold and 
                body_pct > prev_body_pct * 2):
                df.loc[df.index[i], 'bullish_displacement'] = True
                df.loc[df.index[i], 'displacement_strength'] = body_pct
            
            # Strong bearish displacement
            elif (df.iloc[i]['close'] < df.iloc[i]['open'] and 
                  body_pct > threshold and 
                  body_pct > prev_body_pct * 2):
                df.loc[df.index[i], 'bearish_displacement'] = True
                df.loc[df.index[i], 'displacement_strength'] = body_pct
        
        return df
    
    @staticmethod
    def identify_change_of_character(df, window=20):
        """
        Identify Change of Character (ChoCh) - more subtle than BOS
        """
        df = df.copy()
        
        # Initialize ChoCh columns
        df['choch_bullish'] = False
        df['choch_bearish'] = False
        
        # Get swing points
        swing_highs = df[df['swing_high'] == True]['high']
        swing_lows = df[df['swing_low'] == True]['low']
        
        # Identify bullish ChoCh (failure to make new low)
        for i in range(window, len(df)):
            recent_lows = swing_lows.iloc[-3:] if len(swing_lows) >= 3 else swing_lows
            if len(recent_lows) >= 2:
                if recent_lows.iloc[-1] > recent_lows.iloc[-2]:  # Higher low
                    df.loc[df.index[i], 'choch_bullish'] = True
        
        # Identify bearish ChoCh (failure to make new high)
        for i in range(window, len(df)):
            recent_highs = swing_highs.iloc[-3:] if len(swing_highs) >= 3 else swing_highs
            if len(recent_highs) >= 2:
                if recent_highs.iloc[-1] < recent_highs.iloc[-2]:  # Lower high
                    df.loc[df.index[i], 'choch_bearish'] = True
        
        return df
    
    @staticmethod
    def apply_all_indicators(df):
        """
        Apply all SMC/ICT indicators to the dataframe
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all indicators applied
        """
        # Apply basic swing points first (required for other indicators)
        df = SMCIndicators.identify_swing_points(df)
        
        # Apply market structure analysis
        df = SMCIndicators.identify_market_structure(df)
        
        # Apply order blocks and fair value gaps
        df = SMCIndicators.identify_order_blocks(df)
        df = SMCIndicators.identify_fair_value_gaps(df)
        
        # Apply liquidity analysis
        df = SMCIndicators.identify_liquidity_levels(df)
        df = SMCIndicators.identify_liquidity_sweep(df)
        
        # Apply enhanced ICT concepts
        df = SMCIndicators.identify_displacement(df)
        df = SMCIndicators.identify_change_of_character(df)
        
        return df


class QualityIndicators:
    """
    Class containing quality enhancement indicators for improved trade accuracy
    """
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """
        Calculate RSI with divergence detection and quality signals
        
        Args:
            df: DataFrame with OHLC data
            period: RSI period (default 14)
            
        Returns:
            DataFrame with RSI indicators
        """
        df = df.copy()
        
        # Calculate RSI using ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # RSI quality signals
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        df['rsi_extreme_oversold'] = df['rsi'] < 20
        df['rsi_extreme_overbought'] = df['rsi'] > 80
        
        # RSI momentum
        df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
        df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1)
        
        # Detect RSI divergence
        df = QualityIndicators._detect_rsi_divergence(df)
        
        # RSI trend strength
        df['rsi_strong_bullish'] = (df['rsi'] > 50) & df['rsi_rising']
        df['rsi_strong_bearish'] = (df['rsi'] < 50) & df['rsi_falling']
        
        return df
    
    @staticmethod
    def _detect_rsi_divergence(df, window=10):
        """
        Detect RSI divergence signals with proper NaN handling
        """
        df['rsi_bullish_div'] = False
        df['rsi_bearish_div'] = False
        
        # Ensure we have valid RSI data before proceeding
        if 'rsi' not in df.columns or df['rsi'].isna().all():
            return df
        
        # Look for divergences over the specified window
        for i in range(window, len(df)):
            price_window = df['close'].iloc[i-window:i+1]
            rsi_window = df['rsi'].iloc[i-window:i+1]
            
            # Check if we have sufficient valid data
            if (len(price_window) > window // 2 and 
                not price_window.isna().all() and 
                not rsi_window.isna().all() and
                not pd.isna(df.iloc[i]['rsi'])):
                
                # Get valid (non-NaN) values only
                valid_price = price_window.dropna()
                valid_rsi = rsi_window.dropna()
                
                # Need at least 3 valid points for meaningful divergence
                if len(valid_price) >= 3 and len(valid_rsi) >= 3:
                    # Bullish divergence: price making lower lows, RSI making higher lows
                    # Compare recent vs older values
                    recent_price_min = valid_price.iloc[-3:].min()
                    older_price_min = valid_price.iloc[:-3].min() if len(valid_price) > 3 else valid_price.iloc[0]
                    recent_rsi_min = valid_rsi.iloc[-3:].min()
                    older_rsi_min = valid_rsi.iloc[:-3].min() if len(valid_rsi) > 3 else valid_rsi.iloc[0]
                    
                    # Bullish divergence: price making lower low, RSI making higher low
                    if (recent_price_min < older_price_min and 
                        recent_rsi_min > older_rsi_min and
                        df.iloc[i]['rsi'] < 35):
                        df.loc[df.index[i], 'rsi_bullish_div'] = True
                    
                    # Bearish divergence: price making higher highs, RSI making lower highs
                    recent_price_max = valid_price.iloc[-3:].max()
                    older_price_max = valid_price.iloc[:-3].max() if len(valid_price) > 3 else valid_price.iloc[0]
                    recent_rsi_max = valid_rsi.iloc[-3:].max()
                    older_rsi_max = valid_rsi.iloc[:-3].max() if len(valid_rsi) > 3 else valid_rsi.iloc[0]
                    
                    if (recent_price_max > older_price_max and 
                        recent_rsi_max < older_rsi_max and
                        df.iloc[i]['rsi'] > 65):
                        df.loc[df.index[i], 'rsi_bearish_div'] = True
        
        return df
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """
        Calculate MACD with signal line and histogram
        
        Args:
            df: DataFrame with OHLC data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD indicators
        """
        df = df.copy()
        
        # Calculate MACD using ta library
        macd_indicator = ta.trend.MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        
        # MACD signals
        df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # MACD momentum
        df['macd_rising'] = df['macd'] > df['macd'].shift(1)
        df['macd_falling'] = df['macd'] < df['macd'].shift(1)
        
        # MACD position relative to zero
        df['macd_above_zero'] = df['macd'] > 0
        df['macd_below_zero'] = df['macd'] < 0
        
        # Strong MACD signals
        df['macd_strong_bullish'] = df['macd_bullish'] & df['macd_above_zero'] & df['macd_rising']
        df['macd_strong_bearish'] = df['macd_bearish'] & df['macd_below_zero'] & df['macd_falling']
        
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df, period=20, std_dev=2):
        """
        Calculate Bollinger Bands with quality signals
        
        Args:
            df: DataFrame with OHLC data
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands indicators
        """
        df = df.copy()
        
        # Calculate Bollinger Bands using ta library
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # Bollinger Band signals
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) < (df['bb_upper'] - df['bb_lower']).rolling(20).mean() * 0.8
        df['bb_expansion'] = (df['bb_upper'] - df['bb_lower']) > (df['bb_upper'] - df['bb_lower']).rolling(20).mean() * 1.2
        
        # Price position relative to bands
        df['bb_above_upper'] = df['close'] > df['bb_upper']
        df['bb_below_lower'] = df['close'] < df['bb_lower']
        df['bb_near_upper'] = (df['close'] > df['bb_middle']) & (df['close'] < df['bb_upper'])
        df['bb_near_lower'] = (df['close'] < df['bb_middle']) & (df['close'] > df['bb_lower'])
        
        # BB reversal signals
        df['bb_reversal_long'] = df['bb_below_lower'] & (df['close'].shift(1) <= df['bb_lower'].shift(1)) & (df['close'] > df['bb_lower'])
        df['bb_reversal_short'] = df['bb_above_upper'] & (df['close'].shift(1) >= df['bb_upper'].shift(1)) & (df['close'] < df['bb_upper'])
        
        return df
    
    @staticmethod
    def calculate_atr(df, period=14):
        """
        Calculate Average True Range for volatility analysis
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            DataFrame with ATR indicators
        """
        df = df.copy()
        
        # Calculate ATR using ta library
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
        
        # ATR-based volatility analysis
        df['atr_ma'] = df['atr'].rolling(period).mean()
        df['volatility_high'] = df['atr'] > df['atr_ma'] * 1.5
        df['volatility_low'] = df['atr'] < df['atr_ma'] * 0.5
        df['volatility_normal'] = ~(df['volatility_high'] | df['volatility_low'])
        
        # ATR trend
        df['atr_rising'] = df['atr'] > df['atr'].shift(1)
        df['atr_falling'] = df['atr'] < df['atr'].shift(1)
        
        return df
    
    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculate ADX for trend strength analysis
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            DataFrame with ADX indicators
        """
        df = df.copy()
        
        # Calculate ADX using ta library
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period)
        df['adx'] = adx_indicator.adx()
        df['plus_di'] = adx_indicator.adx_pos()
        df['minus_di'] = adx_indicator.adx_neg()
        
        # ADX trend strength
        df['trend_weak'] = df['adx'] < 25
        df['trend_strong'] = df['adx'] > 40
        df['trend_very_strong'] = df['adx'] > 60
        
        # Trend direction
        df['trend_bullish'] = (df['plus_di'] > df['minus_di']) & df['trend_strong']
        df['trend_bearish'] = (df['minus_di'] > df['plus_di']) & df['trend_strong']
        
        return df
    
    @staticmethod
    def calculate_stochastic(df, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D smoothing period
            
        Returns:
            DataFrame with Stochastic indicators
        """
        df = df.copy()
        
        # Calculate Stochastic using ta library
        stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()
        
        # Stochastic signals
        df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        
        # Stochastic crossovers
        df['stoch_bullish'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['stoch_bearish'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        return df
    
    @staticmethod
    def calculate_volume_indicators(df):
        """
        Calculate volume-based indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        df = df.copy()
        
        if 'volume' not in df.columns:
            # If no volume data, create dummy volume for calculations
            df['volume'] = 1000
        
        # Volume moving averages
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # Volume signals
        df['high_volume'] = df['volume'] > df['volume_ma_20'] * 1.5
        df['low_volume'] = df['volume'] < df['volume_ma_20'] * 0.5
        
        # Volume trend
        df['volume_rising'] = df['volume'] > df['volume'].shift(1)
        df['volume_falling'] = df['volume'] < df['volume'].shift(1)
        
        # Volume with price confirmation
        df['volume_bullish_confirm'] = df['high_volume'] & (df['close'] > df['open'])
        df['volume_bearish_confirm'] = df['high_volume'] & (df['close'] < df['open'])
        
        return df
    
    @staticmethod
    def calculate_quality_score(df):
        """
        Calculate overall quality score for trade signals
        
        Args:
            df: DataFrame with all quality indicators
            
        Returns:
            DataFrame with quality score
        """
        df = df.copy()
        
        # Initialize quality score
        df['quality_score'] = 0.0
        
        # RSI contribution (weight: 25%)
        rsi_score = np.where(df['rsi_strong_bullish'], 1.0, 0.0)
        rsi_score += np.where(df['rsi_strong_bearish'], 1.0, 0.0)
        rsi_score += np.where(df['rsi_bullish_div'], 2.0, 0.0)  # Higher weight for divergence
        rsi_score += np.where(df['rsi_bearish_div'], 2.0, 0.0)
        df['quality_score'] += (rsi_score / 4) * 0.25
        
        # MACD contribution (weight: 20%)
        macd_score = np.where(df['macd_strong_bullish'], 2.0, 0.0)
        macd_score += np.where(df['macd_strong_bearish'], 2.0, 0.0)
        macd_score += np.where(df['macd_bullish'], 1.0, 0.0)
        macd_score += np.where(df['macd_bearish'], 1.0, 0.0)
        df['quality_score'] += (macd_score / 4) * 0.20
        
        # ADX contribution (weight: 20%)
        adx_score = np.where(df['trend_strong'], 1.0, 0.0)
        adx_score += np.where(df['trend_very_strong'], 2.0, 0.0)
        adx_score += np.where(df['trend_bullish'], 1.0, 0.0)
        adx_score += np.where(df['trend_bearish'], 1.0, 0.0)
        df['quality_score'] += (adx_score / 4) * 0.20
        
        # Bollinger Bands contribution (weight: 15%)
        bb_score = np.where(df['bb_reversal_long'], 2.0, 0.0)
        bb_score += np.where(df['bb_reversal_short'], 2.0, 0.0)
        bb_score += np.where(df['bb_squeeze'], 1.0, 0.0)  # Squeeze often leads to breakouts
        df['quality_score'] += (bb_score / 3) * 0.15
        
        # Volume contribution (weight: 10%)
        volume_score = np.where(df['volume_bullish_confirm'], 1.0, 0.0)
        volume_score += np.where(df['volume_bearish_confirm'], 1.0, 0.0)
        volume_score += np.where(df['high_volume'], 1.0, 0.0)
        df['quality_score'] += (volume_score / 3) * 0.10
        
        # Volatility contribution (weight: 10%)
        vol_score = np.where(df['volatility_normal'], 1.0, 0.0)  # Prefer normal volatility
        vol_score += np.where(df['volatility_low'], 0.5, 0.0)  # Low volatility is okay
        vol_score += np.where(df['volatility_high'], -0.5, 0.0)  # Penalize high volatility
        df['quality_score'] += vol_score * 0.10
        
        # Normalize quality score to 0-1 range
        df['quality_score'] = np.clip(df['quality_score'], 0, 1)
        
        # Quality categories
        df['quality_excellent'] = df['quality_score'] > 0.8
        df['quality_good'] = (df['quality_score'] > 0.6) & (df['quality_score'] <= 0.8)
        df['quality_fair'] = (df['quality_score'] > 0.4) & (df['quality_score'] <= 0.6)
        df['quality_poor'] = df['quality_score'] <= 0.4
        
        return df
    
    @staticmethod
    def apply_all_quality_indicators(df):
        """
        Apply all quality indicators to the dataframe
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all quality indicators applied
        """
        # Apply all quality indicators
        df = QualityIndicators.calculate_rsi(df)
        df = QualityIndicators.calculate_macd(df)
        df = QualityIndicators.calculate_bollinger_bands(df)
        df = QualityIndicators.calculate_atr(df)
        df = QualityIndicators.calculate_adx(df)
        df = QualityIndicators.calculate_stochastic(df)
        df = QualityIndicators.calculate_volume_indicators(df)
        
        # Calculate overall quality score
        df = QualityIndicators.calculate_quality_score(df)
        
        return df


def apply_all_indicators(df):
    """
    Apply both SMC/ICT indicators and Quality indicators
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with all indicators applied
    """
    # Apply SMC/ICT indicators
    df = SMCIndicators.apply_all_indicators(df)
    
    # Apply Quality indicators
    df = QualityIndicators.apply_all_quality_indicators(df)
    
    return df