# Trading strategy implementation based on SMC/ICT concepts

import pandas as pd
import numpy as np
from indicators import SMCIndicators, QualityIndicators, apply_all_indicators
import config
import logging
from ai_analyzer import AITradeAnalyzer

logger = logging.getLogger('gold_trading_bot')


class SMCStrategy:
    """
    Smart Money Concepts (SMC) and ICT trading strategy with single trade management
    """
    
    def __init__(self, mt5_executor=None, risk_percent=config.RISK_PERCENT, tp_ratio=config.TP_RATIO):
        self.risk_percent = risk_percent
        self.tp_ratio = tp_ratio
        self.indicators = SMCIndicators()
        # Enhanced single trade management with trailing
        self.current_trade = None
        self.trailing_activated = False
        self.initial_stop_distance = None
        self.initial_tp_distance = None
        self.last_trail_price = None
        self.breakeven_activated = False
        
        # Multi-symbol support
        self.symbol_trades = {}  # Track trades per symbol
        self.trailing_states = {}  # Track trailing states per symbol
        
        # Advanced trailing settings
        self.trailing_algorithm = config.TRAILING_SETTINGS.get('algorithm', 'enhanced_atr')
        self.atr_multiplier = config.TRAILING_SETTINGS.get('atr_multiplier', 2.0)
        self.min_trail_distance = config.TRAILING_SETTINGS.get('min_trail_distance', 0.001)
        self.use_swing_levels = config.TRAILING_SETTINGS.get('use_swing_levels', True)
        
        # Initialize AI analyzer with MT5 executor for historical data training
        self.ai_analyzer = AITradeAnalyzer(mt5_executor)
        
        # Store MT5 executor reference
        self.mt5_executor = mt5_executor
        
        # Quality settings
        self.min_quality_score = config.RISK_MANAGEMENT.get('min_quality_score', 0.4)
        self.excellent_quality_threshold = 0.8
        self.good_quality_threshold = 0.6
    
    def analyze_market(self, df):
        """
        Analyze market data and apply SMC/ICT indicators with quality enhancement
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all indicators applied including quality score
        """
        # Apply all indicators (SMC/ICT + Quality Indicators)
        df = apply_all_indicators(df)
        
        # Add legacy ATR calculation for compatibility
        if 'atr' not in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals based on SMC/ICT concepts, only if no active trade
        
        Args:
            df: DataFrame with indicators applied
            
        Returns:
            DataFrame with signals added
        """
        df = df.copy()
        
        # Initialize signal columns
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['risk_reward'] = np.nan
        df['signal_type'] = ''
        df['ai_confidence'] = 0.0
        
        # Only generate signals if we don't have an active trade
        if self.current_trade is None:
            # Strategy 1: Liquidity Sweep + Break of Structure (BOS)
            self._apply_liquidity_sweep_bos_strategy(df)
            
            # Strategy 2: Return to Order Block or Fair Value Gap
            self._apply_ob_fvg_return_strategy(df)
            
            # Apply quality filtering first
            self._apply_quality_filter(df)
            
            # Validate signals with AI
            latest = df.iloc[-1]
            # Only log if we have a signal (avoid logging nan values)
            if latest['signal'] != 0:
                logger.info(f"Generated signal: {latest['signal']}, Entry: {latest['entry_price']}, "
                            f"SL: {latest['stop_loss']}, TP: {latest['take_profit']}, "
                            f"Quality Score: {latest.get('quality_score', 'N/A'):.3f}")
            if latest['signal'] != 0:
                # Get AI validation
                validation = self.ai_analyzer.enhanced_validate_trade(df, latest['signal'])
                logger.info(f"AI validation result: {validation}")
                if not validation['valid']:
                    # Clear invalid signal
                    df.loc[df.index[-1], 'signal'] = 0
                    logger.info(f"AI rejected trade signal (confidence: {validation['confidence']:.2f})")
                else:
                    # Store AI confidence
                    df.loc[df.index[-1], 'ai_confidence'] = validation['confidence']
                    
                    # Adjust take profit based on market conditions and quality
                    quality_score = latest.get('quality_score', 0.5)
                    if validation['market_conditions']['volatility_state'] == 'high':
                        # Use tighter take profit in high volatility
                        risk = abs(latest['entry_price'] - latest['stop_loss'])
                        new_tp = latest['entry_price'] + (risk * self.tp_ratio * 0.8) if latest['signal'] > 0 else \
                                latest['entry_price'] - (risk * self.tp_ratio * 0.8)
                        df.loc[df.index[-1], 'take_profit'] = new_tp
                    elif quality_score > self.excellent_quality_threshold:
                        # Use wider take profit for high quality signals
                        risk = abs(latest['entry_price'] - latest['stop_loss'])
                        new_tp = latest['entry_price'] + (risk * self.tp_ratio * 1.2) if latest['signal'] > 0 else \
                                latest['entry_price'] - (risk * self.tp_ratio * 1.2)
                        df.loc[df.index[-1], 'take_profit'] = new_tp
                        logger.info(f"Extended TP for high-quality signal: {new_tp}")
                    
                    # Log AI insights with quality score
                    logger.info(f"AI validated trade with {validation['confidence']:.2f} confidence, "
                               f"Quality Score: {quality_score:.3f}")
                    logger.info(f"Market conditions: {validation['market_conditions']}")
        
        return df
    
    def generate_signals_with_confluence(self, df, confluence, mtf_analysis=None):
        """
        Generate trading signals enhanced with multi-timeframe confluence
        
        Args:
            df: DataFrame with indicators applied
            confluence: Multi-timeframe confluence analysis
            mtf_analysis: Multi-timeframe analysis data for DeepSeek AI integration
            
        Returns:  
            DataFrame with enhanced signals
        """
        # Start with regular signal generation
        df = self.generate_signals(df)
        
        logger.info(f"Confluence analysis - Signal: {confluence['signal']}, "
                   f"Confidence: {confluence['confidence']:.2f}, "
                   f"Bullish votes: {confluence['bullish_votes']}, "
                   f"Bearish votes: {confluence['bearish_votes']}")
        
        # Enhance signals with confluence
        if confluence['signal'] != 0:
            latest_idx = df.index[-1]
            
            # CRITICAL: Check if we have valid market data before creating confluence signals
            current_price = df.loc[latest_idx, 'close']
            atr = df.loc[latest_idx, 'atr'] if 'atr' in df.columns else None
            
            # Validate that we have proper market data
            if pd.isna(current_price) or pd.isna(atr) or atr is None or atr <= 0:
                logger.warning("Invalid market data - skipping confluence signal creation")
                logger.warning(f"Current price: {current_price}, ATR: {atr}")
                return df
            
            # Additional validation: Check if basic indicators are working
            latest = df.iloc[-1]
            
            # Check if we have essential indicator columns
            required_indicators = ['bos_bullish', 'bos_bearish', 'bullish_ob', 'bearish_ob']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            
            if missing_indicators:
                logger.warning(f"Missing essential indicators: {missing_indicators} - skipping confluence signal")
                return df
            logger.info(f"Latest indicators - BOS Bullish: {latest}")
            # Check if we have valid indicator values (not all NaN)
            has_valid_structure = any([
                not pd.isna(latest.get(ind, np.nan)) and latest.get(ind, False) 
                for ind in required_indicators
            ])
            
            logger.info(f"Valid market structure detected: {has_valid_structure}")
            if not has_valid_structure:
                logger.warning("No valid market structure detected - skipping confluence signal")
                logger.warning(f"Indicator values: BOS Bull: {latest.get('bos_bullish')}, "
                             f"BOS Bear: {latest.get('bos_bearish')}, "
                             f"Bull OB: {latest.get('bullish_ob')}, "
                             f"Bear OB: {latest.get('bearish_ob')}")
                return df
            
            # Only proceed if we have confluence signal AND valid market conditions
            if confluence['confidence'] > 0.6:
                # Adjust signal based on confluence
                if confluence['signal'] == 1 and df.loc[latest_idx, 'signal'] != -1:
                    df.loc[latest_idx, 'signal'] = 1
                    df.loc[latest_idx, 'signal_type'] = 'mtf_confluence_bullish'
                    df.loc[latest_idx, 'ai_confidence'] = confluence['confidence']
                    
                    # Calculate entry levels (using already validated data)
                    entry_price = current_price
                    atr_risk = atr * 2.0  # Use 2x ATR for stop loss
                    stop_loss = entry_price - atr_risk
                    take_profit = entry_price + (atr_risk * self.tp_ratio)
                    
                    df.loc[latest_idx, 'entry_price'] = entry_price
                    df.loc[latest_idx, 'stop_loss'] = stop_loss
                    df.loc[latest_idx, 'take_profit'] = take_profit
                    df.loc[latest_idx, 'risk_reward'] = self.tp_ratio
                    
                    logger.info(f"Multi-timeframe bullish confluence signal created - Entry: {entry_price:.5f}, "
                              f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                
                elif confluence['signal'] == -1 and df.loc[latest_idx, 'signal'] != 1:
                    df.loc[latest_idx, 'signal'] = -1
                    df.loc[latest_idx, 'signal_type'] = 'mtf_confluence_bearish'
                    df.loc[latest_idx, 'ai_confidence'] = confluence['confidence']
                    
                    # Calculate entry levels for bearish signal
                    entry_price = current_price
                    atr_risk = atr * 2.0
                    stop_loss = entry_price + atr_risk
                    take_profit = entry_price - (atr_risk * self.tp_ratio)
                    
                    df.loc[latest_idx, 'entry_price'] = entry_price
                    df.loc[latest_idx, 'stop_loss'] = stop_loss
                    df.loc[latest_idx, 'take_profit'] = take_profit
                    df.loc[latest_idx, 'risk_reward'] = self.tp_ratio
                    
                    logger.info(f"Multi-timeframe bearish confluence signal created - Entry: {entry_price:.5f}, "
                              f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return df
    
    def generate_swing_signals_with_confluence(self, df, confluence, mtf_analysis=None,symbol='XAUUSDm'):
        """
        Generate swing trading signals enhanced with multi-timeframe confluence
        Optimized for 4h, 1h, and 15m timeframes with larger profit targets
        
        Args:
            df: DataFrame with indicators applied (15m data)
            confluence: Multi-timeframe confluence analysis
            mtf_analysis: Multi-timeframe analysis data
            
        Returns:  
            DataFrame with enhanced swing trading signals
        """
        # Start with regular signal generation but with swing trading parameters
        df = self.generate_signals(df)
        
        logger.info(f"Swing trading confluence - Signal: {confluence['signal']}, "
                   f"Confidence: {confluence['confidence']:.2f}")
        
        # Enhance signals with swing trading confluence (more conservative)
        if confluence['signal'] != 0:
            latest_idx = df.index[-1]
            
            # Get market data for swing trading
            current_price = df.loc[latest_idx, 'close']
            atr = df.loc[latest_idx, 'atr'] if 'atr' in df.columns else None
            
            # Validate data
            if pd.isna(current_price) or pd.isna(atr) or atr is None or atr <= 0:
                logger.warning("Invalid market data for swing trading - skipping signal")
                # Clear any NaN values before returning
                self._clear_invalid_signal_values(df, latest_idx)
                return df
            
            # Volatility filter - avoid trading during extreme volatility (especially for crypto)
            if len(df) >= 20:
                recent_atr = df['atr'].rolling(5).mean().iloc[-1]
                avg_atr = df['atr'].rolling(20).mean().iloc[-1]
                
                if recent_atr > 0 and avg_atr > 0:
                    volatility_ratio = recent_atr / avg_atr
                    
                    # Skip trading during extreme volatility spikes
                    max_volatility_ratio = 2.5 if 'BTC' in symbol else 2.0
                    if volatility_ratio > max_volatility_ratio:
                        logger.warning(f"Extreme volatility detected: {volatility_ratio:.2f}x average - skipping for safety")
                        # Clear any NaN values before returning
                        self._clear_invalid_signal_values(df, latest_idx)
                        return df
            
            # Higher confidence threshold for swing trading (more conservative)
            # Require very high confidence AND additional validations
            min_confidence = 0.75 if 'BTC' in symbol else 0.65  # Even higher for volatile crypto
            
            if confluence['confidence'] > min_confidence:
                # Additional confluence validation - require strong alignment
                bullish_votes = confluence.get('bullish_votes', 0)
                bearish_votes = confluence.get('bearish_votes', 0)
                total_timeframes = len(mtf_analysis) if mtf_analysis else 3
                
                # Require majority alignment across timeframes
                required_alignment = max(2, int(total_timeframes * 0.67))  # At least 67% alignment
                
                if confluence['signal'] == 1 and bullish_votes < required_alignment:
                    logger.info(f"Insufficient bullish alignment: {bullish_votes}/{total_timeframes} - need {required_alignment}")
                    # Clear any NaN values before returning
                    self._clear_invalid_signal_values(df, latest_idx)
                    return df
                elif confluence['signal'] == -1 and bearish_votes < required_alignment:
                    logger.info(f"Insufficient bearish alignment: {bearish_votes}/{total_timeframes} - need {required_alignment}")
                    # Clear any NaN values before returning
                    self._clear_invalid_signal_values(df, latest_idx)
                    return df
                # Calculate ATR-based minimum distances first
                min_atr_multiple = 1.5 if 'BTC' in symbol else 2.0 if 'XAU' in symbol or 'GOLD' in symbol.upper() else 1.8
                min_swing_distance = atr * min_atr_multiple
                
                # Determine symbol type and calculate adaptive pip targets based on ATR
                if 'BTC' in symbol:
                    # Bitcoin swing trading - much larger distances for volatility
                    pip_value = 1.0      # $1 movements for BTC
                    # Ensure minimum ATR distance but cap at reasonable maximum
                    min_stop_distance = max(min_swing_distance, 320.0)  # At least $320 or ATR minimum
                    max_stop_distance = 2000.0  # Cap at $2000 for risk management
                    stop_pips = min(min_stop_distance, max_stop_distance)
                    target_pips = stop_pips * 2.5  # 2.5:1 ratio for swing
                    
                elif 'USD' in symbol and symbol.endswith('m'):
                    # Forex pairs swing trading
                    pip_value = 0.0001   # Standard pip size
                    # Calculate minimum stop in pips based on ATR with reasonable cap
                    min_stop_pips = max(min_swing_distance / pip_value, 40)  # At least 40 pips or ATR minimum
                    max_stop_pips = 200  # Cap at 200 pips for risk management
                    stop_pips = min(min_stop_pips, max_stop_pips)
                    target_pips = stop_pips * 2.5  # 2.5:1 ratio for swing trading
                    
                elif 'XAU' in symbol or 'GOLD' in symbol.upper():
                    # Gold swing trading
                    pip_value = 0.01     # $0.01 movements
                    # Calculate minimum stop in gold points based on ATR with reasonable cap
                    min_stop_points = max(min_swing_distance / pip_value, 32)  # At least 32 points or ATR minimum
                    max_stop_points = 150  # Cap at 150 points ($1.50) for risk management
                    stop_pips = min(min_stop_points, max_stop_points)
                    target_pips = stop_pips * 2.5  # 2.5:1 ratio for swing trading
                    
                else:
                    # Default swing trading
                    pip_value = 0.0001
                    # Calculate minimum stop based on ATR with reasonable cap
                    min_stop_pips = max(min_swing_distance / pip_value, 32)  # At least 32 pips or ATR minimum
                    max_stop_pips = 120  # Cap at 120 pips for risk management
                    stop_pips = min(min_stop_pips, max_stop_pips)
                    target_pips = stop_pips * 2.5  # 2.5:1 ratio for swing trading
                
                # Check if we're hitting the maximum cap due to high volatility
                if 'BTC' in symbol:
                    min_stop_distance = max(min_swing_distance, 320.0)
                    if min_stop_distance > 2000.0:
                        logger.warning(f"{symbol} - Extremely high volatility! ATR requires ${min_stop_distance:.0f} stop, capped at ${stop_pips:.0f}")
                elif 'USD' in symbol and symbol.endswith('m'):
                    min_stop_pips = max(min_swing_distance / pip_value, 40)
                    if min_stop_pips > 200:
                        logger.warning(f"{symbol} - Extremely high volatility! ATR requires {min_stop_pips:.0f} pip stop, capped at {stop_pips:.0f} pips")
                elif 'XAU' in symbol or 'GOLD' in symbol.upper():
                    min_stop_points = max(min_swing_distance / pip_value, 32)
                    if min_stop_points > 150:
                        logger.warning(f"{symbol} - Extremely high volatility! ATR requires {min_stop_points:.0f} point stop, capped at {stop_pips:.0f} points")
                else:
                    min_stop_pips = max(min_swing_distance / pip_value, 32)
                    if min_stop_pips > 120:
                        logger.warning(f"{symbol} - Extremely high volatility! ATR requires {min_stop_pips:.0f} pip stop, capped at {stop_pips:.0f} pips")
                
                logger.info(f"{symbol} ATR-based swing calculation:")
                logger.info(f"  ATR: {atr:.5f}, Min distance: {min_swing_distance:.5f}")
                logger.info(f"  Calculated stop: {stop_pips:.1f} {'$' if 'BTC' in symbol else 'pips/points'}")
                logger.info(f"  Calculated target: {target_pips:.1f} {'$' if 'BTC' in symbol else 'pips/points'}")
                
                if confluence['signal'] == 1:
                    df.loc[latest_idx, 'signal'] = 1
                    df.loc[latest_idx, 'signal_type'] = 'swing_bullish'
                    df.loc[latest_idx, 'ai_confidence'] = confluence['confidence']
                    
                    # Swing trading entry levels - ATR-based validation
                    entry_price = current_price
                    stop_loss = entry_price - (stop_pips * pip_value)
                    take_profit = entry_price + (target_pips * pip_value)
                    
                    # Validate all values are valid numbers FIRST
                    if not all(not pd.isna(val) and val > 0 for val in [entry_price, stop_loss, take_profit]):
                        logger.error(f"Invalid signal values: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
                        df.loc[latest_idx, 'signal'] = 0  # Cancel signal
                        self._clear_invalid_signal_values(df, latest_idx)
                        return df
                    
                    # Validation - log the actual distances for verification
                    actual_stop_distance = abs(entry_price - stop_loss)
                    logger.info(f"  Final stop distance: {actual_stop_distance:.5f} (meets ATR minimum: {min_swing_distance:.5f})")
                    
                    # False breakout protection - ensure price movement confirmation
                    if len(df) >= 5:
                        recent_highs = df['high'].rolling(5).max().iloc[-1]
                        recent_lows = df['low'].rolling(5).min().iloc[-1]
                        price_range = recent_highs - recent_lows
                        
                        # Require minimum price movement relative to recent range
                        if price_range > 0 and actual_stop_distance < price_range * 0.3:
                            logger.warning(f"Potential false breakout - insufficient range confirmation - skipping")
                            # Clear any NaN values before returning
                            self._clear_invalid_signal_values(df, latest_idx)
                            return df
                    
                    # Set signal values (validation already done above)
                    df.loc[latest_idx, 'entry_price'] = entry_price
                    df.loc[latest_idx, 'stop_loss'] = stop_loss
                    df.loc[latest_idx, 'take_profit'] = take_profit
                    df.loc[latest_idx, 'risk_reward'] = target_pips / stop_pips
                    
                    # Dynamic logging based on symbol type
                    if 'BTC' in symbol:
                        logger.info(f"SWING BUY signal - Entry: {entry_price:.2f}, "
                                  f"Target: +${target_pips}, Risk: -${stop_pips}, ATR: {atr:.2f}")
                    else:
                        logger.info(f"SWING BUY signal - Entry: {entry_price:.5f}, "
                                  f"Target: +{target_pips} pips, Risk: -{stop_pips} pips, ATR: {atr:.5f}")
                
                elif confluence['signal'] == -1:
                    df.loc[latest_idx, 'signal'] = -1
                    df.loc[latest_idx, 'signal_type'] = 'swing_bearish'
                    df.loc[latest_idx, 'ai_confidence'] = confluence['confidence']
                    
                    # Swing trading entry levels for sell - ATR-based validation
                    entry_price = current_price
                    stop_loss = entry_price + (stop_pips * pip_value)
                    take_profit = entry_price - (target_pips * pip_value)
                    
                    # Validate all values are valid numbers FIRST
                    if not all(not pd.isna(val) and val > 0 for val in [entry_price, stop_loss, take_profit]):
                        logger.error(f"Invalid signal values: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
                        df.loc[latest_idx, 'signal'] = 0  # Cancel signal
                        self._clear_invalid_signal_values(df, latest_idx)
                        return df
                    
                    # Validation - log the actual distances for verification
                    actual_stop_distance = abs(stop_loss - entry_price)
                    logger.info(f"  Final stop distance: {actual_stop_distance:.5f} (meets ATR minimum: {min_swing_distance:.5f})")
                    
                    # False breakout protection - ensure price movement confirmation
                    if len(df) >= 5:
                        recent_highs = df['high'].rolling(5).max().iloc[-1]
                        recent_lows = df['low'].rolling(5).min().iloc[-1]
                        price_range = recent_highs - recent_lows
                        
                        # Require minimum price movement relative to recent range
                        if price_range > 0 and actual_stop_distance < price_range * 0.3:
                            logger.warning(f"Potential false breakout - insufficient range confirmation - skipping")
                            # Clear any NaN values before returning
                            self._clear_invalid_signal_values(df, latest_idx)
                            return df
                    
                    # Set signal values (validation already done above)
                    df.loc[latest_idx, 'entry_price'] = entry_price
                    df.loc[latest_idx, 'stop_loss'] = stop_loss
                    df.loc[latest_idx, 'take_profit'] = take_profit
                    df.loc[latest_idx, 'risk_reward'] = target_pips / stop_pips
                    
                    # Dynamic logging based on symbol type
                    if 'BTC' in symbol:
                        logger.info(f"SWING SELL signal - Entry: {entry_price:.2f}, "
                                  f"Target: +${target_pips}, Risk: -${stop_pips}, ATR: {atr:.2f}")
                    else:
                        logger.info(f"SWING SELL signal - Entry: {entry_price:.5f}, "
                                  f"Target: +{target_pips} pips, Risk: -{stop_pips} pips, ATR: {atr:.5f}")
        
        # Final safety check - clear any remaining NaN values if no signal was generated
        latest_idx = df.index[-1]
        if df.loc[latest_idx, 'signal'] == 0:
            self._clear_invalid_signal_values(df, latest_idx)
        
        return df
    
    def _clear_invalid_signal_values(self, df, idx):
        """
        Clear NaN values from signal columns to prevent invalid trading attempts
        """
        try:
            df.loc[idx, 'signal'] = 0
            df.loc[idx, 'entry_price'] = 0.0
            df.loc[idx, 'stop_loss'] = 0.0
            df.loc[idx, 'take_profit'] = 0.0
            df.loc[idx, 'risk_reward'] = 0.0
            df.loc[idx, 'signal_type'] = ''
            df.loc[idx, 'ai_confidence'] = 0.0
        except Exception as e:
            logger.error(f"Error clearing invalid signal values: {str(e)}")
    
    def set_current_trade(self, trade_info):
        """
        Set the current active trade and initialize trailing stop parameters
        """
        self.current_trade = trade_info
        if trade_info:
            self.trailing_activated = False
            self.initial_stop_distance = abs(trade_info['entry_price'] - trade_info['stop_loss'])
            self.initial_tp_distance = abs(trade_info['entry_price'] - trade_info['take_profit'])
            self.last_trail_price = trade_info['entry_price']
            self.breakeven_activated = False
        else:
            self.trailing_activated = False
            self.initial_stop_distance = None
            self.initial_tp_distance = None
            self.last_trail_price = None
            self.breakeven_activated = False

    def update_trailing_stop(self, df, current_price):
        """
        Enhanced trailing stop with multiple algorithms and take profit trailing
        """
        if not self.current_trade:
            return {'stop_loss': None, 'take_profit': None}
        
        try:
            side = self.current_trade['side']
            entry_price = self.current_trade['entry_price']
            current_stop = self.current_trade['stop_loss']
            current_tp = self.current_trade['take_profit']
            symbol = self.current_trade.get('symbol', 'XAUUSDm')
            
            # Get symbol-specific trailing settings
            symbol_config = config.SYMBOLS.get(symbol, {})
            trailing_config = symbol_config.get('trailing_settings', {})
            
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
            if atr is None or atr <= 0:
                logger.warning("Invalid ATR value for trailing calculation")
                return {'stop_loss': None, 'take_profit': None}
            
            # Calculate profit in terms of initial risk (R)
            if side == 'buy':
                profit = current_price - entry_price
            else:
                profit = entry_price - current_price
                
            profit_r = profit / self.initial_stop_distance if self.initial_stop_distance > 0 else 0
            
            # Results to return
            result = {'stop_loss': None, 'take_profit': None}
            
            # 1. Breakeven Logic
            breakeven_ratio = trailing_config.get('breakeven_ratio', 0.8)
            if not self.breakeven_activated and profit_r >= breakeven_ratio:
                result['stop_loss'] = entry_price + (0.0001 if side == 'buy' else -0.0001)  # Small buffer
                self.breakeven_activated = True
                logger.info(f"Moving stop to breakeven at {result['stop_loss']:.5f} (profit: {profit_r:.2f}R)")
                return result
            
            # 2. Trailing Stop Logic
            start_ratio = trailing_config.get('start_ratio', 1.0)
            trail_step = trailing_config.get('trail_step', 0.5)
            
            if profit_r >= start_ratio:
                self.trailing_activated = True
                new_stop = self._calculate_trailing_stop(df, current_price, current_stop, atr, side)
                if new_stop and new_stop != current_stop:
                    result['stop_loss'] = new_stop
            
            # 3. Trailing Take Profit Logic
            if trailing_config.get('trail_tp', False) and profit_r >= 1.5:
                new_tp = self._calculate_trailing_take_profit(df, current_price, current_tp, atr, side)
                if new_tp and new_tp != current_tp:
                    result['take_profit'] = new_tp
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trailing calculation: {str(e)}")
            return {'stop_loss': None, 'take_profit': None}
    
    def _calculate_trailing_stop(self, df, current_price, current_stop, atr, side):
        """
        Calculate new trailing stop using selected algorithm
        """
        try:
            if self.trailing_algorithm == 'simple':
                return self._simple_trailing(current_price, current_stop, atr, side)
            elif self.trailing_algorithm == 'atr':
                return self._atr_trailing(current_price, current_stop, atr, side)
            elif self.trailing_algorithm == 'enhanced_atr':
                return self._enhanced_atr_trailing(df, current_price, current_stop, atr, side)
            elif self.trailing_algorithm == 'parabolic':
                return self._parabolic_trailing(df, current_price, current_stop, side)
            else:
                return self._enhanced_atr_trailing(df, current_price, current_stop, atr, side)
        except Exception as e:
            logger.error(f"Error in trailing stop calculation: {str(e)}")
            return None
    
    def _enhanced_atr_trailing(self, df, current_price, current_stop, atr, side):
        """
        Enhanced ATR-based trailing with volatility adjustment and swing levels
        """
        try:
            # Base trailing distance
            trail_distance = atr * self.atr_multiplier
            
            # Volatility adjustment
            recent_atr = df['atr'].rolling(5).mean().iloc[-1] if len(df) >= 5 else atr
            avg_atr = df['atr'].rolling(20).mean().iloc[-1] if len(df) >= 20 else atr
            
            if recent_atr > 0 and avg_atr > 0:
                volatility_ratio = recent_atr / avg_atr
                if volatility_ratio > 1.3:  # High volatility
                    trail_distance *= 1.2
                elif volatility_ratio < 0.8:  # Low volatility
                    trail_distance *= 0.8
            
            # Swing level adjustment if enabled
            if self.use_swing_levels:
                swing_level = self._find_swing_level(df, current_price, side)
                if swing_level:
                    if side == 'buy':
                        trail_distance = max(trail_distance, current_price - swing_level)
                    else:
                        trail_distance = max(trail_distance, swing_level - current_price)
            
            # Calculate new stop
            if side == 'buy':
                new_stop = current_price - trail_distance
                # Only update if new stop is higher
                if new_stop > current_stop and new_stop > (self.last_trail_price or 0):
                    self.last_trail_price = new_stop
                    return max(new_stop, current_stop + self.min_trail_distance)
            else:
                new_stop = current_price + trail_distance
                # Only update if new stop is lower
                if new_stop < current_stop and new_stop < (self.last_trail_price or float('inf')):
                    self.last_trail_price = new_stop
                    return min(new_stop, current_stop - self.min_trail_distance)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced ATR trailing: {str(e)}")
            return None
    
    def _simple_trailing(self, current_price, current_stop, atr, side):
        """
        Simple ATR-based trailing stop
        """
        try:
            trail_distance = atr * 1.5  # Fixed multiplier for simple trailing
            
            if side == 'buy':
                new_stop = current_price - trail_distance
                if new_stop > current_stop:
                    return new_stop
            else:
                new_stop = current_price + trail_distance
                if new_stop < current_stop:
                    return new_stop
            
            return None
        except Exception as e:
            logger.error(f"Error in simple trailing: {str(e)}")
            return None
    
    def _atr_trailing(self, current_price, current_stop, atr, side):
        """
        Standard ATR-based trailing stop
        """
        try:
            trail_distance = atr * self.atr_multiplier
            
            if side == 'buy':
                new_stop = current_price - trail_distance
                if new_stop > current_stop:
                    return new_stop
            else:
                new_stop = current_price + trail_distance
                if new_stop < current_stop:
                    return new_stop
            
            return None
        except Exception as e:
            logger.error(f"Error in ATR trailing: {str(e)}")
            return None
    
    def _parabolic_trailing(self, df, current_price, current_stop, side):
        """
        Parabolic SAR-based trailing stop
        """
        try:
            # Simple parabolic SAR calculation
            acceleration = 0.02
            max_acceleration = 0.2
            
            # This is a simplified version - full implementation would track SAR state
            if side == 'buy':
                # Use recent low as reference
                recent_low = df['low'].rolling(5).min().iloc[-1]
                new_stop = recent_low
                if new_stop > current_stop:
                    return new_stop
            else:
                # Use recent high as reference
                recent_high = df['high'].rolling(5).max().iloc[-1]
                new_stop = recent_high
                if new_stop < current_stop:
                    return new_stop
            
            return None
        except Exception as e:
            logger.error(f"Error in parabolic trailing: {str(e)}")
            return None
    
    def _find_swing_level(self, df, current_price, side, lookback=10):
        """
        Find recent swing high/low for better trailing levels
        """
        try:
            if len(df) < lookback + 2:
                return None
            
            recent_data = df.iloc[-lookback:]
            
            if side == 'buy':
                # Find recent swing low
                lows = recent_data['low']
                min_idx = lows.idxmin()
                swing_low = lows.loc[min_idx]
                
                # Validate swing low (should be below current price)
                if swing_low < current_price * 0.99:  # At least 1% below
                    return swing_low
            else:
                # Find recent swing high
                highs = recent_data['high']
                max_idx = highs.idxmax()
                swing_high = highs.loc[max_idx]
                
                # Validate swing high (should be above current price)
                if swing_high > current_price * 1.01:  # At least 1% above
                    return swing_high
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding swing level: {str(e)}")
            return None
    
    def _calculate_trailing_take_profit(self, df, current_price, current_tp, atr, side):
        """
        Calculate trailing take profit to capture extended moves
        """
        try:
            # Use smaller multiplier for TP trailing
            trail_distance = atr * (self.atr_multiplier * 0.6)
            
            if side == 'buy':
                new_tp = current_price + trail_distance
                # Only update if new TP is higher and price has moved significantly
                if new_tp > current_tp and current_price > (current_tp - trail_distance * 0.5):
                    return new_tp
            else:
                new_tp = current_price - trail_distance
                # Only update if new TP is lower and price has moved significantly
                if new_tp < current_tp and current_price < (current_tp + trail_distance * 0.5):
                    return new_tp
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trailing take profit: {str(e)}")
            return None

    def should_exit_trade(self, df):
        """
        Check if we should exit the trade based on market conditions
        Returns True if we should exit, False otherwise
        """
        if not self.current_trade:
            return False
            
        latest = df.iloc[-1]
        
        # Exit if market structure changes against our position
        if self.current_trade['side'] == 'buy':
            if latest['bos_bearish'] or (latest['bearish_ob'] and self.trailing_activated):
                return True
        else:
            if latest['bos_bullish'] or (latest['bullish_ob'] and self.trailing_activated):
                return True
            
        return False
    
    def _apply_liquidity_sweep_bos_strategy(self, df):
        """
        Apply the Liquidity Sweep + Break of Structure strategy
        
        Args:
            df: DataFrame with indicators applied
        """
        # Look for bullish setup: liquidity sweep low + bullish BOS
        for i in range(5, len(df)):
            # Check for liquidity sweep low in the last 3 bars
            sweep_low = df.iloc[i-3:i]['sweep_low'].any()
            
            # Check for bullish break of structure in current bar
            bos_bullish = df.iloc[i]['bos_bullish']
            
            if sweep_low and bos_bullish:
                # Generate buy signal
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                
                # Set stop loss below the swept liquidity level
                sweep_idx = df.iloc[i-3:i][df.iloc[i-3:i]['sweep_low']].index[-1]
                df.loc[df.index[i], 'stop_loss'] = df.loc[sweep_idx, 'low'] - (config.SL_PADDING * 0.0001)
                
                # Calculate take profit based on risk-reward ratio
                risk = df.iloc[i]['entry_price'] - df.iloc[i]['stop_loss']
                df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] + (risk * self.tp_ratio)
                df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                df.loc[df.index[i], 'signal_type'] = 'Liquidity Sweep + BOS (Buy)'
        
        # Look for bearish setup: liquidity sweep high + bearish BOS
        for i in range(5, len(df)):
            # Check for liquidity sweep high in the last 3 bars
            sweep_high = df.iloc[i-3:i]['sweep_high'].any()
            
            # Check for bearish break of structure in current bar
            bos_bearish = df.iloc[i]['bos_bearish']
            
            if sweep_high and bos_bearish:
                # Generate sell signal
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                
                # Set stop loss above the swept liquidity level
                sweep_idx = df.iloc[i-3:i][df.iloc[i-3:i]['sweep_high']].index[-1]
                df.loc[df.index[i], 'stop_loss'] = df.loc[sweep_idx, 'high'] + (config.SL_PADDING * 0.0001)
                
                # Calculate take profit based on risk-reward ratio
                risk = df.iloc[i]['stop_loss'] - df.iloc[i]['entry_price']
                df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] - (risk * self.tp_ratio)
                df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                df.loc[df.index[i], 'signal_type'] = 'Liquidity Sweep + BOS (Sell)'
    
    def _apply_ob_fvg_return_strategy(self, df):
        """
        Apply the Order Block or Fair Value Gap return strategy
        
        Args:
            df: DataFrame with indicators applied
        """
        # Look for bullish setup: price returning to bearish order block or bearish FVG
        for i in range(10, len(df)):
            # Find recent bearish order blocks and FVGs
            recent_bearish_obs = df.iloc[i-10:i-1][df.iloc[i-10:i-1]['bearish_ob']]
            recent_bearish_fvgs = df.iloc[i-10:i-1][df.iloc[i-10:i-1]['bearish_fvg']]
            
            # Check if current price is near any bearish order block
            for ob_idx in recent_bearish_obs.index:
                ob_low = df.loc[ob_idx, 'bearish_ob_low']
                ob_high = df.loc[ob_idx, 'bearish_ob_high']
                
                # Check if price is returning to the order block
                if df.iloc[i]['low'] <= ob_high and df.iloc[i]['high'] >= ob_low:
                    # Check for additional confirmation (e.g., bullish candle)
                    if df.iloc[i]['close'] > df.iloc[i]['open']:
                        # Generate buy signal
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                        
                        # Set stop loss below the order block
                        df.loc[df.index[i], 'stop_loss'] = ob_low - (config.SL_PADDING * 0.0001)
                        
                        # Calculate take profit based on risk-reward ratio
                        risk = df.iloc[i]['entry_price'] - df.iloc[i]['stop_loss']
                        df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] + (risk * self.tp_ratio)
                        df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                        df.loc[df.index[i], 'signal_type'] = 'Bearish OB Return (Buy)'
                        break
            
            # Check if current price is near any bearish FVG
            for fvg_idx in recent_bearish_fvgs.index:
                fvg_low = df.loc[fvg_idx, 'bearish_fvg_low']
                fvg_high = df.loc[fvg_idx, 'bearish_fvg_high']
                
                # Check if price is returning to the FVG
                if df.iloc[i]['low'] <= fvg_high and df.iloc[i]['high'] >= fvg_low:
                    # Check for additional confirmation (e.g., bullish candle)
                    if df.iloc[i]['close'] > df.iloc[i]['open']:
                        # Generate buy signal
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                        
                        # Set stop loss below the FVG
                        df.loc[df.index[i], 'stop_loss'] = fvg_low - (config.SL_PADDING * 0.0001)
                        
                        # Calculate take profit based on risk-reward ratio
                        risk = df.iloc[i]['entry_price'] - df.iloc[i]['stop_loss']
                        df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] + (risk * self.tp_ratio)
                        df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                        df.loc[df.index[i], 'signal_type'] = 'Bearish FVG Return (Buy)'
                        break
        
        # Look for bearish setup: price returning to bullish order block or bullish FVG
        for i in range(10, len(df)):
            # Find recent bullish order blocks and FVGs
            recent_bullish_obs = df.iloc[i-10:i-1][df.iloc[i-10:i-1]['bullish_ob']]
            recent_bullish_fvgs = df.iloc[i-10:i-1][df.iloc[i-10:i-1]['bullish_fvg']]
            
            # Check if current price is near any bullish order block
            for ob_idx in recent_bullish_obs.index:
                ob_low = df.loc[ob_idx, 'bullish_ob_low']
                ob_high = df.loc[ob_idx, 'bullish_ob_high']
                
                # Check if price is returning to the order block
                if df.iloc[i]['low'] <= ob_high and df.iloc[i]['high'] >= ob_low:
                    # Check for additional confirmation (e.g., bearish candle)
                    if df.iloc[i]['close'] < df.iloc[i]['open']:
                        # Generate sell signal
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                        
                        # Set stop loss above the order block
                        df.loc[df.index[i], 'stop_loss'] = ob_high + (config.SL_PADDING * 0.0001)
                        
                        # Calculate take profit based on risk-reward ratio
                        risk = df.iloc[i]['stop_loss'] - df.iloc[i]['entry_price']
                        df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] - (risk * self.tp_ratio)
                        df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                        df.loc[df.index[i], 'signal_type'] = 'Bullish OB Return (Sell)'
                        break
            
            # Check if current price is near any bullish FVG
            for fvg_idx in recent_bullish_fvgs.index:
                fvg_low = df.loc[fvg_idx, 'bullish_fvg_low']
                fvg_high = df.loc[fvg_idx, 'bullish_fvg_high']
                
                # Check if price is returning to the FVG
                if df.iloc[i]['low'] <= fvg_high and df.iloc[i]['high'] >= fvg_low:
                    # Check for additional confirmation (e.g., bearish candle)
                    if df.iloc[i]['close'] < df.iloc[i]['open']:
                        # Generate sell signal
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'entry_price'] = df.iloc[i]['close']
                        
                        # Set stop loss above the FVG
                        df.loc[df.index[i], 'stop_loss'] = fvg_high + (config.SL_PADDING * 0.0001)
                        
                        # Calculate take profit based on risk-reward ratio
                        risk = df.iloc[i]['stop_loss'] - df.iloc[i]['entry_price']
                        df.loc[df.index[i], 'take_profit'] = df.iloc[i]['entry_price'] - (risk * self.tp_ratio)
                        df.loc[df.index[i], 'risk_reward'] = self.tp_ratio
                        df.loc[df.index[i], 'signal_type'] = 'Bullish FVG Return (Sell)'
                        break
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """
        Calculate position size based on risk percentage
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price for the trade
            
        Returns:
            Position size in lots
        """
        # Calculate risk amount in account currency
        risk_amount = account_balance * (self.risk_percent / 100)
        
        # Calculate risk per pip
        risk_per_pip = abs(entry_price - stop_loss)
        
        # Calculate position size (for XAUUSD, 1 lot = 100 oz)
        # For gold, 1 pip is typically $0.01 per oz
        position_size = risk_amount / (risk_per_pip * 100 * 100)  # Convert to lots
        
        # Round to 2 decimal places (minimum lot size is typically 0.01)
        position_size = round(position_size, 2)
        
        # Ensure minimum position size
        position_size = max(position_size, 0.01)
        
        return position_size
    
    def _apply_quality_filter(self, df):
        """
        Apply quality filtering to signals based on technical indicator confluence
        
        Args:
            df: DataFrame with signals and quality indicators
        """
        if len(df) == 0:
            return
        
        latest_idx = df.index[-1]
        latest = df.iloc[-1]
        
        # Only filter if we have a signal
        if latest['signal'] == 0:
            return
        
        logger.info("=" * 70)
        logger.info("QUALITY FILTER ANALYSIS STARTING")
        logger.info("=" * 70)
        
        # Get quality score
        quality_score = latest.get('quality_score', 0.0)
        original_quality = quality_score
        
        logger.info(f"Initial Quality Score: {original_quality:.3f}")
        
        # Quality-based filtering
        if quality_score < self.min_quality_score:
            logger.info("=" * 70)
            logger.info("SIGNAL REJECTED - LOW INITIAL QUALITY")
            logger.info("=" * 70)
            logger.info(f"Quality score {quality_score:.3f} < minimum {self.min_quality_score}")
            df.loc[latest_idx, 'signal'] = 0
            df.loc[latest_idx, 'signal_type'] = f"REJECTED_LOW_QUALITY_{latest['signal_type']}"
            return
        
        # Enhanced quality checks for specific indicators
        bullish_signal = latest['signal'] > 0
        bearish_signal = latest['signal'] < 0
        signal_direction = "BULLISH" if bullish_signal else "BEARISH"
        
        logger.info(f"Signal Direction: {signal_direction}")
        logger.info("=" * 70)
        logger.info("RSI ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        # RSI quality checks
        rsi = latest.get('rsi', 50)
        rsi_oversold = latest.get('rsi_oversold', False)
        rsi_overbought = latest.get('rsi_overbought', False)
        rsi_extreme_oversold = latest.get('rsi_extreme_oversold', False)
        rsi_extreme_overbought = latest.get('rsi_extreme_overbought', False)
        rsi_bullish_div = latest.get('rsi_bullish_div', False)
        rsi_bearish_div = latest.get('rsi_bearish_div', False)
        rsi_rising = latest.get('rsi_rising', False)
        rsi_falling = latest.get('rsi_falling', False)
        
        logger.info(f"RSI Current Value: {rsi:.1f}")
        logger.info(f"RSI Status:")
        logger.info(f"  ├─ Oversold (< 30): {'YES' if rsi_oversold else 'NO'}")
        logger.info(f"  ├─ Overbought (> 70): {'YES' if rsi_overbought else 'NO'}")
        logger.info(f"  ├─ Extreme Oversold (< 20): {'YES' if rsi_extreme_oversold else 'NO'}")
        logger.info(f"  ├─ Extreme Overbought (> 80): {'YES' if rsi_extreme_overbought else 'NO'}")
        logger.info(f"  ├─ Rising: {'YES' if rsi_rising else 'NO'}")
        logger.info(f"  ├─ Falling: {'YES' if rsi_falling else 'NO'}")
        logger.info(f"  ├─ Bullish Divergence: {'YES' if rsi_bullish_div else 'NO'}")
        logger.info(f"  └─ Bearish Divergence: {'YES' if rsi_bearish_div else 'NO'}")
        
        if bullish_signal:
            logger.info("RSI Analysis for BULLISH Signal:")
            # For bullish signals, prefer RSI not extremely overbought
            if rsi > 85:
                quality_adjustment = 0.7
                quality_score *= quality_adjustment
                logger.info(f"RSI QUALITY CONCERN: Extremely overbought ({rsi:.1f} > 85)")
                logger.info(f"Quality penalty applied: {quality_adjustment:.1f}x")
                logger.info(f"New quality score: {quality_score:.3f}")
            # Bonus for RSI divergence
            elif rsi_bullish_div:
                quality_adjustment = 1.2
                quality_score *= quality_adjustment
                logger.info(f"RSI QUALITY BONUS: Bullish divergence detected!")
                logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
                logger.info(f"New quality score: {quality_score:.3f}")
            else:
                logger.info(f"RSI acceptable for bullish signal ({rsi:.1f})")
        
        elif bearish_signal:
            logger.info("RSI Analysis for BEARISH Signal:")
            # For bearish signals, prefer RSI not extremely oversold
            if rsi < 15:
                quality_adjustment = 0.7
                quality_score *= quality_adjustment
                logger.info(f"RSI QUALITY CONCERN: Extremely oversold ({rsi:.1f} < 15)")
                logger.info(f"Quality penalty applied: {quality_adjustment:.1f}x")
                logger.info(f"New quality score: {quality_score:.3f}")
            # Bonus for RSI divergence
            elif rsi_bearish_div:
                quality_adjustment = 1.2
                quality_score *= quality_adjustment
                logger.info(f"RSI QUALITY BONUS: Bearish divergence detected!")
                logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
                logger.info(f"New quality score: {quality_score:.3f}")
            else:
                logger.info(f"RSI acceptable for bearish signal ({rsi:.1f})")
        
        # MACD quality checks
        logger.info("=" * 70)
        logger.info("MACD ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        macd_strong_bullish = latest.get('macd_strong_bullish', False)
        macd_strong_bearish = latest.get('macd_strong_bearish', False)
        macd_bullish = latest.get('macd_bullish', False)
        macd_bearish = latest.get('macd_bearish', False)
        
        logger.info(f"MACD Status:")
        logger.info(f"  ├─ Strong Bullish: {'YES' if macd_strong_bullish else 'NO'}")
        logger.info(f"  ├─ Strong Bearish: {'YES' if macd_strong_bearish else 'NO'}")
        logger.info(f"  ├─ Regular Bullish: {'YES' if macd_bullish else 'NO'}")
        logger.info(f"  └─ Regular Bearish: {'YES' if macd_bearish else 'NO'}")
        
        if bullish_signal and macd_strong_bullish:
            quality_adjustment = 1.15
            quality_score *= quality_adjustment
            logger.info(f"MACD QUALITY BONUS: Strong bullish signal confirmed!")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif bearish_signal and macd_strong_bearish:
            quality_adjustment = 1.15
            quality_score *= quality_adjustment
            logger.info(f"MACD QUALITY BONUS: Strong bearish signal confirmed!")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        else:
            logger.info("No MACD quality bonus applied")
        
        # ADX trend strength check
        logger.info("=" * 70)
        logger.info("TREND STRENGTH ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        adx = latest.get('adx', 0)
        trend_strong = latest.get('trend_strong', False)
        trend_very_strong = latest.get('trend_very_strong', False)
        trend_weak = adx < 20
        
        logger.info(f"ADX Value: {adx:.1f}")
        logger.info(f"Trend Strength Status:")
        logger.info(f"  ├─ Very Strong (ADX > 40): {'YES' if trend_very_strong else 'NO'}")
        logger.info(f"  ├─ Strong (ADX > 25): {'YES' if trend_strong else 'NO'}")
        logger.info(f"  └─ Weak (ADX < 20): {'YES' if trend_weak else 'NO'}")
        
        if trend_very_strong:
            quality_adjustment = 1.2
            quality_score *= quality_adjustment
            logger.info(f"TREND STRENGTH BONUS: Very strong trend (ADX: {adx:.1f})")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif trend_strong:
            quality_adjustment = 1.1
            quality_score *= quality_adjustment
            logger.info(f"TREND STRENGTH BONUS: Strong trend (ADX: {adx:.1f})")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif trend_weak:
            quality_adjustment = 0.8
            quality_score *= quality_adjustment
            logger.info(f"TREND STRENGTH CONCERN: Weak trend (ADX: {adx:.1f})")
            logger.info(f"Quality penalty applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        else:
            logger.info(f"Moderate trend strength - no adjustment")
        
        # Volatility quality checks
        logger.info("=" * 70)
        logger.info("VOLATILITY ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        volatility_high = latest.get('volatility_high', False)
        volatility_normal = latest.get('volatility_normal', False)
        volatility_low = latest.get('volatility_low', False)
        atr_value = latest.get('atr', 0)
        
        logger.info(f"ATR Value: {atr_value:.5f}")
        logger.info(f"Volatility Status:")
        logger.info(f"  ├─ High Volatility: {'YES' if volatility_high else 'NO'}")
        logger.info(f"  ├─ Normal Volatility: {'YES' if volatility_normal else 'NO'}")
        logger.info(f"  └─ Low Volatility: {'YES' if volatility_low else 'NO'}")
        
        if volatility_high:
            quality_adjustment = 0.9
            quality_score *= quality_adjustment
            logger.info(f"VOLATILITY CONCERN: High volatility environment")
            logger.info(f"Quality penalty applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif volatility_normal:
            quality_adjustment = 1.05
            quality_score *= quality_adjustment
            logger.info(f"VOLATILITY BONUS: Normal volatility environment")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        else:
            logger.info("No volatility adjustment applied")
        
        # Bollinger Bands quality checks
        logger.info("=" * 70)
        logger.info("BOLLINGER BANDS ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        bb_squeeze = latest.get('bb_squeeze', False)
        bb_reversal_long = latest.get('bb_reversal_long', False)
        bb_reversal_short = latest.get('bb_reversal_short', False)
        bb_position = latest.get('bb_position', 0.5)
        
        logger.info(f"Bollinger Band Position: {bb_position:.3f} (0=lower, 1=upper)")
        logger.info(f"Bollinger Band Signals:")
        logger.info(f"  ├─ Squeeze Detected: {'YES' if bb_squeeze else 'NO'}")
        logger.info(f"  ├─ Reversal Long: {'YES' if bb_reversal_long else 'NO'}")
        logger.info(f"  └─ Reversal Short: {'YES' if bb_reversal_short else 'NO'}")
        
        if bb_squeeze:
            quality_adjustment = 1.1
            quality_score *= quality_adjustment
            logger.info(f"BOLLINGER BONUS: Squeeze detected (potential breakout)")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        
        if bullish_signal and bb_reversal_long:
            quality_adjustment = 1.15
            quality_score *= quality_adjustment
            logger.info(f"BOLLINGER BONUS: Reversal long signal confirmed")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif bearish_signal and bb_reversal_short:
            quality_adjustment = 1.15
            quality_score *= quality_adjustment
            logger.info(f"BOLLINGER BONUS: Reversal short signal confirmed")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        
        # Volume confirmation
        logger.info("=" * 70)
        logger.info("VOLUME ANALYSIS & DECISION")
        logger.info("=" * 70)
        
        volume_bullish_confirm = latest.get('volume_bullish_confirm', False)
        volume_bearish_confirm = latest.get('volume_bearish_confirm', False)
        high_volume = latest.get('high_volume', False)
        volume_value = latest.get('volume', 0)
        
        logger.info(f"Volume Value: {volume_value}")
        logger.info(f"Volume Signals:")
        logger.info(f"  ├─ High Volume: {'YES' if high_volume else 'NO'}")
        logger.info(f"  ├─ Bullish Confirmation: {'YES' if volume_bullish_confirm else 'NO'}")
        logger.info(f"  └─ Bearish Confirmation: {'YES' if volume_bearish_confirm else 'NO'}")
        
        if bullish_signal and volume_bullish_confirm:
            quality_adjustment = 1.1
            quality_score *= quality_adjustment
            logger.info(f"VOLUME BONUS: Volume confirms bullish move")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        elif bearish_signal and volume_bearish_confirm:
            quality_adjustment = 1.1
            quality_score *= quality_adjustment
            logger.info(f"VOLUME BONUS: Volume confirms bearish move")
            logger.info(f"Quality bonus applied: {quality_adjustment:.1f}x")
            logger.info(f"New quality score: {quality_score:.3f}")
        else:
            logger.info("No volume confirmation bonus")
        
        # Update quality score (capped at 1.0)
        quality_score = min(quality_score, 1.0)
        df.loc[latest_idx, 'quality_score'] = quality_score
        
        logger.info("=" * 70)
        logger.info("FINAL QUALITY DECISION")
        logger.info("=" * 70)
        logger.info(f"Quality Score Journey:")
        logger.info(f"  ├─ Initial Score: {original_quality:.3f}")
        logger.info(f"  ├─ After Adjustments: {quality_score:.3f}")
        logger.info(f"  └─ Change: {'+' if quality_score > original_quality else ''}{quality_score - original_quality:.3f}")
        
        # Final quality check after adjustments
        if quality_score < self.min_quality_score:
            logger.info("=" * 70)
            logger.info("SIGNAL REJECTED AFTER QUALITY ANALYSIS")
            logger.info("=" * 70)
            logger.info(f"Final quality score {quality_score:.3f} < minimum {self.min_quality_score}")
            df.loc[latest_idx, 'signal'] = 0
            df.loc[latest_idx, 'signal_type'] = f"REJECTED_ADJUSTED_QUALITY_{latest['signal_type']}"
            return
        
        # Update signal type based on quality
        original_signal_type = latest['signal_type']
        if quality_score > self.excellent_quality_threshold:
            df.loc[latest_idx, 'signal_type'] = f"EXCELLENT_QUALITY_{original_signal_type}"
            quality_category = "EXCELLENT"
            logger.info("EXCELLENT QUALITY SIGNAL APPROVED")
        elif quality_score > self.good_quality_threshold:
            df.loc[latest_idx, 'signal_type'] = f"GOOD_QUALITY_{original_signal_type}"
            quality_category = "GOOD"
            logger.info("GOOD QUALITY SIGNAL APPROVED")
        else:
            df.loc[latest_idx, 'signal_type'] = f"FAIR_QUALITY_{original_signal_type}"
            quality_category = "FAIR"
            logger.info("FAIR QUALITY SIGNAL APPROVED")
        
        # Log quality analysis summary
        logger.info("=" * 70)
        logger.info("QUALITY FILTER SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Signal: {signal_direction}")
        logger.info(f"Quality Category: {quality_category}")
        logger.info(f"Final Quality Score: {quality_score:.3f}")
        logger.info(f"RSI: {rsi:.1f} ({'Rising' if rsi_rising else 'Falling' if rsi_falling else 'Stable'})")
        logger.info(f"ADX: {adx:.1f} ({'Very Strong' if trend_very_strong else 'Strong' if trend_strong else 'Weak'} trend)")
        logger.info(f"Volatility: {'High' if volatility_high else 'Normal' if volatility_normal else 'Low'}")
        logger.info(f"Final Signal Type: {df.loc[latest_idx, 'signal_type']}")
        logger.info("=" * 70)
    
    def get_quality_insights(self, df):
        """
        Get detailed quality insights for the current market conditions
        
        Args:
            df: DataFrame with all indicators applied
            
        Returns:
            Dictionary with quality insights
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        
        insights = {
            'overall_quality': latest.get('quality_score', 0.0),
            'quality_category': 'poor',
            'rsi_analysis': {
                'value': latest.get('rsi', 50),
                'status': 'neutral',
                'divergence': False
            },
            'macd_analysis': {
                'bullish': latest.get('macd_bullish', False),
                'bearish': latest.get('macd_bearish', False),
                'strong_signal': latest.get('macd_strong_bullish', False) or latest.get('macd_strong_bearish', False)
            },
            'trend_analysis': {
                'adx': latest.get('adx', 0),
                'strength': 'weak',
                'direction': 'sideways'
            },
            'volatility_analysis': {
                'state': 'normal',
                'atr': latest.get('atr', 0)
            },
            'volume_analysis': {
                'confirmation': latest.get('volume_bullish_confirm', False) or latest.get('volume_bearish_confirm', False),
                'high_volume': latest.get('high_volume', False)
            }
        }
        
        # Determine quality category
        quality_score = insights['overall_quality']
        if quality_score > self.excellent_quality_threshold:
            insights['quality_category'] = 'excellent'
        elif quality_score > self.good_quality_threshold:
            insights['quality_category'] = 'good'
        elif quality_score > self.min_quality_score:
            insights['quality_category'] = 'fair'
        else:
            insights['quality_category'] = 'poor'
        
        # RSI analysis
        rsi = insights['rsi_analysis']['value']
        if rsi > 70:
            insights['rsi_analysis']['status'] = 'overbought'
        elif rsi < 30:
            insights['rsi_analysis']['status'] = 'oversold'
        
        insights['rsi_analysis']['divergence'] = latest.get('rsi_bullish_div', False) or latest.get('rsi_bearish_div', False)
        
        # Trend analysis
        adx = insights['trend_analysis']['adx']
        if adx > 40:
            insights['trend_analysis']['strength'] = 'very_strong'
        elif adx > 25:
            insights['trend_analysis']['strength'] = 'strong'
        elif adx < 20:
            insights['trend_analysis']['strength'] = 'weak'
        
        if latest.get('trend_bullish', False):
            insights['trend_analysis']['direction'] = 'bullish'
        elif latest.get('trend_bearish', False):
            insights['trend_analysis']['direction'] = 'bearish'
        
        # Volatility analysis
        if latest.get('volatility_high', False):
            insights['volatility_analysis']['state'] = 'high'
        elif latest.get('volatility_low', False):
            insights['volatility_analysis']['state'] = 'low'
        
        return insights