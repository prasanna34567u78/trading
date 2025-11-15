import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
import os
import json
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
import config

# Add DeepSeek import with error handling
try:
    import httpx
    from deepseek_integration import DeepSeekClient
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("DeepSeek dependencies not available. DeepSeek features will be disabled.")

logger = logging.getLogger('gold_trading_bot')

class DeepSeekAnalyzer:
    """
    DeepSeek AI-powered market analysis and trade suggestions
    """
    def __init__(self):
        self.enabled = (DEEPSEEK_AVAILABLE and 
                       bool(config.DEEPSEEK_API_KEY) and 
                       config.AI_SETTINGS.get('enable_deepseek', False))
        if self.enabled:
            self.client = DeepSeekClient(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL
            )
            logger.info("DeepSeek analyzer initialized")
        else:
            self.client = None
            logger.warning("DeepSeek analyzer disabled - API key not configured, package unavailable, or disabled in config")
    
    async def analyze_market_sentiment(self, market_data, timeframe_analysis):
        """
        Analyze market sentiment using DeepSeek AI
        
        Args:
            market_data: Current market data
            timeframe_analysis: Multi-timeframe analysis results
            
        Returns:
            dict with sentiment analysis
        """
        if not self.enabled:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': 'DeepSeek AI not available'}
        
        try:
            # Prepare market summary for DeepSeek AI
            market_summary = self._prepare_market_summary(market_data, timeframe_analysis)
            
            prompt = f"""
            As an expert ICT/SMC trader, analyze the following gold market data and provide sentiment analysis:

            Market Summary:
            {market_summary}

            Provide analysis in JSON format with:
            1. sentiment: 'bullish', 'bearish', or 'neutral'
            2. confidence: 0.0-1.0
            3. key_factors: list of key factors influencing sentiment
            4. risk_assessment: 'low', 'medium', 'high'
            5. time_horizon: suggested time horizon for the sentiment
            6. reasoning: brief explanation of the analysis

            Focus on ICT concepts like liquidity sweeps, order blocks, fair value gaps, and market structure.
            """
            
            # Try primary model first, then fallback models
            models_to_try = [config.DEEPSEEK_MODEL, 'deepseek-chat', 'deepseek-coder']
            response = None
            
            for model in models_to_try:
                try:
                    response = await self.client.chat_completion(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert ICT/SMC forex trader specializing in gold trading."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=config.DEEPSEEK_MAX_TOKENS,
                        temperature=config.DEEPSEEK_TEMPERATURE
                    )
                    if response and response.get('choices'):
                        break
                except Exception as model_error:
                    logger.warning(f"Async model {model} failed: {str(model_error)}")
                    continue
            
            if response is None or not response.get('choices'):
                raise Exception("All async DeepSeek models failed")
            
            analysis = json.loads(response['choices'][0]['message']['content'])
            logger.info(f"DeepSeek sentiment analysis: {analysis['sentiment']} (confidence: {analysis['confidence']})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in DeepSeek sentiment analysis: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': f'Analysis error: {str(e)}'}
    
    async def get_trade_suggestion(self, market_data, timeframe_analysis, current_position=None):
        """
        Get trade suggestion from DeepSeek AI based on market analysis
        
        Args:
            market_data: Current market data
            timeframe_analysis: Multi-timeframe analysis results
            current_position: Current open position if any
            
        Returns:
            dict with trade suggestion
        """
        if not self.enabled:
            return {'action': 'hold', 'confidence': 0.5, 'reasoning': 'DeepSeek AI not available'}
        
        try:
            market_summary = self._prepare_market_summary(market_data, timeframe_analysis)
            position_info = f"Current Position: {current_position}" if current_position else "No current position"
            
            prompt = f"""
            As an expert ICT/SMC trader, analyze this gold market data and provide a trade suggestion:

            {market_summary}
            {position_info}

            Consider:
            - Market structure (BOS, ChoCh)
            - Liquidity levels and sweeps
            - Order blocks and fair value gaps
            - Premium/discount areas
            - Market sessions and time of day
            - Risk management

            Provide response in JSON format:
            {{
                "action": "buy|sell|hold|close",
                "confidence": 0.0-1.0,
                "entry_zone": [price_low, price_high],
                "stop_loss": price,
                "take_profit": [tp1, tp2, tp3],
                "risk_reward": ratio,
                "reasoning": "detailed explanation",
                "confluence_factors": ["factor1", "factor2", ...],
                "session_bias": "london|new_york|asian",
                "time_sensitivity": "immediate|wait_for_confirmation|end_of_session"
            }}
            """
            
            # Try primary model first, then fallback models
            models_to_try = [config.DEEPSEEK_MODEL, 'deepseek-chat', 'deepseek-coder']
            response = None
            
            for model in models_to_try:
                try:
                    response = await self.client.chat_completion(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert ICT/SMC forex trader with 10+ years experience trading gold. Focus on high-probability setups with proper risk management."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=config.DEEPSEEK_MAX_TOKENS,
                        temperature=config.DEEPSEEK_TEMPERATURE
                    )
                    if response and response.get('choices'):
                        break
                except Exception as model_error:
                    logger.warning(f"Async model {model} failed: {str(model_error)}")
                    continue
            
            if response is None or not response.get('choices'):
                raise Exception("All async DeepSeek models failed")
            
            suggestion = json.loads(response['choices'][0]['message']['content'])
            logger.info(f"DeepSeek trade suggestion: {suggestion['action']} (confidence: {suggestion['confidence']})")
            return suggestion
            
        except Exception as e:
            logger.error(f"Error in DeepSeek trade suggestion: {str(e)}")
            return {'action': 'hold', 'confidence': 0.5, 'reasoning': f'Suggestion error: {str(e)}'}
    
    def analyze_market_sentiment_sync(self, market_data, timeframe_analysis):
        """
        Synchronous version of market sentiment analysis using DeepSeek AI
        
        Args:
            market_data: Current market data
            timeframe_analysis: Multi-timeframe analysis results
            
        Returns:
            dict with sentiment analysis
        """
        if not self.enabled:
            logger.debug("DeepSeek sentiment analysis skipped - not enabled")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': 'DeepSeek AI not available'}
        
        try:
            logger.info("=" * 60)
            logger.info("DEEPSEEK AI SENTIMENT ANALYSIS STARTING")
            logger.info("=" * 60)
            logger.info(f"Market Data Input:")
            logger.info(f"  ├─ Current Price: {market_data.get('current_price', 'N/A')}")
            logger.info(f"  ├─ High: {market_data.get('high', 'N/A')}")
            logger.info(f"  ├─ Low: {market_data.get('low', 'N/A')}")
            logger.info(f"  └─ ATR: {market_data.get('atr', 'N/A')}")
            
            # Prepare market summary for DeepSeek AI
            market_summary = self._prepare_market_summary(market_data, timeframe_analysis)
            logger.debug(f"Market summary prepared for DeepSeek: {len(market_summary)} characters")
            
            prompt = f"""
            As an expert ICT/SMC trader, analyze the following gold market data and provide sentiment analysis:

            Market Summary:
            {market_summary}

            Provide analysis in JSON format with:
            1. sentiment: 'bullish', 'bearish', or 'neutral'
            2. confidence: 0.0-1.0
            3. key_factors: list of key factors influencing sentiment
            4. risk_assessment: 'low', 'medium', 'high'
            5. time_horizon: suggested time horizon for the sentiment
            6. reasoning: brief explanation of the analysis

            Focus on ICT concepts like liquidity sweeps, order blocks, fair value gaps, and market structure.
            """
            
            logger.info("Sending request to DeepSeek AI for sentiment analysis...")
            
            # Try primary model first, then fallback models
            models_to_try = [config.DEEPSEEK_MODEL, 'deepseek-chat', 'deepseek-coder']
            response = None
            
            for model in models_to_try:
                try:
                    logger.debug(f"Trying DeepSeek model: {model}")
                    response = self.client.chat_completion_sync(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert ICT/SMC forex trader specializing in gold trading."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=config.DEEPSEEK_MAX_TOKENS,
                        temperature=config.DEEPSEEK_TEMPERATURE
                    )
                    if response and response.get('choices'):
                        logger.info(f"Successfully connected to DeepSeek model: {model}")
                        break
                except Exception as model_error:
                    logger.warning(f"DeepSeek model {model} failed: {str(model_error)}")
                    continue
            
            if response is None or not response.get('choices'):
                logger.error("ALL DEEPSEEK MODELS FAILED - No sentiment analysis available")
                raise Exception("All DeepSeek models failed")
            
            logger.info("Received response from DeepSeek, parsing analysis...")
            
            analysis = json.loads(response['choices'][0]['message']['content'])
            
            # Enhanced decision logging
            logger.info("=" * 60)
            logger.info("DEEPSEEK AI SENTIMENT DECISION")
            logger.info("=" * 60)
            logger.info(f"Sentiment: {analysis['sentiment'].upper()}")
            logger.info(f"Confidence: {analysis['confidence']:.2f} ({analysis['confidence']*100:.1f}%)")
            logger.info(f"Risk Assessment: {analysis.get('risk_assessment', 'unknown').upper()}")
            logger.info(f"Time Horizon: {analysis.get('time_horizon', 'not specified')}")
            
            # Log key factors
            key_factors = analysis.get('key_factors', [])
            if key_factors:
                logger.info(f"Key Factors Identified ({len(key_factors)}):")
                for i, factor in enumerate(key_factors, 1):
                    logger.info(f"  {i}. {factor}")
            
            # Log reasoning
            reasoning = analysis.get('reasoning', 'No reasoning provided')
            logger.info(f"AI Reasoning: {reasoning}")
            
            # Decision impact assessment
            confidence_level = analysis['confidence']
            if confidence_level >= 0.8:
                logger.info("HIGH CONFIDENCE AI SENTIMENT - Strong influence on trade decisions")
            elif confidence_level >= 0.6:
                logger.info("MEDIUM CONFIDENCE AI SENTIMENT - Moderate influence on trade decisions")
            else:
                logger.info("LOW CONFIDENCE AI SENTIMENT - Limited influence on trade decisions")
            
            logger.info("=" * 60)
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"DEEPSEEK PARSING ERROR: Failed to parse sentiment response as JSON: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': f'JSON parse error: {str(e)}'}
        except Exception as e:
            logger.error(f"DEEPSEEK SENTIMENT ERROR: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'reasoning': f'Analysis error: {str(e)}'}
    
    def get_trade_suggestion_sync(self, market_data, timeframe_analysis, current_position=None):
        """
        Synchronous version of trade suggestion from DeepSeek AI
        
        Args:
            market_data: Current market data
            timeframe_analysis: Multi-timeframe analysis results
            current_position: Current open position if any
            
        Returns:
            dict with trade suggestion
        """
        if not self.enabled:
            logger.info("DeepSeek AI trade suggestion skipped - not enabled")
            return {'action': 'hold', 'confidence': 0.5, 'reasoning': 'DeepSeek AI not available'}
        
        try:
            logger.info("=" * 60)
            logger.info("DEEPSEEK AI TRADE SUGGESTION ANALYSIS")
            logger.info("=" * 60)
            
            market_summary = self._prepare_market_summary(market_data, timeframe_analysis)
            position_info = f"Current Position: {current_position}" if current_position else "No current position"
            
            logger.info(f"Position Status: {position_info}")
            
            prompt = f"""
            As an expert ICT/SMC trader, analyze this gold market data and provide a trade suggestion:

            {market_summary}
            {position_info}

            Consider:
            - Market structure (BOS, ChoCh)
            - Liquidity levels and sweeps
            - Order blocks and fair value gaps
            - Premium/discount areas
            - Market sessions and time of day
            - Risk management

            Provide response in JSON format:
            {{
                "action": "buy|sell|hold|close",
                "confidence": 0.0-1.0,
                "entry_zone": [price_low, price_high],
                "stop_loss": price,
                "take_profit": [tp1, tp2, tp3],
                "risk_reward": ratio,
                "reasoning": "detailed explanation",
                "confluence_factors": ["factor1", "factor2", ...],
                "session_bias": "london|new_york|asian",
                "time_sensitivity": "immediate|wait_for_confirmation|end_of_session"
            }}
            """
            
            logger.info("Requesting trade suggestion from DeepSeek AI...")
            
            # Try primary model first, then fallback models
            models_to_try = [config.DEEPSEEK_MODEL, 'deepseek-chat', 'deepseek-coder']
            response = None
            
            for model in models_to_try:
                try:
                    logger.debug(f"Trying DeepSeek model: {model}")
                    response = self.client.chat_completion_sync(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert ICT/SMC forex trader with 10+ years experience trading gold. Focus on high-probability setups with proper risk management."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=config.DEEPSEEK_MAX_TOKENS,
                        temperature=config.DEEPSEEK_TEMPERATURE
                    )
                    if response and response.get('choices'):
                        logger.info(f"Successfully connected to DeepSeek model: {model}")
                        break
                except Exception as model_error:
                    logger.warning(f"DeepSeek model {model} failed: {str(model_error)}")
                    continue
            
            if response is None or not response.get('choices'):
                logger.error("ALL DEEPSEEK MODELS FAILED - No trade suggestion available")
                raise Exception("All DeepSeek models failed")
            
            suggestion = json.loads(response['choices'][0]['message']['content'])
            
            # Enhanced trade suggestion logging
            logger.info("=" * 60)
            logger.info("DEEPSEEK AI TRADE DECISION")
            logger.info("=" * 60)
            logger.info(f"Action: {suggestion['action'].upper()}")
            logger.info(f"Confidence: {suggestion['confidence']:.2f} ({suggestion['confidence']*100:.1f}%)")
            
            # Log entry details if action is buy/sell
            if suggestion['action'] in ['buy', 'sell']:
                entry_zone = suggestion.get('entry_zone', [])
                stop_loss = suggestion.get('stop_loss', 'N/A')
                take_profits = suggestion.get('take_profit', [])
                risk_reward = suggestion.get('risk_reward', 'N/A')
                
                logger.info(f"Entry Zone: {entry_zone}")
                logger.info(f"Stop Loss: {stop_loss}")
                logger.info(f"Take Profits: {take_profits}")
                logger.info(f"Risk/Reward: {risk_reward}")
                
                # Log confluence factors
                confluence_factors = suggestion.get('confluence_factors', [])
                if confluence_factors:
                    logger.info(f"Confluence Factors ({len(confluence_factors)}):")
                    for i, factor in enumerate(confluence_factors, 1):
                        logger.info(f"  {i}. {factor}")
                
                # Log session and timing info
                session_bias = suggestion.get('session_bias', 'N/A')
                time_sensitivity = suggestion.get('time_sensitivity', 'N/A')
                logger.info(f"Session Bias: {session_bias}")
                logger.info(f"Time Sensitivity: {time_sensitivity}")
            
            # Log reasoning
            reasoning = suggestion.get('reasoning', 'No reasoning provided')
            logger.info(f"AI Reasoning: {reasoning}")
            
            # Decision impact assessment
            confidence_level = suggestion['confidence']
            if confidence_level >= 0.8:
                logger.info("HIGH CONFIDENCE TRADE SUGGESTION - Strong recommendation")
            elif confidence_level >= 0.6:
                logger.info("MEDIUM CONFIDENCE TRADE SUGGESTION - Moderate recommendation")
            else:
                logger.info("LOW CONFIDENCE TRADE SUGGESTION - Weak recommendation")
            
            logger.info("=" * 60)
            
            return suggestion
            
        except json.JSONDecodeError as e:
            logger.error(f"DEEPSEEK PARSING ERROR: Failed to parse trade suggestion as JSON: {str(e)}")
            return {'action': 'hold', 'confidence': 0.5, 'reasoning': f'JSON parse error: {str(e)}'}
        except Exception as e:
            logger.error(f"DEEPSEEK TRADE SUGGESTION ERROR: {str(e)}")
            return {'action': 'hold', 'confidence': 0.5, 'reasoning': f'Analysis error: {str(e)}'}
    
    def _prepare_market_summary(self, market_data, timeframe_analysis):
        """
        Prepare a comprehensive market summary for DeepSeek AI analysis
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Determine trading session
            hour = current_time.hour
            if 8 <= hour < 17:
                session = "London"
            elif 13 <= hour < 22:
                session = "New York" if hour >= 13 else "London/NY Overlap"
            else:
                session = "Asian"
            
            summary = f"""
            Current Time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}
            Trading Session: {session}
            
            Current Price Data:
            - Price: {market_data.get('current_price', 'N/A')}
            - High: {market_data.get('high', 'N/A')}
            - Low: {market_data.get('low', 'N/A')}
            - ATR: {market_data.get('atr', 'N/A')}
            
            Multi-Timeframe Analysis:
            """
            
            for tf, data in timeframe_analysis.items():
                summary += f"""
            {tf.upper()} Timeframe:
            - Trend: {data.get('trend', 'N/A')}
            - Structure: {data.get('structure', 'N/A')}
            - Key Levels: {data.get('key_levels', 'N/A')}
            - Signals: {data.get('signals', 'N/A')}
            """
            
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing market summary: {str(e)}")
            return "Market data unavailable"


class AITradeAnalyzer:
    """
    Enhanced AI-based trade analyzer with real historical data training
    """
    def __init__(self, mt5_executor=None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'models/trade_validator.joblib'
        self.scaler_path = 'models/scaler.joblib'
        self.training_data_path = 'models/training_data.csv'
        self.is_model_trained = False
        self.deepseek_analyzer = DeepSeekAnalyzer()
        self.mt5_executor = mt5_executor
        
        # Define consistent feature names (14 features to match requirements)
        self.feature_names = [
            'price_range', 'body_size', 'upper_wick', 'lower_wick',
            'atr', 'momentum', 'has_bos_bullish', 'has_bos_bearish',
            'has_ob_bullish', 'has_ob_bearish', 'has_fvg_bullish',
            'has_fvg_bearish', 'volatility', 'avg_range'
        ]
        
        self.initialize_model()

    def initialize_model(self):
        """Initialize or load the AI model with proper error handling"""
        try:
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Try to load existing model and scaler
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                try:
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                    self.is_model_trained = True
                    logger.info("Loaded existing AI model and scaler")
                    return
                except Exception as e:
                    logger.warning(f"Error loading existing model: {e}. Creating new model.")
            
            # Create new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train with historical data if MT5 executor is available
            if self.mt5_executor and self.mt5_executor.connected:
                logger.info("Training model with real historical data...")
                self._train_with_historical_data()
            else:
                logger.warning("MT5 executor not available, using synthetic training data")
                self._create_synthetic_training_data()
            
        except Exception as e:
            logger.error(f"Error initializing AI model: {str(e)}")
            # Fallback to basic model
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self._create_synthetic_training_data()

    def _train_with_historical_data(self):
        """Train model with real historical data from the last 9 years"""
        try:
            logger.info("Fetching historical data for model training...")
            
            # Fetch 9 years of data across multiple timeframes
            timeframes = ['15m', '1h', '4h', 'D1']  # Added daily for long-term perspective
            all_features = []
            all_targets = []
            
            # Define time limits for each timeframe to get 9 years worth of data
            timeframe_limits = {
                '15m': 24 * 60 * 365 * 9,  # 9 years of 15-min data
                '1h': 24 * 365 * 9,        # 9 years of 1-hour data  
                '4h': 6 * 365 * 9,         # 9 years of 4-hour data
                'D1': 365 * 9              # 9 years of daily data
            }
            
            for timeframe in timeframes:
                # Calculate appropriate limit for this timeframe to get 9 years worth
                try:
                    # Attempt to fetch data - MT5 has limits on how much data can be fetched at once
                    # So we'll fetch in chunks if needed
                    limit = min(timeframe_limits[timeframe], 100000)  # Limit to prevent memory issues
                    
                    logger.info(f"Attempting to fetch {limit} bars of {timeframe} data...")
                    df = self.mt5_executor.fetch_historical_data_mt5(timeframe, limit)
                    
                    if df is None or len(df) < 100:
                        logger.warning(f"Insufficient data for {timeframe} (got {len(df) if df is not None else 0} bars)")
                        continue
                    
                    logger.info(f"Fetched {len(df)} bars of {timeframe} data")
                    
                    # Apply indicators (assuming strategy has analyze_market method)
                    try:
                        from strategy import SMCStrategy
                        strategy = SMCStrategy()
                        df = strategy.analyze_market(df)
                    except Exception as e:
                        logger.warning(f"Could not apply indicators: {e}")
                        # Add basic indicators manually
                        df = self._add_basic_indicators(df)
                    
                    # Prepare features
                    features = self._prepare_training_features(df)
                    
                    # Create targets based on future price movement
                    targets = self._create_training_targets(df)
                    
                    # Filter valid data
                    valid_indices = ~(features.isna().any(axis=1) | pd.isna(targets))
                    features = features[valid_indices]
                    targets = targets[valid_indices]
                    
                    if len(features) > 0:
                        all_features.append(features)
                        all_targets.extend(targets)
                        logger.info(f"Added {len(features)} training samples from {timeframe}")
                    else:
                        logger.warning(f"No valid samples from {timeframe} after filtering")
                        
                except Exception as e:
                    logger.error(f"Error fetching {timeframe} data: {e}")
                    continue  # Continue with other timeframes even if one fails
            
            if not all_features:
                raise ValueError("No valid training data available from any timeframe")
            
            # Combine all features from different timeframes
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = np.array(all_targets)
            
            logger.info(f"Total training samples: {len(combined_features)} across all timeframes")
            
            # Handle potential memory issues with very large datasets
            if len(combined_features) > 500000:  # If more than 500k samples
                logger.info(f"Large dataset detected ({len(combined_features)} samples), sampling for training...")
                # Randomly sample to manage memory and training time
                indices = np.random.choice(len(combined_features), size=500000, replace=False)
                combined_features = combined_features.iloc[indices]
                combined_targets = combined_targets[indices]
                logger.info(f"Sampled dataset to {len(combined_features)} samples for training")
            
            # Save training data
            training_data = combined_features.copy()
            training_data['target'] = combined_targets
            training_data.to_csv(self.training_data_path, index=False)
            logger.info(f"Saved training data: {len(training_data)} samples to {self.training_data_path}")
            
            # Split data with stratification to ensure balanced representation
            # For regression targets, we'll use different approach
            if len(np.unique(combined_targets)) < len(combined_targets) * 0.1:  # If targets are more categorical
                # Use stratified split if targets are categorical
                X_train, X_test, y_train, y_test = train_test_split(
                    combined_features, combined_targets, 
                    test_size=0.2, random_state=42, stratify=combined_targets
                )
            else:
                # Use regular split if targets are continuous
                X_train, X_test, y_train, y_test = train_test_split(
                    combined_features, combined_targets, 
                    test_size=0.2, random_state=42
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training AI model with 9 years of historical data...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Model training completed:")
            logger.info(f"Training accuracy: {train_score:.3f}")
            logger.info(f"Testing accuracy: {test_score:.3f}")
            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Test samples: {len(X_test)}")
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.is_model_trained = True
            logger.info("Model and scaler saved successfully with 9 years of training data")
            
        except Exception as e:
            logger.error(f"Error training with historical data: {str(e)}")
            logger.exception("Detailed error traceback:")  # Added for better debugging
            logger.info("Falling back to synthetic data training")
            self._create_synthetic_training_data()

    def _add_basic_indicators(self, df):
        """Add basic indicators if strategy indicators are not available"""
        try:
            # Calculate ATR
            df['atr'] = self._calculate_atr(df, period=14)
            
            # Add basic structure indicators (simplified)
            df['bos_bullish'] = (df['close'] > df['high'].shift(5)).astype(int)
            df['bos_bearish'] = (df['close'] < df['low'].shift(5)).astype(int)
            
            # Simple order block detection
            df['bullish_ob'] = ((df['close'] > df['open']) & 
                               (df['close'].shift(1) < df['open'].shift(1))).astype(int)
            df['bearish_ob'] = ((df['close'] < df['open']) & 
                               (df['close'].shift(1) > df['open'].shift(1))).astype(int)
            
            # Simple FVG detection
            df['bullish_fvg'] = (df['low'] > df['high'].shift(2)).astype(int)
            df['bearish_fvg'] = (df['high'] < df['low'].shift(2)).astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error adding basic indicators: {e}")
            return df

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            return true_range.rolling(period).mean()
        except Exception:
            return pd.Series([1.0] * len(df), index=df.index)

    def _prepare_training_features(self, df):
        """Prepare features for training using the consistent feature set"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price action features
            features['price_range'] = df['high'] - df['low']
            features['body_size'] = abs(df['close'] - df['open'])
            features['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            features['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Technical indicators with data validation
            atr_values = df.get('atr')
            if atr_values is None or atr_values.isna().all():
                # Calculate ATR if not available
                atr_values = self._calculate_atr(df, period=14)
            features['atr'] = atr_values.fillna(1.0)
            
            features['momentum'] = df['close'] - df['close'].shift(5)
            features['momentum'] = features['momentum'].fillna(0)
            
            # Market structure features (use simple boolean conversion)
            features['has_bos_bullish'] = df.get('bos_bullish', 0).fillna(0).astype(int)
            features['has_bos_bearish'] = df.get('bos_bearish', 0).fillna(0).astype(int)
            features['has_ob_bullish'] = df.get('bullish_ob', 0).fillna(0).astype(int)
            features['has_ob_bearish'] = df.get('bearish_ob', 0).fillna(0).astype(int)
            features['has_fvg_bullish'] = df.get('bullish_fvg', 0).fillna(0).astype(int)
            features['has_fvg_bearish'] = df.get('bearish_fvg', 0).fillna(0).astype(int)
            
            # Enhanced features with validation
            features['volatility'] = features['atr'] / df['close'].replace(0, 1e-10)  # Avoid division by zero
            features['avg_range'] = features['price_range'].rolling(10).mean().fillna(method='bfill').fillna(method='ffill')
            
            # Ensure all features are present and in correct order
            features = features.reindex(columns=self.feature_names, fill_value=0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing training features: {e}")
            logger.exception("Detailed error traceback:")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(0, index=df.index, columns=self.feature_names)

    def _create_training_targets(self, df, future_periods=10):
        """Create training targets based on future price movement"""
        try:
            # Look ahead to see if price moves favorably
            # For longer-term data, we'll use a dynamic approach based on the timeframe
            
            # Calculate the time horizon based on the size of the dataset
            # For 9 years of data, we want to look ahead appropriately
            if len(df) > 100000:  # Large dataset (likely including daily data)
                # For long-term data, look further ahead
                adjusted_periods = min(future_periods * 3, len(df) // 10)  # Look ahead up to 10% of data
            elif len(df) > 10000:  # Medium dataset (likely including hourly data)
                adjusted_periods = min(future_periods * 2, len(df) // 20)  # Look ahead up to 5% of data
            else:  # Smaller dataset (mainly 15m data)
                adjusted_periods = future_periods  # Keep normal look ahead
            
            logger.debug(f"Creating targets with look-ahead period: {adjusted_periods}")
            
            # Look ahead to see if price moves favorably
            future_high = df['high'].shift(-adjusted_periods).rolling(adjusted_periods).max()
            future_low = df['low'].shift(-adjusted_periods).rolling(adjusted_periods).min()
            current_close = df['close']
            
            # Calculate returns based on the look-ahead period
            future_return = (future_high - current_close) / current_close
            future_loss = (future_low - current_close) / current_close
            
            # Define successful trade conditions
            # For bullish target, require price to move up by a meaningful amount
            bullish_threshold = df['atr'].fillna(df['high'] - df['low']).mean() / current_close.mean() * 1.5  # 1.5x ATR
            
            bullish_target = (future_return > bullish_threshold).astype(int)
            bearish_target = (future_loss < -bullish_threshold).astype(int)
            
            # Create binary target (1 for profitable bullish trade, 0 for no clear signal, -1 for bearish)
            targets = np.where(bullish_target == 1, 1, np.where(bearish_target == 1, -1, 0))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error creating training targets: {e}")
            logger.exception("Detailed error traceback:")
            return np.zeros(len(df))

    def _create_synthetic_training_data(self):
        """Create synthetic training data as fallback"""
        try:
            logger.info("Creating synthetic training data...")
            
            n_samples = 1000
            np.random.seed(42)
            
            # Generate synthetic features
            features_data = {
                'price_range': np.random.uniform(0.1, 2.0, n_samples),
                'body_size': np.random.uniform(0.05, 1.5, n_samples),
                'upper_wick': np.random.uniform(0, 0.5, n_samples),
                'lower_wick': np.random.uniform(0, 0.5, n_samples),
                'atr': np.random.uniform(0.1, 1.0, n_samples),
                'momentum': np.random.uniform(-1.0, 1.0, n_samples),
                'has_bos_bullish': np.random.randint(0, 2, n_samples),
                'has_bos_bearish': np.random.randint(0, 2, n_samples),
                'has_ob_bullish': np.random.randint(0, 2, n_samples),
                'has_ob_bearish': np.random.randint(0, 2, n_samples),
                'has_fvg_bullish': np.random.randint(0, 2, n_samples),
                'has_fvg_bearish': np.random.randint(0, 2, n_samples),
                'volatility': np.random.uniform(0.001, 0.05, n_samples),
                'avg_range': np.random.uniform(0.1, 2.0, n_samples)
            }
            
            features = pd.DataFrame(features_data)
            
            # Create synthetic targets
            targets = np.zeros(n_samples)
            for i in range(n_samples):
                score = 0
                # Simple scoring based on feature combinations
                if features.loc[i, 'momentum'] > 0 and features.loc[i, 'has_bos_bullish']:
                    score += 2
                if features.loc[i, 'has_ob_bullish'] and features.loc[i, 'momentum'] > 0:
                    score += 2
                if 0.01 < features.loc[i, 'volatility'] < 0.03:
                    score += 1
                
                targets[i] = 1 if score >= 3 else 0
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.is_model_trained = True
            logger.info(f"Synthetic model training completed. Accuracy: {self.model.score(X_test_scaled, y_test):.3f}")
            
        except Exception as e:
            logger.error(f"Error creating synthetic training data: {str(e)}")
            self.is_model_trained = False

    def prepare_features(self, df, session_info=None):
        """
        Prepare feature set for AI analysis (consistent with training)
        """
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price action features
            features['price_range'] = df['high'] - df['low']
            features['body_size'] = abs(df['close'] - df['open'])
            features['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            features['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Technical indicators
            features['atr'] = df.get('atr', 1.0)
            features['momentum'] = df['close'] - df['close'].shift(5)
            
            # Market structure features
            features['has_bos_bullish'] = df.get('bos_bullish', False).astype(int)
            features['has_bos_bearish'] = df.get('bos_bearish', False).astype(int)
            features['has_ob_bullish'] = df.get('bullish_ob', False).astype(int)
            features['has_ob_bearish'] = df.get('bearish_ob', False).astype(int)
            features['has_fvg_bullish'] = df.get('bullish_fvg', False).astype(int)
            features['has_fvg_bearish'] = df.get('bearish_fvg', False).astype(int)
            
            # Enhanced features
            features['volatility'] = features['atr'] / df['close']
            features['avg_range'] = features['price_range'].rolling(10).mean()
            
            # Ensure correct order and fill missing values
            features = features[self.feature_names].fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return DataFrame with correct structure
            return pd.DataFrame(0, index=df.index, columns=self.feature_names)

    def enhanced_validate_trade(self, df, signal, timeframe_analysis=None):
        """
        Enhanced trade validation with DeepSeek AI integration (synchronous version)
        """
        try:
            logger.info("=" * 70)
            logger.info("🔍 ENHANCED AI TRADE VALIDATION STARTING")
            logger.info("=" * 70)
            
            # Get traditional ML validation
            ml_validation = self.validate_trade(df, signal)
            
            logger.info(f"🧮 ML Validation Results:")
            logger.info(f"  ├─ Valid: {ml_validation['valid']}")
            logger.info(f"  ├─ Confidence: {ml_validation['confidence']:.3f}")
            logger.info(f"  └─ Market Conditions: {ml_validation.get('market_conditions', {})}")
            
            # Get DeepSeek AI analysis if available and timeframe analysis provided
            if self.deepseek_analyzer.enabled:
                logger.info("Running DeepSeek AI enhanced analysis...")
                
                market_data = {
                    'current_price': df['close'].iloc[-1],
                    'high': df['high'].iloc[-1],
                    'low': df['low'].iloc[-1],
                    'atr': df.get('atr', [1.0]).iloc[-1]
                }
                
                # Get sentiment and trade suggestion using synchronous calls
                sentiment_analysis = self.deepseek_analyzer.analyze_market_sentiment_sync(market_data, timeframe_analysis)
                trade_suggestion = self.deepseek_analyzer.get_trade_suggestion_sync(market_data, timeframe_analysis)
                
                # Combine ML and DeepSeek AI analysis
                combined_confidence = (
                    ml_validation['confidence'] * 0.4 +
                    sentiment_analysis.get('confidence', 0.5) * 0.3 +
                    trade_suggestion.get('confidence', 0.5) * 0.3
                )
                
                # Check confluence factors
                confluence_count = len(trade_suggestion.get('confluence_factors', []))
                
                logger.info("=" * 70)
                logger.info("AI VALIDATION DECISION MATRIX")
                logger.info("=" * 70)
                logger.info(f"ML Confidence (40% weight): {ml_validation['confidence']:.3f}")
                logger.info(f"Sentiment Confidence (30% weight): {sentiment_analysis.get('confidence', 0.5):.3f}")
                logger.info(f"Trade Suggestion Confidence (30% weight): {trade_suggestion.get('confidence', 0.5):.3f}")
                logger.info(f"Confluence Factors Count: {confluence_count}")
                logger.info(f"Combined Confidence Score: {combined_confidence:.3f}")
                
                # Enhanced validation logic
                deepseek_agrees = False
                signal_direction = "BUY" if signal == 1 else "SELL" if signal == -1 else "NONE"
                
                if signal == 1:  # Buy signal
                    deepseek_agrees = (sentiment_analysis.get('sentiment') == 'bullish' and 
                                     trade_suggestion.get('action') in ['buy', 'hold'])
                    logger.info(f"BUY Signal Analysis:")
                    logger.info(f"  ├─ AI Sentiment: {sentiment_analysis.get('sentiment', 'unknown').upper()}")
                    logger.info(f"  ├─ AI Suggestion: {trade_suggestion.get('action', 'unknown').upper()}")
                    logger.info(f"  └─ DeepSeek Agrees: {'YES' if deepseek_agrees else 'NO'}")
                    
                elif signal == -1:  # Sell signal
                    deepseek_agrees = (sentiment_analysis.get('sentiment') == 'bearish' and 
                                     trade_suggestion.get('action') in ['sell', 'hold'])
                    logger.info(f"SELL Signal Analysis:")
                    logger.info(f"  ├─ AI Sentiment: {sentiment_analysis.get('sentiment', 'unknown').upper()}")
                    logger.info(f"  ├─ AI Suggestion: {trade_suggestion.get('action', 'unknown').upper()}")
                    logger.info(f"  └─ DeepSeek Agrees: {'YES' if deepseek_agrees else 'NO'}")
                
                # Final validation criteria
                min_confidence = 0.6
                min_confluence = 2
                
                enhanced_valid = (
                    ml_validation['valid'] and
                    combined_confidence >= min_confidence and
                    confluence_count >= min_confluence and
                    deepseek_agrees
                )
                
                logger.info("=" * 70)
                logger.info("FINAL AI VALIDATION DECISION")
                logger.info("=" * 70)
                logger.info(f"ML Valid: {ml_validation['valid']}")
                logger.info(f"Confidence >= {min_confidence}: {combined_confidence:.3f} >= {min_confidence} = {'YES' if combined_confidence >= min_confidence else 'NO'}")
                logger.info(f"Confluence >= {min_confluence}: {confluence_count} >= {min_confluence} = {'YES' if confluence_count >= min_confluence else 'NO'}")
                logger.info(f"DeepSeek Agrees: {'YES' if deepseek_agrees else 'NO'}")
                logger.info(f"FINAL DECISION: {'TRADE APPROVED' if enhanced_valid else 'TRADE REJECTED'}")
                
                if enhanced_valid:
                    logger.info("ENHANCED AI VALIDATION PASSED - Trade has high probability of success")
                else:
                    logger.info("ENHANCED AI VALIDATION FAILED - Trade does not meet AI criteria")
                    
                logger.info("=" * 70)
                
                return {
                    'valid': enhanced_valid,
                    'confidence': combined_confidence,
                    'ml_confidence': ml_validation['confidence'],
                    'sentiment': sentiment_analysis,
                    'trade_suggestion': trade_suggestion,
                    'confluence_factors': confluence_count,
                    'market_conditions': ml_validation['market_conditions'],
                    'deepseek_agrees': deepseek_agrees
                }
            else:
                logger.warning("DeepSeek AI not available - falling back to ML validation only")
                # Fallback to ML validation only
                return ml_validation
                
        except Exception as e:
            logger.error(f"ERROR in enhanced trade validation: {str(e)}")
            return self.validate_trade(df, signal)

    def validate_trade(self, df, signal):
        """
        Validate a trade signal using AI analysis
        """
        try:
            if not self.is_model_trained:
                logger.warning("Model not trained yet, using default validation")
                return {
                    'valid': True,
                    'confidence': 0.6,
                    'market_conditions': {
                        'trend_strength': 0,
                        'volatility_state': 'normal',
                        'momentum': 'neutral',
                        'risk_level': 'medium'
                    }
                }
            
            # Prepare features for the latest market conditions
            features = self.prepare_features(df)
            if features.empty:
                raise ValueError("No features could be prepared from the data")
                
            latest_features = features.iloc[-1:].copy()
            
            # Scale features
            scaled_features = self.scaler.transform(latest_features)
            
            # Get model prediction and confidence
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            logger.info(f"AI Model Prediction: {prediction}, Probabilities: {probabilities}")
            
            # Calculate confidence score
            confidence = max(probabilities)  # Use the maximum probability as confidence
            valid = prediction == 1  # 1 indicates profitable trade
            
            # Add market condition analysis
            market_conditions = self.analyze_market_conditions(df)
            
            logger.info(f"AI Trade Validation Result: Valid={valid and confidence >= 0.6}, Confidence={confidence:.3f}")
            
            return {
                'valid': valid and confidence >= 0.6,
                'confidence': confidence,
                'market_conditions': market_conditions
            }
            
        except Exception as e:
            logger.error(f"Error in AI trade validation: {str(e)}")
            logger.exception("Detailed error traceback:")
            # Return a conservative default response
            return {
                'valid': False,
                'confidence': 0.0,
                'market_conditions': {
                    'trend_strength': 0,
                    'volatility_state': 'normal',
                    'momentum': 'neutral',
                    'risk_level': 'high'
                }
            }

    def analyze_market_conditions(self, df):
        """
        Analyze current market conditions for additional insights
        """
        try:
            latest = df.iloc[-10:]  # Look at last 10 candles
            
            conditions = {
                'trend_strength': 0,
                'volatility_state': 'normal',
                'momentum': 'neutral',
                'risk_level': 'medium'
            }
            
            # Analyze trend strength
            price_change = (latest['close'].iloc[-1] - latest['close'].iloc[0]) / latest['close'].iloc[0]
            if price_change > 0.01:
                conditions['trend_strength'] = 1  # Strong uptrend
            elif price_change < -0.01:
                conditions['trend_strength'] = -1  # Strong downtrend
                
            # Analyze volatility
            atr_values = latest.get('atr', pd.Series([1.0] * len(latest)))
            avg_atr = atr_values.mean()
            current_atr = atr_values.iloc[-1]
            
            if current_atr > avg_atr * 1.5:
                conditions['volatility_state'] = 'high'
            elif current_atr < avg_atr * 0.5:
                conditions['volatility_state'] = 'low'
                
            # Analyze momentum
            momentum = (latest['close'].iloc[-1] - latest['close'].iloc[-5]) / latest['close'].iloc[-5]
            if momentum > 0.002:  # 0.2% change
                conditions['momentum'] = 'bullish'
            elif momentum < -0.002:
                conditions['momentum'] = 'bearish'
                
            # Calculate risk level
            risk_factors = 0
            if conditions['volatility_state'] == 'high':
                risk_factors += 1
            if abs(conditions['trend_strength']) < 0.5:
                risk_factors += 1
            if conditions['momentum'] == 'neutral':
                risk_factors += 1
                
            if risk_factors >= 2:
                conditions['risk_level'] = 'high'
            elif risk_factors == 0:
                conditions['risk_level'] = 'low'
                
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                'trend_strength': 0,
                'volatility_state': 'normal',
                'momentum': 'neutral',
                'risk_level': 'medium'
            }

    def update_model(self, trade_result):
        """
        Update the AI model with the results of a completed trade
        """
        try:
            if self.model is None or not self.is_model_trained:
                logger.warning("Model not available for updating")
                return
                
            # Prepare training data from trade result
            market_data = trade_result.get('market_data')
            if market_data is None:
                logger.error("No market data in trade result")
                return
                
            features = self.prepare_features(market_data)
            if features.empty:
                logger.error("Could not prepare features from trade result")
                return
                
            latest_features = features.iloc[-1:].copy()
            target = [1 if trade_result['profitable'] else 0]
            
            # Scale features
            scaled_features = self.scaler.transform(latest_features)
            
            # Update model (simple online learning)
            # Note: RandomForest doesn't support incremental learning, so we just log the result
            logger.info(f"Trade result recorded: {'Profitable' if target[0] else 'Loss'}")
            
            # For more sophisticated updating, you would need to:
            # 1. Load all previous training data
            # 2. Add new sample
            # 3. Retrain model
            # This is left as a future enhancement
            
        except Exception as e:
            logger.error(f"Error updating AI model: {str(e)}")

    def retrain_model(self):
        """
        Retrain the model with fresh historical data
        """
        try:
            logger.info("Retraining model with fresh data...")
            
            if self.mt5_executor and self.mt5_executor.connected:
                self._train_with_historical_data()
            else:
                logger.warning("MT5 executor not available for retraining")
                
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")

    def get_model_info(self):
        """
        Get information about the current model
        """
        try:
            info = {
                'is_trained': self.is_model_trained,
                'model_type': type(self.model).__name__ if self.model else 'None',
                'feature_count': len(self.feature_names),
                'features': self.feature_names,
                'model_exists': os.path.exists(self.model_path),
                'scaler_exists': os.path.exists(self.scaler_path),
                'training_data_exists': os.path.exists(self.training_data_path)
            }
            
            if self.is_model_trained and hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
