# DeepSeek AI Integration Module

import httpx
import asyncio
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('gold_trading_bot')

class DeepSeekClient:
    """
    DeepSeek AI API client for trading analysis
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.payment_issue_logged = False  # Track if we've already logged payment issues
    
    def _handle_http_error(self, error: httpx.HTTPError) -> None:
        """
        Handle specific HTTP errors with appropriate logging
        
        Args:
            error: The HTTP error to handle
        """
        if hasattr(error, 'response') and error.response is not None:
            status_code = error.response.status_code
            
            if status_code == 402:  # Payment Required
                if not self.payment_issue_logged:
                    logger.error("=" * 60)
                    logger.error("DEEPSEEK API PAYMENT ISSUE DETECTED")
                    logger.error("=" * 60)
                    logger.error("Your DeepSeek API account has insufficient credits.")
                    logger.error("Please visit https://platform.deepseek.com to:")
                    logger.error("  1. Check your account balance")
                    logger.error("  2. Add credits to your account")
                    logger.error("  3. Verify your payment method")
                    logger.error("")
                    logger.error("The trading bot will continue to work using only")
                    logger.error("ML model validation (without DeepSeek AI analysis).")
                    logger.error("=" * 60)
                    self.payment_issue_logged = True
                else:
                    logger.debug(f"DeepSeek API payment issue (already logged): HTTP {status_code}")
                    
            elif status_code == 401:  # Unauthorized
                logger.error("DeepSeek API authentication failed. Please check your API key.")
                logger.error("Visit https://platform.deepseek.com to get a valid API key.")
                
            elif status_code == 429:  # Rate Limited
                logger.warning("DeepSeek API rate limit exceeded. Please wait before making more requests.")
                
            elif status_code >= 500:  # Server errors
                logger.warning(f"DeepSeek API server error (HTTP {status_code}). Service may be temporarily unavailable.")
                
            else:
                logger.error(f"DeepSeek API error: HTTP {status_code} - {error}")
        else:
            logger.error(f"DeepSeek API network error: {error}")
    
    async def chat_completion(self, messages: list, model: str = "deepseek-chat", 
                            max_tokens: int = 500, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        Create a chat completion using DeepSeek API
        
        Args:
            messages: List of message objects
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Response from DeepSeek API or None if failed
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            self._handle_http_error(e)
            return None
        except Exception as e:
            logger.error(f"DeepSeek API unexpected error: {e}")
            return None
    
    def chat_completion_sync(self, messages: list, model: str = "deepseek-chat", 
                           max_tokens: int = 500, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        Synchronous version of chat completion
        
        Args:
            messages: List of message objects
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Response from DeepSeek API or None if failed
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            self._handle_http_error(e)
            return None
        except Exception as e:
            logger.error(f"DeepSeek API unexpected error: {e}")
            return None
    
    def reset_payment_logging(self):
        """Reset the payment issue logging flag"""
        self.payment_issue_logged = False 