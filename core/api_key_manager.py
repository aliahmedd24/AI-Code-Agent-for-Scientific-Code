"""
API Key Manager - Multi-Key Rotation and Rate Limiting

Handles:
- Multiple API key rotation
- Automatic failover when keys are exhausted
- Rate limiting per key
- Cooldown periods for exhausted keys
- Usage tracking and reporting
"""

import os
import time
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import random


@dataclass
class APIKeyStats:
    """Statistics for a single API key."""
    key_id: str  # Masked identifier (last 4 chars)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    exhausted_count: int = 0
    last_used: Optional[datetime] = None
    last_exhausted: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    
    def is_available(self) -> bool:
        """Check if this key is available for use."""
        if self.cooldown_until is None:
            return True
        return datetime.now() > self.cooldown_until
    
    def mark_exhausted(self, cooldown_seconds: int = 60):
        """Mark this key as exhausted with a cooldown period."""
        self.exhausted_count += 1
        self.last_exhausted = datetime.now()
        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
    
    def mark_used(self, success: bool = True):
        """Record a request with this key."""
        self.total_requests += 1
        self.last_used = datetime.now()
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    requests_per_minute: int = 15  # Gemini free tier limit
    tokens: float = field(default=15.0)
    last_update: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """Try to acquire a token, waiting if necessary."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            time.sleep(0.5)
        
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.requests_per_minute,
            self.tokens + elapsed * (self.requests_per_minute / 60.0)
        )
        self.last_update = now


class APIKeyManager:
    """
    Manages multiple API keys with rotation, rate limiting, and failover.
    
    Features:
    - Round-robin key rotation
    - Automatic cooldown for exhausted keys
    - Per-key rate limiting
    - Usage statistics
    - Graceful degradation
    """
    
    # Default cooldown periods (in seconds)
    COOLDOWN_EXHAUSTED = 60      # 1 minute cooldown when rate limited
    COOLDOWN_ERROR = 30          # 30 second cooldown on errors
    COOLDOWN_QUOTA = 3600        # 1 hour cooldown when quota exceeded
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        requests_per_minute: int = 15,
        auto_load_from_env: bool = True
    ):
        """
        Initialize the API key manager.
        
        Args:
            api_keys: List of API keys
            requests_per_minute: Rate limit per key
            auto_load_from_env: Load keys from environment variables
        """
        self._keys: List[str] = []
        self._key_stats: Dict[str, APIKeyStats] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()
        self._requests_per_minute = requests_per_minute
        
        # Load keys
        if api_keys:
            for key in api_keys:
                self.add_key(key)
        
        if auto_load_from_env:
            self._load_from_environment()
    
    def _load_from_environment(self):
        """Load API keys from environment variables."""
        # Primary key
        primary = os.getenv("GEMINI_API_KEY")
        if primary:
            self.add_key(primary)
        
        # Additional keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
        for i in range(1, 11):  # Support up to 10 additional keys
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.add_key(key)
        
        # Also check comma-separated list
        keys_list = os.getenv("GEMINI_API_KEYS", "")
        if keys_list:
            for key in keys_list.split(","):
                key = key.strip()
                if key:
                    self.add_key(key)
    
    def add_key(self, api_key: str) -> bool:
        """Add a new API key to the pool."""
        if not api_key or api_key in self._keys:
            return False
        
        key_id = f"...{api_key[-4:]}"
        
        self._keys.append(api_key)
        self._key_stats[api_key] = APIKeyStats(key_id=key_id)
        self._rate_limiters[api_key] = RateLimiter(
            requests_per_minute=self._requests_per_minute
        )
        
        print(f"✓ Added API key {key_id} (total: {len(self._keys)} keys)")
        return True
    
    def remove_key(self, api_key: str) -> bool:
        """Remove an API key from the pool."""
        if api_key in self._keys:
            self._keys.remove(api_key)
            del self._key_stats[api_key]
            del self._rate_limiters[api_key]
            return True
        return False
    
    async def get_key(self, wait: bool = True, timeout: float = 60.0) -> Optional[str]:
        """
        Get an available API key.
        
        Args:
            wait: Whether to wait for a key to become available
            timeout: Maximum time to wait
        
        Returns:
            An available API key or None if none available
        """
        if not self._keys:
            return None
        
        start_time = time.time()
        
        while True:
            async with self._lock:
                # Try each key in round-robin order
                for _ in range(len(self._keys)):
                    key = self._keys[self._current_index]
                    self._current_index = (self._current_index + 1) % len(self._keys)
                    
                    stats = self._key_stats[key]
                    rate_limiter = self._rate_limiters[key]
                    
                    # Check if key is available (not in cooldown)
                    if stats.is_available():
                        # Try to acquire rate limit token
                        if rate_limiter.acquire(timeout=0.1):
                            return key
            
            # No key available right now
            if not wait or (time.time() - start_time) > timeout:
                return None
            
            # Wait a bit before retrying
            await asyncio.sleep(1.0)
    
    def mark_success(self, api_key: str):
        """Mark a successful request for the given key."""
        if api_key in self._key_stats:
            self._key_stats[api_key].mark_used(success=True)
    
    def mark_failure(self, api_key: str, error_type: str = "unknown"):
        """Mark a failed request for the given key."""
        if api_key not in self._key_stats:
            return
        
        stats = self._key_stats[api_key]
        stats.mark_used(success=False)
        
        # Determine cooldown based on error type
        if "ResourceExhausted" in error_type or "rate" in error_type.lower():
            stats.mark_exhausted(self.COOLDOWN_EXHAUSTED)
            print(f"⚠️  Key {stats.key_id} rate limited, cooldown {self.COOLDOWN_EXHAUSTED}s")
        elif "quota" in error_type.lower():
            stats.mark_exhausted(self.COOLDOWN_QUOTA)
            print(f"⚠️  Key {stats.key_id} quota exceeded, cooldown {self.COOLDOWN_QUOTA}s")
        else:
            stats.mark_exhausted(self.COOLDOWN_ERROR)
    
    def get_available_count(self) -> int:
        """Get the number of currently available keys."""
        return sum(1 for key in self._keys if self._key_stats[key].is_available())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all keys."""
        stats = {
            "total_keys": len(self._keys),
            "available_keys": self.get_available_count(),
            "keys": []
        }
        
        for key in self._keys:
            key_stats = self._key_stats[key]
            stats["keys"].append({
                "key_id": key_stats.key_id,
                "total_requests": key_stats.total_requests,
                "successful": key_stats.successful_requests,
                "failed": key_stats.failed_requests,
                "exhausted_count": key_stats.exhausted_count,
                "available": key_stats.is_available(),
                "cooldown_remaining": (
                    max(0, (key_stats.cooldown_until - datetime.now()).total_seconds())
                    if key_stats.cooldown_until else 0
                )
            })
        
        return stats
    
    def print_status(self):
        """Print current status of all keys."""
        stats = self.get_stats()
        print(f"\n📊 API Key Status: {stats['available_keys']}/{stats['total_keys']} available")
        
        for key_info in stats["keys"]:
            status = "✓" if key_info["available"] else f"⏳ {key_info['cooldown_remaining']:.0f}s"
            print(f"  {key_info['key_id']}: {status} "
                  f"({key_info['successful']}/{key_info['total_requests']} requests)")


# Global instance
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def set_key_manager(manager: APIKeyManager):
    """Set the global API key manager instance."""
    global _key_manager
    _key_manager = manager


def add_api_key(key: str) -> bool:
    """Add an API key to the global manager."""
    return get_key_manager().add_key(key)


async def get_available_key(wait: bool = True, timeout: float = 60.0) -> Optional[str]:
    """Get an available API key from the global manager."""
    return await get_key_manager().get_key(wait=wait, timeout=timeout)
