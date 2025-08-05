"""Redis-based session storage backend."""

from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any

try:
    import aioredis
    from aioredis import Redis
except ImportError:
    aioredis = None
    Redis = None

from .session_backend import SessionBackend, SessionData


class RedisSessionBackend(SessionBackend):
    """Redis-based session backend for production use."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_password: Optional[str] = None,
        key_prefix: str = "sdb_session:",
        failed_login_prefix: str = "sdb_login:",
        pool_size: int = 10,
    ):
        if aioredis is None:
            raise RuntimeError("aioredis is required for Redis session backend")
        
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.key_prefix = key_prefix
        self.failed_login_prefix = failed_login_prefix
        self.pool_size = pool_size
        self._redis: Optional[Redis] = None
    
    async def _get_redis(self) -> Redis:
        """Get Redis connection, initializing if needed."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url,
                password=self.redis_password,
                max_connections=self.pool_size,
                decode_responses=True,
            )
        return self._redis
    
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session data."""
        return f"{self.key_prefix}{session_id}"
    
    def _refresh_token_key(self, refresh_token: str) -> str:
        """Generate Redis key for refresh token mapping."""
        return f"{self.key_prefix}refresh:{refresh_token}"
    
    def _message_key(self, session_id: str) -> str:
        """Generate Redis key for message timestamps."""
        return f"{self.key_prefix}messages:{session_id}"
    
    def _failed_login_key(self, identifier: str) -> str:
        """Generate Redis key for failed login attempts."""
        return f"{self.failed_login_prefix}{identifier}"
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by session ID."""
        redis = await self._get_redis()
        key = self._session_key(session_id)
        
        data = await redis.hgetall(key)
        if not data:
            return None
        
        # Convert Redis strings back to appropriate types
        try:
            session_data = {
                "session_id": data["session_id"],
                "username": data["username"],
                "group_name": data.get("group_name", "default"),
                "refresh_token": data.get("refresh_token", ""),
                "issue_time": float(data.get("issue_time", time.time())),
                "budget_limit": float(data["budget_limit"]) if data.get("budget_limit") else None,
                "amount_spent": float(data.get("amount_spent", 0.0)),
                "metadata": json.loads(data.get("metadata", "{}")),
            }
            
            # Get message timestamps from separate list
            message_key = self._message_key(session_id)
            message_timestamps = await redis.lrange(message_key, 0, -1)
            session_data["message_timestamps"] = [float(ts) for ts in message_timestamps]
            
            # Failed login attempts are stored separately by IP, not per session
            session_data["failed_login_attempts"] = []
            
            return SessionData.from_dict(session_data)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            # Corrupted session data, remove it
            await redis.delete(key)
            return None
    
    async def set_session(self, session_data: SessionData, ttl: int) -> None:
        """Store or update session data with TTL."""
        redis = await self._get_redis()
        key = self._session_key(session_data.session_id)
        
        # Prepare session data for Redis hash
        redis_data = {
            "session_id": session_data.session_id,
            "username": session_data.username,
            "group_name": session_data.group_name,
            "refresh_token": session_data.refresh_token,
            "issue_time": str(session_data.issue_time),
            "amount_spent": str(session_data.amount_spent),
            "metadata": json.dumps(session_data.metadata),
        }
        
        if session_data.budget_limit is not None:
            redis_data["budget_limit"] = str(session_data.budget_limit)
        
        # Use pipeline for atomic operations
        pipeline = redis.pipeline()
        
        # Store session data
        pipeline.hset(key, mapping=redis_data)
        pipeline.expire(key, ttl)
        
        # Store message timestamps in separate list
        message_key = self._message_key(session_data.session_id)
        if session_data.message_timestamps:
            pipeline.delete(message_key)  # Clear old timestamps
            pipeline.lpush(message_key, *[str(ts) for ts in session_data.message_timestamps])
            pipeline.expire(message_key, ttl)
        
        # Store refresh token mapping
        if session_data.refresh_token:
            refresh_key = self._refresh_token_key(session_data.refresh_token)
            pipeline.set(refresh_key, session_data.session_id, ex=ttl)
        
        await pipeline.execute()
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data. Returns True if session existed."""
        redis = await self._get_redis()
        
        # First get session to find refresh token
        session = await self.get_session(session_id)
        if not session:
            return False
        
        pipeline = redis.pipeline()
        
        # Delete session data
        key = self._session_key(session_id)
        pipeline.delete(key)
        
        # Delete message timestamps
        message_key = self._message_key(session_id)
        pipeline.delete(message_key)
        
        # Delete refresh token mapping
        if session.refresh_token:
            refresh_key = self._refresh_token_key(session.refresh_token)
            pipeline.delete(refresh_key)
        
        results = await pipeline.execute()
        return results[0] > 0  # First delete result indicates if session existed
    
    async def find_by_refresh_token(self, refresh_token: str) -> Optional[SessionData]:
        """Find session by refresh token."""
        redis = await self._get_redis()
        refresh_key = self._refresh_token_key(refresh_token)
        
        session_id = await redis.get(refresh_key)
        if not session_id:
            return None
        
        return await self.get_session(session_id)
    
    async def update_refresh_token(self, session_id: str, new_refresh_token: str) -> bool:
        """Update refresh token for session. Returns True if successful."""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        redis = await self._get_redis()
        pipeline = redis.pipeline()
        
        # Remove old refresh token mapping
        if session.refresh_token:
            old_refresh_key = self._refresh_token_key(session.refresh_token)
            pipeline.delete(old_refresh_key)
        
        # Update session with new refresh token and timestamp
        session.refresh_token = new_refresh_token
        session.issue_time = time.time()
        
        # Store updated session (will be handled by set_session TTL)
        await self.set_session(session, 3600)  # Default TTL, should be configurable
        
        await pipeline.execute()
        return True
    
    async def update_budget(self, session_id: str, amount_spent: float) -> bool:
        """Update budget spent amount. Returns True if successful."""
        redis = await self._get_redis()
        key = self._session_key(session_id)
        
        # Use Redis hash set for atomic update
        result = await redis.hset(key, "amount_spent", str(amount_spent))
        return True  # hset always succeeds, check existence separately if needed
    
    async def add_message_timestamp(self, session_id: str, timestamp: Optional[float] = None) -> bool:
        """Add message timestamp for rate limiting. Returns True if successful."""
        timestamp = timestamp or time.time()
        redis = await self._get_redis()
        message_key = self._message_key(session_id)
        
        pipeline = redis.pipeline()
        pipeline.lpush(message_key, str(timestamp))
        pipeline.ltrim(message_key, 0, 99)  # Keep only last 100 timestamps
        pipeline.expire(message_key, 3600)  # Reset TTL
        
        await pipeline.execute()
        return True
    
    async def get_message_count(self, session_id: str, window_seconds: int) -> int:
        """Get count of messages within the time window."""
        redis = await self._get_redis()
        message_key = self._message_key(session_id)
        
        timestamps = await redis.lrange(message_key, 0, -1)
        if not timestamps:
            return 0
        
        cutoff = time.time() - window_seconds
        recent_count = sum(1 for ts in timestamps if float(ts) >= cutoff)
        return recent_count
    
    async def add_failed_login(self, identifier: str, timestamp: Optional[float] = None) -> bool:
        """Add failed login attempt for IP or session. Returns True if successful."""
        timestamp = timestamp or time.time()
        redis = await self._get_redis()
        key = self._failed_login_key(identifier)
        
        pipeline = redis.pipeline()
        pipeline.lpush(key, str(timestamp))
        pipeline.ltrim(key, 0, 19)  # Keep only last 20 attempts
        pipeline.expire(key, 3600)  # 1 hour TTL
        
        await pipeline.execute()
        return True
    
    async def get_failed_login_count(self, identifier: str, window_seconds: int) -> int:
        """Get count of failed login attempts within the time window."""
        redis = await self._get_redis()
        key = self._failed_login_key(identifier)
        
        timestamps = await redis.lrange(key, 0, -1)
        if not timestamps:
            return 0
        
        cutoff = time.time() - window_seconds
        recent_count = sum(1 for ts in timestamps if float(ts) >= cutoff)
        return recent_count
    
    async def cleanup_expired_sessions(self, ttl: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        # Redis handles TTL automatically, but we can scan for any orphaned keys
        # This is more of a maintenance operation
        redis = await self._get_redis()
        
        pattern = f"{self.key_prefix}*"
        expired_count = 0
        
        async for key in redis.scan_iter(match=pattern):
            ttl_remaining = await redis.ttl(key)
            if ttl_remaining == -1:  # Key without TTL (shouldn't happen)
                await redis.expire(key, ttl)
            elif ttl_remaining == -2:  # Key doesn't exist (race condition)
                expired_count += 1
        
        return expired_count
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        redis = await self._get_redis()
        pattern = f"{self.key_prefix}*"
        
        count = 0
        async for key in redis.scan_iter(match=pattern):
            # Only count main session keys, not message or refresh token keys
            if not (":" in key.replace(self.key_prefix, "", 1)):
                count += 1
        
        return count
    
    async def health_check(self) -> bool:
        """Check if backend is healthy and responsive."""
        try:
            redis = await self._get_redis()
            await redis.ping()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None