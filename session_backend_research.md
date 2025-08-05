# Session Backend Research & Selection

## Requirements Analysis

Based on the current session management analysis, we need a backend that provides:

### Functional Requirements
1. **Thread-safe concurrent access** - Multiple WebSocket connections
2. **Session data persistence** - Survive application restarts  
3. **Automatic expiration** - TTL-based cleanup
4. **Rate limiting storage** - Per-session message history
5. **Failed login tracking** - IP-based brute force protection
6. **Budget tracking integration** - Seamless with existing BudgetManager

### Non-Functional Requirements
1. **Performance**: < 5ms session operations
2. **Scalability**: Support 1000+ concurrent sessions
3. **Memory efficiency**: Bounded memory usage
4. **Deployment simplicity**: Easy to configure and maintain
5. **Development workflow**: Fast local development setup

## Backend Options Evaluation

### Option 1: Redis (Recommended) ✅

#### Advantages
- **Excellent concurrency**: Built for high-performance concurrent access
- **Native TTL support**: Automatic key expiration
- **Rich data structures**: Lists, Sets, Hashes for complex session data
- **Memory efficiency**: Optimized memory usage and eviction policies
- **Atomic operations**: Prevents race conditions
- **Simple deployment**: Single Redis instance for development/staging
- **Clustering support**: Redis Cluster for production scaling

#### Disadvantages
- **Additional dependency**: Requires Redis server
- **Network latency**: Remote calls vs. local SQLite
- **Memory-only**: Data loss if Redis crashes (mitigated with persistence)

#### Use Cases
- **Development**: Redis in Docker container
- **Staging**: Single Redis instance
- **Production**: Redis Cluster or managed service (AWS ElastiCache)

#### Implementation Complexity: **Low-Medium**
```python
# Example Redis session operations
redis_client.hset(f"session:{session_id}", mapping=session_data)
redis_client.expire(f"session:{session_id}", ttl_seconds)
redis_client.lpush(f"messages:{session_id}", timestamp)
redis_client.ltrim(f"messages:{session_id}", 0, rate_limit - 1)
```

### Option 2: PostgreSQL ⚠️

#### Advantages
- **ACID compliance**: Strong consistency guarantees
- **SQL familiarity**: Easy to query and debug
- **Rich querying**: Complex session analytics possible
- **Mature tooling**: Extensive monitoring and backup tools
- **JSON support**: Native JSON columns for session data

#### Disadvantages
- **Connection overhead**: Connection pooling required
- **Complex TTL**: Manual cleanup jobs needed
- **Slower operations**: Higher latency than Redis
- **Heavyweight**: Overkill for simple session storage

#### Use Cases
- **Large enterprises**: Where PostgreSQL is already deployed
- **Complex session analytics**: Need rich querying capabilities
- **Strong durability requirements**: Cannot tolerate any data loss

#### Implementation Complexity: **Medium-High**
```sql
-- Requires background cleanup job
DELETE FROM sessions WHERE expires_at < NOW();
-- Complex rate limiting queries
SELECT COUNT(*) FROM message_history 
WHERE session_id = ? AND timestamp > NOW() - INTERVAL '60 seconds';
```

### Option 3: Enhanced SQLite (Current + Improvements) ⚠️

#### Advantages
- **No additional dependencies**: Uses existing SQLite
- **Simple deployment**: Single file database
- **Atomic transactions**: ACID compliance
- **Familiar technology**: Team already knows SQLite

#### Disadvantages
- **Limited concurrency**: Write locks limit scalability
- **Manual cleanup**: Need background jobs for TTL
- **Global state issues**: Doesn't solve thread-safety problems
- **Performance ceiling**: Not suitable for high concurrency

#### Use Cases
- **Development only**: Quick local development
- **Very small deployments**: < 50 concurrent users
- **Embedded scenarios**: Single-user applications

#### Implementation Complexity: **Low**
```python
# Still need to solve global dictionary issues
# SQLite alone doesn't fix thread-safety problems
```

### Option 4: In-Memory with Locking (Quick Fix) ⚠️

#### Advantages
- **Minimal changes**: Fix existing code with locks
- **No new dependencies**: Pure Python solution
- **Fast operations**: No I/O overhead

#### Disadvantages
- **Data loss**: All session data lost on restart
- **Memory leaks**: Still unbounded growth potential
- **Lock contention**: Performance bottleneck under load
- **Not production-ready**: Doesn't solve fundamental issues

#### Use Cases
- **Emergency fix**: Temporary solution for immediate thread-safety
- **Development**: Quick local testing

#### Implementation Complexity: **Very Low**
```python
from threading import RLock
session_lock = RLock()
with session_lock:
    MESSAGE_HISTORY[session_id].append(timestamp)
```

## Decision Matrix

| Criteria | Redis | PostgreSQL | Enhanced SQLite | In-Memory + Locks |
|----------|--------|------------|-----------------|-------------------|
| **Thread Safety** | ✅ Excellent | ✅ Excellent | ⚠️ Limited | ✅ Good |
| **Performance** | ✅ Excellent | ⚠️ Good | ⚠️ Limited | ✅ Excellent |
| **Scalability** | ✅ Excellent | ✅ Good | ❌ Poor | ❌ Poor |
| **TTL Support** | ✅ Native | ⚠️ Manual | ⚠️ Manual | ❌ None |
| **Deployment** | ⚠️ Additional Service | ⚠️ Additional Service | ✅ Built-in | ✅ Built-in |
| **Development** | ✅ Docker | ⚠️ Setup Required | ✅ Simple | ✅ Simple |
| **Memory Usage** | ✅ Bounded | ✅ Bounded | ✅ Bounded | ❌ Unbounded |
| **Data Persistence** | ✅ Configurable | ✅ Durable | ✅ Durable | ❌ Volatile |

## Architectural Decision

### **Selected Backend: Redis** ✅

#### Rationale
1. **Best fit for requirements**: Excellent concurrency, native TTL, atomic operations
2. **Deployment flexibility**: Can be deployed standalone or as managed service
3. **Development friendly**: Easy Docker setup for local development
4. **Production proven**: Used by many high-scale web applications
5. **Addresses all current issues**: Solves thread-safety, memory leaks, and scalability

#### Implementation Strategy

**Phase 1: Abstract Interface**
```python
class SessionBackend(ABC):
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionData]: ...
    @abstractmethod
    async def set_session(self, session_id: str, data: SessionData, ttl: int): ...
    @abstractmethod
    async def delete_session(self, session_id: str): ...
    @abstractmethod
    async def add_message_timestamp(self, session_id: str, timestamp: float): ...
    @abstractmethod
    async def get_message_count(self, session_id: str, window: int) -> int: ...
```

**Phase 2: Redis Implementation**
```python
class RedisSessionBackend(SessionBackend):
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url)
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        data = await self.redis.hgetall(f"session:{session_id}")
        return SessionData.from_dict(data) if data else None
```

**Phase 3: Fallback Strategy**
- Redis for production/staging
- In-memory with locks for development (if Redis unavailable)
- Graceful degradation with warnings

#### Configuration Integration

Add to `sdb/config.py`:
```python
class Settings(BaseModel):
    # Session Backend Configuration
    session_backend: str = "redis"  # redis, sqlite, memory
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    session_ttl: int = 3600
    message_rate_window: int = 60
    failed_login_window: int = 300
```

#### Deployment Scenarios

**Development:**
```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

**Staging:**
```bash
# Single Redis instance
export REDIS_URL=redis://staging-redis:6379
```

**Production:**
```bash
# Managed Redis service
export REDIS_URL=redis://elasticache-cluster.amazonaws.com:6379
export REDIS_PASSWORD=your-secure-password
```

## Next Steps

1. **Implement abstract session interface** (Issue #310 Chunk 3)
2. **Create Redis backend implementation** (Issue #310 Chunk 4)
3. **Integrate with FastAPI application** (Issue #310 Chunk 5)
4. **Comprehensive testing and performance validation** (Issue #310 Chunk 6)

## Alternative Backup Plan

If Redis deployment becomes problematic:
1. **Immediate fix**: Implement in-memory with locks to solve thread-safety
2. **Medium term**: Enhanced SQLite with background cleanup
3. **Long term**: Return to Redis when infrastructure is ready

This ensures we can make progress on thread-safety issues regardless of backend selection.