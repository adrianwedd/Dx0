# Dx0 API Quick Start Guide

This guide will help you get started with the Dx0 Physician API quickly. Follow these steps to authenticate, connect to the WebSocket, and start using the diagnostic features.

## Prerequisites

- Dx0 API server running (default: `http://localhost:8000`)
- Valid user credentials
- Basic understanding of REST APIs and WebSockets

## Quick Setup Checklist

- [ ] Server is running and accessible
- [ ] You have valid user credentials
- [ ] You can make HTTP requests (curl, Postman, or your preferred tool)
- [ ] You can establish WebSocket connections

## Step 1: Authentication

### Login and Get Tokens

First, authenticate to get your access and refresh tokens:

```bash
curl -X POST http://localhost:8000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
}
```

**Save these tokens** - you'll need them for all subsequent API calls.

### Using the Access Token

Include the access token in the Authorization header for all API requests:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/case
```

## Step 2: Basic API Usage

### Get Current Case

Retrieve the current diagnostic case information:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/case
```

**Response:**
```json
{
  "summary": "A 30 year old with cough"
}
```

### Get Available Tests

See what diagnostic tests are available:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/tests
```

**Response:**
```json
{
  "tests": [
    "basic metabolic panel", 
    "complete blood count"
  ]
}
```

## Step 3: WebSocket Connection

### Connect to WebSocket

The real diagnostic conversations happen over WebSocket. Connect using your access token:

**WebSocket URL:**
```
ws://localhost:8000/api/v1/ws?token=YOUR_ACCESS_TOKEN
```

**With Budget Limit (Optional):**
```
ws://localhost:8000/api/v1/ws?token=YOUR_ACCESS_TOKEN&budget=100.0
```

### JavaScript Example

```javascript
const token = 'YOUR_ACCESS_TOKEN';
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws?token=${token}`);

ws.onopen = function() {
    console.log('Connected to Dx0 API');
    
    // Send your first diagnostic question
    ws.send(JSON.stringify({
        action: 'QUESTION',
        content: 'What should I ask about the patient\'s cough?'
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('AI Response:', response.reply);
    
    if (response.done) {
        console.log('Cost:', response.cost);
        console.log('Total Spent:', response.total_spent);
        console.log('Remaining Budget:', response.remaining_budget);
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

### Python Example

```python
import asyncio
import websockets
import json

async def diagnostic_session():
    token = "YOUR_ACCESS_TOKEN"
    uri = f"ws://localhost:8000/api/v1/ws?token={token}"
    
    async with websockets.connect(uri) as websocket:
        # Send a diagnostic question
        message = {
            "action": "QUESTION",
            "content": "What should I ask about the patient's cough?"
        }
        await websocket.send(json.dumps(message))
        
        # Receive the response
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            print(f"AI: {data['reply']}")
            
            if data.get('done'):
                print(f"Cost: ${data.get('cost', 0)}")
                print(f"Total Spent: ${data.get('total_spent', 0)}")
                break

# Run the session
asyncio.run(diagnostic_session())
```

## Step 4: Common Diagnostic Workflows

### Ask Diagnostic Questions

Send questions to get AI-powered diagnostic guidance:

```javascript
ws.send(JSON.stringify({
    action: 'QUESTION',
    content: 'What additional symptoms should I ask about for chest pain?'
}));
```

### Order Diagnostic Tests

Order specific tests based on the diagnostic conversation:

```javascript
ws.send(JSON.stringify({
    action: 'TEST',
    content: 'complete blood count'
}));
```

### Provide Diagnosis

Share your diagnostic thinking or ask for diagnostic suggestions:

```javascript
ws.send(JSON.stringify({
    action: 'DIAGNOSIS',
    content: 'Based on the symptoms, I think this might be pneumonia'
}));
```

## Step 5: Budget Management

### Understanding Costs

Each AI interaction has a cost based on token usage:

```json
{
  "reply": "Based on the symptoms...",
  "done": true,
  "cost": 0.25,              // Cost of this interaction
  "total_spent": 5.75,       // Total spent in session
  "remaining_budget": 94.25  // Remaining budget
}
```

### Setting Budget Limits

Set a budget limit when connecting:

```javascript
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws?token=${token}&budget=50.0`);
```

### Monitoring Budget

Track your spending in real-time:

```javascript
ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    
    if (response.done && response.remaining_budget !== null) {
        if (response.remaining_budget < 10) {
            console.warn('Low budget remaining:', response.remaining_budget);
        }
    }
};
```

## Step 6: Token Management

### Refresh Tokens

Access tokens expire after 1 hour. Use your refresh token to get new tokens:

```bash
curl -X POST http://localhost:8000/api/v1/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

**Response:**
```json
{
  "access_token": "NEW_ACCESS_TOKEN",
  "refresh_token": "NEW_REFRESH_TOKEN"
}
```

**Important:** The old refresh token becomes invalid immediately. Store the new tokens.

### Automatic Token Refresh (JavaScript)

```javascript
class Dx0Client {
    constructor(initialTokens) {
        this.accessToken = initialTokens.access_token;
        this.refreshToken = initialTokens.refresh_token;
    }
    
    async refreshTokens() {
        const response = await fetch('http://localhost:8000/api/v1/refresh', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: this.refreshToken })
        });
        
        if (response.ok) {
            const tokens = await response.json();
            this.accessToken = tokens.access_token;
            this.refreshToken = tokens.refresh_token;
            return true;
        }
        return false;
    }
    
    async apiCall(endpoint) {
        let response = await fetch(`http://localhost:8000/api/v1${endpoint}`, {
            headers: { 'Authorization': `Bearer ${this.accessToken}` }
        });
        
        if (response.status === 401) {
            // Token expired, try to refresh
            if (await this.refreshTokens()) {
                response = await fetch(`http://localhost:8000/api/v1${endpoint}`, {
                    headers: { 'Authorization': `Bearer ${this.accessToken}` }
                });
            }
        }
        
        return response;
    }
}
```

### Logout

Always logout when done to invalidate your session:

```bash
curl -X POST http://localhost:8000/api/v1/logout \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

## Complete Example: Diagnostic Session

Here's a complete example that demonstrates a full diagnostic workflow:

```python
import asyncio
import websockets
import json
import aiohttp

class Dx0DiagnosticClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.access_token = None
        self.refresh_token = None
    
    async def login(self, username, password):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/login",
                json={"username": username, "password": password}
            ) as response:
                if response.status == 200:
                    tokens = await response.json()
                    self.access_token = tokens["access_token"]
                    self.refresh_token = tokens["refresh_token"]
                    return True
                return False
    
    async def get_case(self):
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with session.get(
                f"{self.base_url}/api/v1/case", 
                headers=headers
            ) as response:
                return await response.json()
    
    async def diagnostic_conversation(self):
        uri = f"{self.ws_url}/api/v1/ws?token={self.access_token}&budget=100.0"
        
        async with websockets.connect(uri) as websocket:
            # Start with a question about the case
            await websocket.send(json.dumps({
                "action": "QUESTION",
                "content": "What key questions should I ask about this cough?"
            }))
            
            response = await self.receive_complete_response(websocket)
            print(f"AI Guidance: {response['complete_reply']}")
            
            # Order a test based on the guidance
            await websocket.send(json.dumps({
                "action": "TEST",
                "content": "complete blood count"
            }))
            
            response = await self.receive_complete_response(websocket)
            print(f"Test Results: {response['complete_reply']}")
            print(f"Ordered Tests: {response.get('ordered_tests', [])}")
            
            # Ask for diagnostic suggestion
            await websocket.send(json.dumps({
                "action": "DIAGNOSIS",
                "content": "What are the most likely diagnoses given the symptoms?"
            }))
            
            response = await self.receive_complete_response(websocket)
            print(f"Diagnostic Suggestions: {response['complete_reply']}")
            print(f"Total Cost: ${response.get('total_spent', 0)}")
    
    async def receive_complete_response(self, websocket):
        complete_reply = ""
        final_data = {}
        
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                raise Exception(f"WebSocket error: {data['error']}")
            
            complete_reply += data.get('reply', '')
            
            if data.get('done'):
                final_data = data
                final_data['complete_reply'] = complete_reply
                break
        
        return final_data
    
    async def logout(self):
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{self.base_url}/api/v1/logout",
                json={"refresh_token": self.refresh_token}
            )

# Usage
async def main():
    client = Dx0DiagnosticClient()
    
    # Login
    if await client.login("your_username", "your_password"):
        print("✅ Logged in successfully")
        
        # Get case information
        case = await client.get_case()
        print(f"📋 Current Case: {case['summary']}")
        
        # Start diagnostic conversation
        print("🔍 Starting diagnostic conversation...")
        await client.diagnostic_conversation()
        
        # Logout
        await client.logout()
        print("👋 Logged out")
    else:
        print("❌ Login failed")

# Run the example
asyncio.run(main())
```

## Error Handling Best Practices

### Handle Authentication Errors

```javascript
ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    if (event.code === 1008) {
        console.error('Authentication failed - invalid token');
        // Redirect to login or refresh token
    }
};
```

### Handle Rate Limiting

```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.error === "Rate limit exceeded") {
        console.warn("Rate limit exceeded, please slow down");
        // Implement backoff strategy
    }
};
```

### Handle Budget Exhaustion

```javascript
ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    
    if (response.remaining_budget !== null && response.remaining_budget <= 0) {
        console.warn("Budget exhausted!");
        // Handle budget exhaustion
    }
};
```

## Next Steps

1. **Explore FHIR Export**: If you have admin access, try the FHIR export endpoints
2. **Build a Client**: Create a more sophisticated client application
3. **Monitor Usage**: Track your API usage and costs
4. **Read Full Documentation**: Check the complete API reference for advanced features

## Troubleshooting

### Common Issues

**"Invalid token" errors:**
- Check that your access token hasn't expired (1-hour limit)
- Refresh your token using the refresh endpoint
- Ensure the token is included in the Authorization header

**WebSocket connection fails:**
- Verify the token is passed as a query parameter
- Check that the WebSocket URL uses `ws://` (or `wss://` for HTTPS)
- Ensure the server is running and accessible

**Rate limit exceeded:**
- Slow down your request rate
- Check that you're not exceeding 30 messages per 60 seconds
- Wait for the rate limit window to reset

**Budget-related errors:**
- Check your remaining budget in WebSocket responses
- Set an appropriate budget limit for your session
- Monitor costs to avoid unexpected budget exhaustion

### Getting Help

- Check the full [API Reference](api-reference.md) for detailed documentation
- Review error messages carefully - they often contain helpful information
- Ensure your server configuration matches your client expectations