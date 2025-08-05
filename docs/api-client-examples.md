# Dx0 API Client Examples

This document provides practical code examples for integrating with the Dx0 Physician API in various programming languages and scenarios.

## Table of Contents

- [Python Examples](#python-examples)
- [JavaScript/Node.js Examples](#javascriptnodejs-examples)
- [cURL Examples](#curl-examples)
- [WebSocket Examples](#websocket-examples)
- [FHIR Export Examples](#fhir-export-examples)
- [Error Handling Patterns](#error-handling-patterns)
- [Production Ready Client](#production-ready-client)

## Python Examples

### Basic Python Client with aiohttp

```python
import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any

class Dx0Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def login(self, username: str, password: str) -> bool:
        """Login and store tokens."""
        async with self.session.post(
            f"{self.base_url}/api/v1/login",
            json={"username": username, "password": password}
        ) as response:
            if response.status == 200:
                tokens = await response.json()
                self.access_token = tokens["access_token"]
                self.refresh_token = tokens["refresh_token"]
                return True
            elif response.status == 401:
                raise ValueError("Invalid credentials")
            elif response.status == 429:
                raise ValueError("Too many failed login attempts")
            else:
                raise RuntimeError(f"Login failed with status {response.status}")
    
    async def refresh_tokens(self) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            return False
        
        async with self.session.post(
            f"{self.base_url}/api/v1/refresh",
            json={"refresh_token": self.refresh_token}
        ) as response:
            if response.status == 200:
                tokens = await response.json()
                self.access_token = tokens["access_token"]
                self.refresh_token = tokens["refresh_token"]
                return True
            return False
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request with automatic token refresh."""
        if not self.access_token:
            raise RuntimeError("Not authenticated")
        
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.access_token}"
        kwargs["headers"] = headers
        
        async with self.session.request(method, f"{self.base_url}/api/v1{endpoint}", **kwargs) as response:
            if response.status == 401:
                # Try to refresh token
                if await self.refresh_tokens():
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    async with self.session.request(method, f"{self.base_url}/api/v1{endpoint}", **kwargs) as retry_response:
                        return await retry_response.json()
                else:
                    raise RuntimeError("Authentication failed")
            
            if response.status >= 400:
                error_data = await response.json()
                raise RuntimeError(f"API error: {error_data.get('detail', 'Unknown error')}")
            
            return await response.json()
    
    async def get_case(self) -> Dict[str, Any]:
        """Get current case summary."""
        return await self._make_request("GET", "/case")
    
    async def get_tests(self) -> Dict[str, Any]:
        """Get available diagnostic tests."""
        return await self._make_request("GET", "/tests")
    
    async def export_transcript_fhir(self, transcript: list, patient_id: str = "example") -> Dict[str, Any]:
        """Export transcript to FHIR Bundle (requires admin access)."""
        return await self._make_request(
            "POST", 
            "/fhir/transcript",
            json={"transcript": transcript, "patient_id": patient_id}
        )
    
    async def export_tests_fhir(self, tests: list, patient_id: str = "example") -> Dict[str, Any]:
        """Export ordered tests to FHIR Bundle (requires admin access)."""
        return await self._make_request(
            "POST", 
            "/fhir/tests",
            json={"tests": tests, "patient_id": patient_id}
        )
    
    async def logout(self):
        """Logout and invalidate tokens."""
        if self.refresh_token:
            try:
                await self._make_request("POST", "/logout", json={"refresh_token": self.refresh_token})
            except:
                pass  # Logout errors are not critical
            finally:
                self.access_token = None
                self.refresh_token = None

# Usage Example
async def main():
    async with Dx0Client() as client:
        # Login
        await client.login("your_username", "your_password")
        print("✅ Logged in successfully")
        
        # Get case information
        case = await client.get_case()
        print(f"📋 Current case: {case['summary']}")
        
        # Get available tests
        tests = await client.get_tests()
        print(f"🧪 Available tests: {tests['tests']}")
        
        # Logout
        await client.logout()
        print("👋 Logged out")

# Run the example
asyncio.run(main())
```

### Python WebSocket Client

```python
import asyncio
import websockets
import json
from typing import AsyncIterator, Dict, Any

class Dx0WebSocketClient:
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
    
    def set_token(self, access_token: str):
        """Set the access token for WebSocket authentication."""
        self.access_token = access_token
    
    async def connect(self, budget: Optional[float] = None) -> websockets.WebSocketServerProtocol:
        """Connect to the WebSocket endpoint."""
        if not self.access_token:
            raise RuntimeError("Access token required")
        
        uri = f"{self.base_url}/api/v1/ws?token={self.access_token}"
        if budget is not None:
            uri += f"&budget={budget}"
        
        return await websockets.connect(uri)
    
    async def send_message(self, websocket, action: str, content: str):
        """Send a message to the diagnostic system."""
        message = {
            "action": action.upper(),
            "content": content
        }
        await websocket.send(json.dumps(message))
    
    async def receive_complete_response(self, websocket) -> Dict[str, Any]:
        """Receive a complete response from the diagnostic system."""
        complete_reply = ""
        final_data = {}
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                
                if 'error' in data:
                    raise RuntimeError(f"WebSocket error: {data['error']}")
                
                complete_reply += data.get('reply', '')
                
                if data.get('done'):
                    final_data = data
                    final_data['complete_reply'] = complete_reply
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                raise RuntimeError("WebSocket connection closed unexpectedly")
        
        return final_data
    
    async def diagnostic_session(self, questions: list, budget: Optional[float] = None) -> list:
        """Run a complete diagnostic session with multiple questions."""
        results = []
        
        async with await self.connect(budget) as websocket:
            for question in questions:
                await self.send_message(websocket, "QUESTION", question)
                response = await self.receive_complete_response(websocket)
                results.append({
                    "question": question,
                    "response": response['complete_reply'],
                    "cost": response.get('cost'),
                    "total_spent": response.get('total_spent'),
                    "remaining_budget": response.get('remaining_budget')
                })
        
        return results

# Combined Usage Example
async def complete_example():
    # First, authenticate using the REST client
    async with Dx0Client() as rest_client:
        await rest_client.login("your_username", "your_password")
        
        # Create WebSocket client and set token
        ws_client = Dx0WebSocketClient()
        ws_client.set_token(rest_client.access_token)
        
        # Run diagnostic session
        questions = [
            "What are the key questions I should ask about chest pain?",
            "What diagnostic tests would be most helpful?",
            "What are the most likely diagnoses?"
        ]
        
        results = await ws_client.diagnostic_session(questions, budget=50.0)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {result['question']}")
            print(f"A: {result['response']}")
            print(f"Cost: ${result['cost']:.2f}")
            print(f"Total: ${result['total_spent']:.2f}")
            print(f"Remaining: ${result['remaining_budget']:.2f}")
        
        # Logout
        await rest_client.logout()

asyncio.run(complete_example())
```

## JavaScript/Node.js Examples

### Node.js Client with axios and ws

```javascript
const axios = require('axios');
const WebSocket = require('ws');

class Dx0Client {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.wsUrl = baseUrl.replace('http', 'ws');
        this.accessToken = null;
        this.refreshToken = null;
        
        // Setup axios instance with interceptors
        this.http = axios.create({
            baseURL: `${baseUrl}/api/v1`
        });
        
        // Request interceptor to add auth header
        this.http.interceptors.request.use(config => {
            if (this.accessToken) {
                config.headers.Authorization = `Bearer ${this.accessToken}`;
            }
            return config;
        });
        
        // Response interceptor for token refresh
        this.http.interceptors.response.use(
            response => response,
            async error => {
                const originalRequest = error.config;
                
                if (error.response?.status === 401 && !originalRequest._retry) {
                    originalRequest._retry = true;
                    
                    if (await this.refreshTokens()) {
                        originalRequest.headers.Authorization = `Bearer ${this.accessToken}`;
                        return this.http(originalRequest);
                    }
                }
                
                return Promise.reject(error);
            }
        );
    }
    
    async login(username, password) {
        try {
            const response = await this.http.post('/login', {
                username,
                password
            });
            
            this.accessToken = response.data.access_token;
            this.refreshToken = response.data.refresh_token;
            return true;
        } catch (error) {
            if (error.response?.status === 401) {
                throw new Error('Invalid credentials');
            } else if (error.response?.status === 429) {
                throw new Error('Too many failed login attempts');
            }
            throw error;
        }
    }
    
    async refreshTokens() {
        if (!this.refreshToken) return false;
        
        try {
            const response = await axios.post(`${this.baseUrl}/api/v1/refresh`, {
                refresh_token: this.refreshToken
            });
            
            this.accessToken = response.data.access_token;
            this.refreshToken = response.data.refresh_token;
            return true;
        } catch (error) {
            return false;
        }
    }
    
    async getCase() {
        const response = await this.http.get('/case');
        return response.data;
    }
    
    async getTests() {
        const response = await this.http.get('/tests');
        return response.data;
    }
    
    async exportTranscriptFHIR(transcript, patientId = 'example') {
        const response = await this.http.post('/fhir/transcript', {
            transcript,
            patient_id: patientId
        });
        return response.data;
    }
    
    async exportTestsFHIR(tests, patientId = 'example') {
        const response = await this.http.post('/fhir/tests', {
            tests,
            patient_id: patientId
        });
        return response.data;
    }
    
    createWebSocket(budget = null) {
        if (!this.accessToken) {
            throw new Error('Access token required');
        }
        
        let url = `${this.wsUrl}/api/v1/ws?token=${this.accessToken}`;
        if (budget !== null) {
            url += `&budget=${budget}`;
        }
        
        return new WebSocket(url);
    }
    
    async sendDiagnosticMessage(ws, action, content) {
        return new Promise((resolve, reject) => {
            const message = { action: action.toUpperCase(), content };
            let completeReply = '';
            let finalData = {};
            
            const messageHandler = (data) => {
                try {
                    const response = JSON.parse(data);
                    
                    if (response.error) {
                        reject(new Error(`WebSocket error: ${response.error}`));
                        return;
                    }
                    
                    completeReply += response.reply || '';
                    
                    if (response.done) {
                        finalData = { ...response, complete_reply: completeReply };
                        ws.off('message', messageHandler);
                        resolve(finalData);
                    }
                } catch (error) {
                    reject(error);
                }
            };
            
            ws.on('message', messageHandler);
            ws.send(JSON.stringify(message));
        });
    }
    
    async logout() {
        if (this.refreshToken) {
            try {
                await this.http.post('/logout', {
                    refresh_token: this.refreshToken
                });
            } catch (error) {
                // Logout errors are not critical
            } finally {
                this.accessToken = null;
                this.refreshToken = null;
            }
        }
    }
}

// Usage Example
async function main() {
    const client = new Dx0Client();
    
    try {
        // Login
        await client.login('your_username', 'your_password');
        console.log('✅ Logged in successfully');
        
        // Get case information
        const caseInfo = await client.getCase();
        console.log(`📋 Current case: ${caseInfo.summary}`);
        
        // WebSocket diagnostic session
        const ws = client.createWebSocket(100.0);
        
        ws.on('open', async () => {
            console.log('🔌 WebSocket connected');
            
            try {
                // Ask a diagnostic question
                const response = await client.sendDiagnosticMessage(
                    ws, 
                    'QUESTION', 
                    'What should I ask about chest pain?'
                );
                
                console.log(`🤖 AI Response: ${response.complete_reply}`);
                console.log(`💰 Cost: $${response.cost}`);
                console.log(`💳 Total Spent: $${response.total_spent}`);
                
                ws.close();
            } catch (error) {
                console.error('❌ WebSocket error:', error.message);
                ws.close();
            }
        });
        
        ws.on('error', (error) => {
            console.error('❌ WebSocket connection error:', error);
        });
        
        ws.on('close', async () => {
            console.log('🔌 WebSocket disconnected');
            await client.logout();
            console.log('👋 Logged out');
        });
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

main().catch(console.error);
```

### Browser JavaScript Client

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dx0 API Browser Client</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #e3f2fd; }
        .ai-message { background-color: #f3e5f5; }
        .error-message { background-color: #ffebee; color: #c62828; }
        .status { background-color: #e8f5e8; color: #2e7d32; }
        textarea { width: 100%; height: 100px; }
        button { padding: 10px 20px; margin: 5px; }
        #budget-info { background-color: #fff3e0; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Dx0 Diagnostic Assistant</h1>
    
    <div id="auth-section">
        <h2>Authentication</h2>
        <input type="text" id="username" placeholder="Username" />
        <input type="password" id="password" placeholder="Password" />
        <button onclick="login()">Login</button>
        <button onclick="logout()">Logout</button>
    </div>
    
    <div id="case-info" style="display: none;">
        <h2>Case Information</h2>
        <div id="case-summary"></div>
        <div id="available-tests"></div>
    </div>
    
    <div id="diagnostic-section" style="display: none;">
        <h2>Diagnostic Conversation</h2>
        <div id="budget-info"></div>
        <div id="messages"></div>
        <textarea id="message-input" placeholder="Ask a diagnostic question..."></textarea>
        <br>
        <button onclick="sendMessage('QUESTION')">Ask Question</button>
        <button onclick="sendMessage('TEST')">Order Test</button>
        <button onclick="sendMessage('DIAGNOSIS')">Discuss Diagnosis</button>
    </div>

    <script>
        class Dx0BrowserClient {
            constructor() {
                this.baseUrl = 'http://localhost:8000';
                this.wsUrl = 'ws://localhost:8000';
                this.accessToken = null;
                this.refreshToken = null;
                this.websocket = null;
                this.totalSpent = 0;
                this.remainingBudget = null;
            }
            
            async login(username, password) {
                try {
                    const response = await fetch(`${this.baseUrl}/api/v1/login`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username, password })
                    });
                    
                    if (response.ok) {
                        const tokens = await response.json();
                        this.accessToken = tokens.access_token;
                        this.refreshToken = tokens.refresh_token;
                        this.showMessage('✅ Logged in successfully', 'status');
                        return true;
                    } else {
                        const error = await response.json();
                        throw new Error(error.detail || 'Login failed');
                    }
                } catch (error) {
                    this.showMessage(`❌ Login error: ${error.message}`, 'error-message');
                    return false;
                }
            }
            
            async refreshTokens() {
                if (!this.refreshToken) return false;
                
                try {
                    const response = await fetch(`${this.baseUrl}/api/v1/refresh`, {
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
                } catch (error) {
                    console.error('Token refresh failed:', error);
                }
                return false;
            }
            
            async apiCall(endpoint, options = {}) {
                if (!this.accessToken) throw new Error('Not authenticated');
                
                const headers = {
                    'Authorization': `Bearer ${this.accessToken}`,
                    'Content-Type': 'application/json',
                    ...options.headers
                };
                
                let response = await fetch(`${this.baseUrl}/api/v1${endpoint}`, {
                    ...options,
                    headers
                });
                
                if (response.status === 401) {
                    if (await this.refreshTokens()) {
                        headers['Authorization'] = `Bearer ${this.accessToken}`;
                        response = await fetch(`${this.baseUrl}/api/v1${endpoint}`, {
                            ...options,
                            headers
                        });
                    }
                }
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'API call failed');
                }
                
                return response.json();
            }
            
            async loadCaseInfo() {
                try {
                    const caseData = await this.apiCall('/case');
                    const testsData = await this.apiCall('/tests');
                    
                    document.getElementById('case-summary').innerHTML = `
                        <strong>Case:</strong> ${caseData.summary}
                    `;
                    document.getElementById('available-tests').innerHTML = `
                        <strong>Available Tests:</strong> ${testsData.tests.join(', ')}
                    `;
                    document.getElementById('case-info').style.display = 'block';
                } catch (error) {
                    this.showMessage(`❌ Error loading case info: ${error.message}`, 'error-message');
                }
            }
            
            connectWebSocket() {
                if (!this.accessToken) return;
                
                const url = `${this.wsUrl}/api/v1/ws?token=${this.accessToken}&budget=100.0`;
                this.websocket = new WebSocket(url);
                
                this.websocket.onopen = () => {
                    this.showMessage('🔌 Connected to diagnostic system', 'status');
                    document.getElementById('diagnostic-section').style.display = 'block';
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.error) {
                        this.showMessage(`❌ Error: ${data.error}`, 'error-message');
                        return;
                    }
                    
                    if (data.reply) {
                        this.showMessage(data.reply, 'ai-message');
                    }
                    
                    if (data.done) {
                        this.totalSpent = data.total_spent || this.totalSpent;
                        this.remainingBudget = data.remaining_budget;
                        this.updateBudgetInfo();
                        
                        if (data.ordered_tests && data.ordered_tests.length > 0) {
                            this.showMessage(`🧪 Ordered tests: ${data.ordered_tests.join(', ')}`, 'status');
                        }
                    }
                };
                
                this.websocket.onerror = (error) => {
                    this.showMessage('❌ WebSocket connection error', 'error-message');
                };
                
                this.websocket.onclose = () => {
                    this.showMessage('🔌 Disconnected from diagnostic system', 'status');
                };
            }
            
            sendMessage(action) {
                const input = document.getElementById('message-input');
                const content = input.value.trim();
                
                if (!content || !this.websocket) return;
                
                this.showMessage(`You (${action}): ${content}`, 'user-message');
                
                this.websocket.send(JSON.stringify({
                    action: action,
                    content: content
                }));
                
                input.value = '';
            }
            
            showMessage(text, className) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${className}`;
                messageDiv.textContent = text;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            updateBudgetInfo() {
                const budgetDiv = document.getElementById('budget-info');
                budgetDiv.innerHTML = `
                    💰 Total Spent: $${this.totalSpent.toFixed(2)} | 
                    💳 Remaining Budget: $${this.remainingBudget ? this.remainingBudget.toFixed(2) : 'N/A'}
                `;
            }
            
            async logout() {
                if (this.websocket) {
                    this.websocket.close();
                }
                
                if (this.refreshToken) {
                    try {
                        await this.apiCall('/logout', {
                            method: 'POST',
                            body: JSON.stringify({ refresh_token: this.refreshToken })
                        });
                    } catch (error) {
                        // Logout errors are not critical
                    }
                }
                
                this.accessToken = null;
                this.refreshToken = null;
                
                document.getElementById('case-info').style.display = 'none';
                document.getElementById('diagnostic-section').style.display = 'none';
                document.getElementById('messages').innerHTML = '';
                
                this.showMessage('👋 Logged out successfully', 'status');
            }
        }
        
        const client = new Dx0BrowserClient();
        
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (await client.login(username, password)) {
                await client.loadCaseInfo();
                client.connectWebSocket();
            }
        }
        
        function logout() {
            client.logout();
        }
        
        function sendMessage(action) {
            client.sendMessage(action);
        }
        
        // Allow Enter key to send messages
        document.getElementById('message-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage('QUESTION');
            }
        });
    </script>
</body>
</html>
```

## cURL Examples

### Authentication Flow

```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:8000"
USERNAME="your_username"
PASSWORD="your_password"

# Login and get tokens
echo "🔐 Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

# Extract tokens
ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')
REFRESH_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.refresh_token')

if [ "$ACCESS_TOKEN" = "null" ]; then
    echo "❌ Login failed"
    echo $LOGIN_RESPONSE | jq .
    exit 1
fi

echo "✅ Login successful"

# Use access token for API calls
echo "📋 Getting case information..."
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "$BASE_URL/api/v1/case" | jq .

echo "🧪 Getting available tests..."
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "$BASE_URL/api/v1/tests" | jq .

# Refresh tokens
echo "🔄 Refreshing tokens..."
REFRESH_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}")

NEW_ACCESS_TOKEN=$(echo $REFRESH_RESPONSE | jq -r '.access_token')
NEW_REFRESH_TOKEN=$(echo $REFRESH_RESPONSE | jq -r '.refresh_token')

echo "✅ Tokens refreshed"

# Logout
echo "👋 Logging out..."
curl -s -X POST "$BASE_URL/api/v1/logout" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"$NEW_REFRESH_TOKEN\"}"

echo "✅ Logged out successfully"
```

### FHIR Export Examples

```bash
#!/bin/bash

# Assuming you have admin access and tokens from previous example

# Export transcript to FHIR
echo "📄 Exporting transcript to FHIR..."
curl -s -X POST "$BASE_URL/api/v1/fhir/transcript" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": [
      ["user", "Patient presents with chest pain"],
      ["assistant", "Can you describe the onset and character of the pain?"],
      ["user", "Sudden onset, sharp, radiating to left arm"]
    ],
    "patient_id": "patient-12345"
  }' | jq .

# Export ordered tests to FHIR
echo "🧪 Exporting ordered tests to FHIR..."
curl -s -X POST "$BASE_URL/api/v1/fhir/tests" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tests": [
      "complete blood count",
      "basic metabolic panel",
      "troponin levels"
    ],
    "patient_id": "patient-12345"
  }' | jq .
```

## Error Handling Patterns

### Python Error Handling

```python
import asyncio
import aiohttp
import logging
from typing import Optional

class Dx0ErrorHandler:
    @staticmethod
    async def handle_api_errors(func, *args, **kwargs):
        """Generic error handler for API calls."""
        try:
            return await func(*args, **kwargs)
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                raise AuthenticationError("Authentication failed")
            elif e.status == 403:
                raise AuthorizationError("Insufficient permissions")
            elif e.status == 429:
                raise RateLimitError("Rate limit exceeded")
            elif e.status >= 500:
                raise ServerError(f"Server error: {e.status}")
            else:
                raise APIError(f"API error: {e.status} - {e.message}")
        except aiohttp.ClientConnectionError:
            raise ConnectionError("Unable to connect to server")
        except asyncio.TimeoutError:
            raise TimeoutError("Request timeout")

class Dx0Exception(Exception):
    """Base exception for Dx0 API errors."""
    pass

class AuthenticationError(Dx0Exception):
    """Authentication failed."""
    pass

class AuthorizationError(Dx0Exception):
    """Insufficient permissions."""
    pass

class RateLimitError(Dx0Exception):
    """Rate limit exceeded."""
    pass

class ServerError(Dx0Exception):
    """Server error."""
    pass

class APIError(Dx0Exception):
    """General API error."""
    pass

class ConnectionError(Dx0Exception):
    """Connection error."""
    pass

# Usage with retry logic
async def robust_api_call(client, operation, max_retries=3, backoff_factor=1.0):
    """Make API call with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await Dx0ErrorHandler.handle_api_errors(operation)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)
            else:
                raise
        except (ConnectionError, ServerError) as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Connection error, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)
            else:
                raise
        except (AuthenticationError, AuthorizationError):
            # Don't retry auth errors
            raise
```

### JavaScript Error Handling

```javascript
class Dx0ErrorHandler {
    static async handleApiCall(apiCall, options = {}) {
        const {
            maxRetries = 3,
            backoffFactor = 1000,
            retryCondition = (error) => error.isRetryable
        } = options;
        
        for (let attempt = 0; attempt < maxRetries; attempt++) {
            try {
                return await apiCall();
            } catch (error) {
                const dx0Error = this.convertError(error);
                
                if (attempt < maxRetries - 1 && retryCondition(dx0Error)) {
                    const waitTime = backoffFactor * Math.pow(2, attempt);
                    console.warn(`Retrying in ${waitTime}ms (attempt ${attempt + 1})`);
                    await this.sleep(waitTime);
                    continue;
                }
                
                throw dx0Error;
            }
        }
    }
    
    static convertError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.detail || error.message;
            
            switch (status) {
                case 401:
                    return new Dx0AuthenticationError(message);
                case 403:
                    return new Dx0AuthorizationError(message);
                case 429:
                    return new Dx0RateLimitError(message);
                case 500:
                case 502:
                case 503:
                case 504:
                    return new Dx0ServerError(message, true); // isRetryable = true
                default:
                    return new Dx0APIError(message, status >= 500);
            }
        } else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            return new Dx0ConnectionError(error.message, true);
        } else {
            return new Dx0APIError(error.message);
        }
    }
    
    static sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

class Dx0Error extends Error {
    constructor(message, isRetryable = false) {
        super(message);
        this.name = this.constructor.name;
        this.isRetryable = isRetryable;
    }
}

class Dx0AuthenticationError extends Dx0Error {}
class Dx0AuthorizationError extends Dx0Error {}
class Dx0RateLimitError extends Dx0Error { 
    constructor(message) { super(message, true); }
}
class Dx0ServerError extends Dx0Error {}
class Dx0ConnectionError extends Dx0Error {}
class Dx0APIError extends Dx0Error {}

// Usage
async function robustApiCall(client) {
    try {
        return await Dx0ErrorHandler.handleApiCall(
            () => client.getCase(),
            {
                maxRetries: 3,
                retryCondition: (error) => error.isRetryable
            }
        );
    } catch (error) {
        if (error instanceof Dx0AuthenticationError) {
            console.error('Authentication failed, redirecting to login...');
            // Handle auth error
        } else if (error instanceof Dx0RateLimitError) {
            console.error('Rate limited, please try again later');
            // Handle rate limit
        } else {
            console.error('API call failed:', error.message);
            // Handle other errors
        }
    }
}
```

## Production Ready Client

Here's a production-ready Python client with all best practices:

```python
import asyncio
import aiohttp
import websockets
import json
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    QUESTION = "QUESTION"
    TEST = "TEST"
    DIAGNOSIS = "DIAGNOSIS"

@dataclass
class DiagnosticResponse:
    reply: str
    cost: Optional[float] = None
    total_spent: Optional[float] = None
    remaining_budget: Optional[float] = None
    ordered_tests: Optional[List[str]] = None

class Dx0ProductionClient:
    """Production-ready Dx0 API client with comprehensive error handling, retry logic, and logging."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Callbacks
        self.on_token_refresh: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement client-side rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    async def _retry_with_backoff(self, coro, *args, **kwargs):
        """Execute coroutine with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                return await coro(*args, **kwargs)
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"Connection error (attempt {attempt + 1}), retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
            except aiohttp.ClientResponseError as e:
                if e.status >= 500 and attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"Server error {e.status} (attempt {attempt + 1}), retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def login(self, username: str, password: str) -> bool:
        """Login with comprehensive error handling."""
        try:
            async def _login():
                async with self.session.post(
                    f"{self.base_url}/api/v1/login",
                    json={"username": username, "password": password}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data["access_token"]
                        self.refresh_token = data["refresh_token"]
                        logger.info("Login successful")
                        return True
                    elif response.status == 401:
                        error_data = await response.json()
                        raise ValueError(f"Invalid credentials: {error_data.get('detail', 'Unknown error')}")
                    elif response.status == 429:
                        error_data = await response.json()
                        raise ValueError(f"Rate limited: {error_data.get('detail', 'Too many attempts')}")
                    else:
                        error_data = await response.json()
                        raise RuntimeError(f"Login failed: {error_data.get('detail', f'Status {response.status}')}")
            
            return await self._retry_with_backoff(_login)
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            if self.on_error:
                await self.on_error("login", e)
            raise
    
    async def refresh_tokens(self) -> bool:
        """Refresh access token with proper error handling."""
        if not self.refresh_token:
            return False
        
        try:
            async def _refresh():
                async with self.session.post(
                    f"{self.base_url}/api/v1/refresh",
                    json={"refresh_token": self.refresh_token}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data["access_token"]
                        self.refresh_token = data["refresh_token"]
                        logger.info("Tokens refreshed successfully")
                        if self.on_token_refresh:
                            await self.on_token_refresh(data)
                        return True
                    else:
                        logger.warning("Token refresh failed")
                        return False
            
            return await self._retry_with_backoff(_refresh)
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    async def _make_authenticated_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request with automatic token refresh."""
        if not self.access_token:
            raise RuntimeError("Not authenticated")
        
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.access_token}"
        kwargs["headers"] = headers
        
        async def _request():
            async with self.session.request(
                method, 
                f"{self.base_url}/api/v1{endpoint}", 
                **kwargs
            ) as response:
                if response.status == 401:
                    # Try token refresh
                    if await self.refresh_tokens():
                        headers["Authorization"] = f"Bearer {self.access_token}"
                        async with self.session.request(
                            method, 
                            f"{self.base_url}/api/v1{endpoint}", 
                            **kwargs
                        ) as retry_response:
                            if retry_response.status >= 400:
                                error_data = await retry_response.json()
                                raise aiohttp.ClientResponseError(
                                    retry_response.request_info,
                                    retry_response.history,
                                    status=retry_response.status,
                                    message=error_data.get('detail', 'Request failed')
                                )
                            return await retry_response.json()
                    else:
                        raise RuntimeError("Authentication failed and token refresh failed")
                
                if response.status >= 400:
                    error_data = await response.json()
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_data.get('detail', 'Request failed')
                    )
                
                return await response.json()
        
        return await self._retry_with_backoff(_request)
    
    async def get_case(self) -> Dict[str, Any]:
        """Get current case with error handling."""
        return await self._make_authenticated_request("GET", "/case")
    
    async def get_tests(self) -> Dict[str, Any]:
        """Get available tests with error handling."""
        return await self._make_authenticated_request("GET", "/tests")
    
    async def diagnostic_conversation(
        self, 
        messages: List[tuple[ActionType, str]], 
        budget: Optional[float] = None,
        message_callback: Optional[Callable[[str, DiagnosticResponse], None]] = None
    ) -> List[DiagnosticResponse]:
        """
        Conduct a diagnostic conversation via WebSocket with comprehensive error handling.
        
        Args:
            messages: List of (action_type, content) tuples
            budget: Optional budget limit
            message_callback: Optional callback for each response
        
        Returns:
            List of diagnostic responses
        """
        if not self.access_token:
            raise RuntimeError("Not authenticated")
        
        results = []
        max_ws_retries = 3
        
        for ws_attempt in range(max_ws_retries):
            try:
                uri = f"{self.ws_url}/api/v1/ws?token={self.access_token}"
                if budget is not None:
                    uri += f"&budget={budget}"
                
                async with websockets.connect(uri) as websocket:
                    logger.info("WebSocket connected")
                    
                    for action, content in messages:
                        try:
                            # Send message
                            message = {"action": action.value, "content": content}
                            await websocket.send(json.dumps(message))
                            
                            # Receive complete response
                            response = await self._receive_complete_response(websocket)
                            results.append(response)
                            
                            if message_callback:
                                message_callback(content, response)
                            
                            logger.info(f"Processed message: {action.value}")
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("WebSocket connection closed unexpectedly")
                            raise
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            raise
                    
                    logger.info("Diagnostic conversation completed successfully")
                    return results
                    
            except websockets.exceptions.InvalidStatusCode as e:
                if e.status_code == 1008:  # Authentication error
                    logger.warning("WebSocket authentication failed, refreshing token")
                    if await self.refresh_tokens() and ws_attempt < max_ws_retries - 1:
                        continue
                    else:
                        raise RuntimeError("WebSocket authentication failed")
                else:
                    raise
            except Exception as e:
                if ws_attempt < max_ws_retries - 1:
                    wait_time = self.backoff_factor * (2 ** ws_attempt)
                    logger.warning(f"WebSocket error (attempt {ws_attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"WebSocket failed after {max_ws_retries} attempts: {e}")
                    if self.on_error:
                        await self.on_error("websocket", e)
                    raise
        
        return results
    
    async def _receive_complete_response(self, websocket) -> DiagnosticResponse:
        """Receive a complete response from WebSocket."""
        complete_reply = ""
        final_data = {}
        
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(response)
                
                if 'error' in data:
                    raise RuntimeError(f"WebSocket error: {data['error']}")
                
                complete_reply += data.get('reply', '')
                
                if data.get('done'):
                    final_data = data
                    break
                    
            except asyncio.TimeoutError:
                raise RuntimeError("WebSocket response timeout")
            except websockets.exceptions.ConnectionClosed:
                raise RuntimeError("WebSocket connection closed")
        
        return DiagnosticResponse(
            reply=complete_reply,
            cost=final_data.get('cost'),
            total_spent=final_data.get('total_spent'),
            remaining_budget=final_data.get('remaining_budget'),
            ordered_tests=final_data.get('ordered_tests')
        )
    
    async def export_transcript_fhir(self, transcript: List[tuple[str, str]], patient_id: str = "example") -> Dict[str, Any]:
        """Export transcript to FHIR with error handling."""
        return await self._make_authenticated_request(
            "POST", 
            "/fhir/transcript",
            json={"transcript": transcript, "patient_id": patient_id}
        )
    
    async def export_tests_fhir(self, tests: List[str], patient_id: str = "example") -> Dict[str, Any]:
        """Export tests to FHIR with error handling."""
        return await self._make_authenticated_request(
            "POST", 
            "/fhir/tests",
            json={"tests": tests, "patient_id": patient_id}
        )
    
    async def logout(self):
        """Logout with proper cleanup."""
        if self.refresh_token:
            try:
                await self._make_authenticated_request(
                    "POST", 
                    "/logout", 
                    json={"refresh_token": self.refresh_token}
                )
                logger.info("Logout successful")
            except Exception as e:
                logger.warning(f"Logout error (non-critical): {e}")
            finally:
                self.access_token = None
                self.refresh_token = None

# Usage Example
async def production_example():
    """Comprehensive production usage example."""
    
    async def token_refresh_callback(tokens):
        """Handle token refresh events."""
        logger.info("Tokens refreshed, saving to secure storage...")
        # Save tokens to secure storage
    
    async def error_callback(operation, error):
        """Handle errors with logging/monitoring."""
        logger.error(f"Operation {operation} failed: {error}")
        # Send to monitoring system
    
    def message_callback(content, response):
        """Handle each diagnostic response."""
        print(f"Q: {content}")
        print(f"A: {response.reply}")
        print(f"Cost: ${response.cost or 0:.2f}")
        print("---")
    
    async with Dx0ProductionClient(
        base_url="http://localhost:8000",
        timeout=30,
        max_retries=3,
        backoff_factor=1.0
    ) as client:
        # Set callbacks
        client.on_token_refresh = token_refresh_callback
        client.on_error = error_callback
        
        try:
            # Login
            await client.login("your_username", "your_password")
            
            # Get case info
            case_info = await client.get_case()
            print(f"Case: {case_info['summary']}")
            
            # Diagnostic conversation
            messages = [
                (ActionType.QUESTION, "What should I ask about chest pain?"),
                (ActionType.TEST, "complete blood count"),
                (ActionType.DIAGNOSIS, "What are the likely diagnoses?")
            ]
            
            responses = await client.diagnostic_conversation(
                messages=messages,
                budget=100.0,
                message_callback=message_callback
            )
            
            print(f"Total responses: {len(responses)}")
            print(f"Final budget: ${responses[-1].remaining_budget or 0:.2f}")
            
            # Export to FHIR (if admin user)
            try:
                transcript = [("user", "Chest pain"), ("assistant", "Tell me more")]
                fhir_bundle = await client.export_transcript_fhir(transcript, "patient-123")
                print("FHIR export successful")
            except Exception as e:
                print(f"FHIR export failed (may require admin access): {e}")
            
        except Exception as e:
            logger.error(f"Example failed: {e}")
            raise
        finally:
            await client.logout()

# Run the production example
if __name__ == "__main__":
    asyncio.run(production_example())
```

These examples provide comprehensive, production-ready code for integrating with the Dx0 API across multiple programming languages and scenarios. Each example includes proper error handling, retry logic, rate limiting, and security best practices.