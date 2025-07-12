import { useEffect, useRef, useState } from 'react'
import CollapsiblePanel from './CollapsiblePanel.jsx'
import './App.css'

export default function App() {
  const [token, setToken] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [summary, setSummary] = useState('')
  const [results, setResults] = useState([])
  const [message, setMessage] = useState('')
  const [log, setLog] = useState([])
  const wsRef = useRef(null)

  useEffect(() => {
    if (token) {
      fetch('/api/v1/case')
        .then(r => r.ok ? r.json() : {summary: ''})
        .then(d => setSummary(d.summary || ''))
      const ws = new WebSocket(`/api/v1/ws?token=${token}`)
      ws.onmessage = ev => {
        const data = JSON.parse(ev.data)
        if (data.error) return
        setLog(l => [...l, {sender: 'Gatekeeper', text: data.reply}])
        if (data.ordered_tests) setResults(data.ordered_tests)
      }
      wsRef.current = ws
      return () => ws.close()
    }
  }, [token])

  const handleLogin = async (e) => {
    e.preventDefault()
    const res = await fetch('/api/v1/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    })
    if (res.ok) {
      const data = await res.json()
      setToken(data.token)
    }
  }

  const send = () => {
    if (!wsRef.current) return
    wsRef.current.send(JSON.stringify({action: 'question', content: message}))
    setLog(l => [...l, {sender: 'You', text: message}])
    setMessage('')
  }

  if (!token) {
    return (
      <form onSubmit={handleLogin}>
        <div>
          <label>Username <input value={username} onChange={e => setUsername(e.target.value)} /></label>
        </div>
        <div>
          <label>Password <input type="password" value={password} onChange={e => setPassword(e.target.value)} /></label>
        </div>
        <button type="submit">Login</button>
      </form>
    )
  }

  return (
    <div id="root">
      <h2>Gatekeeper Chat</h2>
      <CollapsiblePanel title="Case Summary">
        <p>{summary}</p>
      </CollapsiblePanel>
      <CollapsiblePanel title="Chat">
        <div className="log">
          {log.map((m, i) => <div key={i}><b>{m.sender}:</b> {m.text}</div>)}
        </div>
        <input value={message} onChange={e => setMessage(e.target.value)} />
        <button onClick={send}>Send</button>
      </CollapsiblePanel>
      <CollapsiblePanel title="Results">
        <ul>
          {results.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </CollapsiblePanel>
    </div>
  )
}
