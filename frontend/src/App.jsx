import { useState, useEffect, useRef } from 'react'
import Collapsible from './Collapsible'
import './App.css'

export default function App() {
  const [token, setToken] = useState('')
  const [summary, setSummary] = useState('')
  const [msg, setMsg] = useState('')
  const [log, setLog] = useState([])
  const [tests, setTests] = useState([])
  const wsRef = useRef(null)

  useEffect(() => {
    if (!token) return
    fetch('/api/v1/case')
      .then(r => r.json())
      .then(d => setSummary(d.summary))
      .catch(() => setSummary(''))
    const ws = new WebSocket(`ws://${location.host}/api/v1/ws?token=${token}`)
    ws.onmessage = ev => {
      const data = JSON.parse(ev.data)
      setLog(l => [...l, { sender: 'Gatekeeper', text: data.reply }])
      if (data.ordered_tests) setTests(data.ordered_tests)
    }
    ws.onclose = () => { wsRef.current = null }
    wsRef.current = ws
    return () => ws.close()
  }, [token])

  const send = () => {
    if (!wsRef.current) return
    wsRef.current.send(JSON.stringify({ action: 'question', content: msg }))
    setLog(l => [...l, { sender: 'You', text: msg }])
    setMsg('')
  }

  const login = async e => {
    e.preventDefault()
    const user = e.target.user.value
    const pass = e.target.pass.value
    const res = await fetch('/api/v1/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass })
    })
    if (res.ok) {
      const data = await res.json()
      setToken(data.token)
    }
  }

  if (!token) {
    return (
      <form onSubmit={login}>
        <input name="user" placeholder="Username" />
        <input type="password" name="pass" placeholder="Password" />
        <button type="submit">Login</button>
      </form>
    )
  }

  return (
    <div>
      <Collapsible title="Case Summary">
        <p>{summary}</p>
      </Collapsible>
      <Collapsible title="Chat">
        <div id="chat-log">
          {log.map((m, i) => (
            <div key={i}><b>{m.sender}:</b> {m.text}</div>
          ))}
        </div>
        <input value={msg} onChange={e => setMsg(e.target.value)} />
        <button onClick={send}>Send</button>
      </Collapsible>
      <Collapsible title="Results">
        <ul>
          {tests.map((t, i) => <li key={i}>{t}</li>)}
        </ul>
      </Collapsible>
    </div>
  )
}
