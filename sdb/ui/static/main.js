function App() {
  const [token, setToken] = React.useState(null);
  const [summary, setSummary] = React.useState('');
  const [log, setLog] = React.useState([]);
  const [msg, setMsg] = React.useState('');
  const [action, setAction] = React.useState('question');
  const [ws, setWs] = React.useState(null);
  const [cost, setCost] = React.useState(0);
  const [tests, setTests] = React.useState([]);
  const [flow, setFlow] = React.useState([]);
  const [availableTests, setAvailableTests] = React.useState([]);

  React.useEffect(() => {
    if (token) {
      fetch('/tests')
        .then(res => res.ok ? res.json() : {tests: []})
        .then(data => setAvailableTests(data.tests || []))
        .catch(() => setAvailableTests([]));
    }
  }, [token]);

  const login = async (e) => {
    e.preventDefault();
    const user = e.target.user.value;
    const pass = e.target.pass.value;
    const res = await fetch('/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username: user, password: pass})
    });
    if (res.ok) {
      const data = await res.json();
      setToken(data.token);
      try {
        const caseRes = await fetch('/case');
        if (caseRes.ok) {
          const caseData = await caseRes.json();
          setSummary(caseData.summary);
        } else {
          setSummary('Unable to load case summary.');
        }
      } catch (err) {
        console.error('Failed to fetch case:', err);
        setSummary('Unable to load case summary.');
      }
      const socket = new WebSocket(`ws://${location.host}/ws?token=${data.token}`);
      socket.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        let msgText = '';
        setLog(l => {
          const log = [...l];
          if (
            log.length === 0 ||
            log[log.length - 1].sender !== 'Gatekeeper' ||
            log[log.length - 1].done
          ) {
            log.push({sender: 'Gatekeeper', text: d.reply, done: d.done});
            msgText = d.reply;
          } else {
            log[log.length - 1] = {
              ...log[log.length - 1],
              text: log[log.length - 1].text + d.reply,
              done: d.done
            };
            msgText = log[log.length - 1].text;
          }
          return log;
        });
        if (d.done) {
          setCost(d.total_spent);
          if (d.ordered_tests) setTests(d.ordered_tests);
          setFlow(f => [...f, {sender: 'Gatekeeper', text: msgText}]);
        }
      };
      setWs(socket);
    } else {
      alert('Login failed');
    }
  };

  const send = () => {
    if (!ws) return;
    const content = msg;
    setLog(l => [...l, {sender: 'You', text: content}]);
    setFlow(f => [...f, {sender: 'You', text: content}]);
    ws.send(JSON.stringify({action, content}));
    setMsg('');
  };

  if (!token) {
    return (
      <form onSubmit={login}>
        <input name='user' placeholder='Username'/>
        <input name='pass' type='password' placeholder='Password'/>
        <button type='submit'>Login</button>
      </form>
    );
  }

  return (
    <div id='layout'>
      <div id='summary-panel'>
        <h3>Case Summary</h3>
        <div>{summary}</div>
      </div>
      <div id='tests-panel'>
        <h3>Ordered Tests</h3>
        <ul>{tests.map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
      <div id='chat-panel'>
        <h2>SDBench Physician Chat</h2>
        <div>Running Cost: ${cost.toFixed(2)}</div>
        <div id='log'>
          {log.map((m, i) => <div key={i}><b>{m.sender}:</b> {m.text}</div>)}
        </div>
        <div>
          <select value={action} onChange={e => setAction(e.target.value)}>
            <option value='question'>Ask Question</option>
            <option value='test'>Order Test</option>
            <option value='diagnosis'>Provide Diagnosis</option>
          </select>
          <input
            value={msg}
            onChange={e => setMsg(e.target.value)}
            size='80'
            list={action === 'test' ? 'tests-list' : undefined}
            placeholder={action === 'test' ? 'Search tests' : ''}
          />
          <datalist id='tests-list'>
            {availableTests.map((t, i) => <option key={i} value={t} />)}
          </datalist>
          <button onClick={send}>Send</button>
        </div>
      </div>
      <div id='flow-panel'>
        <h3>Diagnostic Flow</h3>
        <ol>{flow.map((m, i) => <li key={i}>{m.sender}: {m.text}</li>)}</ol>
      </div>
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
