function App() {
  const [token, setToken] = React.useState(null);
  const [summary, setSummary] = React.useState('');
  const [log, setLog] = React.useState([]);
  const [msg, setMsg] = React.useState('');
  const [action, setAction] = React.useState('question');
  const [ws, setWs] = React.useState(null);
  const [cost, setCost] = React.useState(0);
  const [stepCost, setStepCost] = React.useState(0);
  const [tests, setTests] = React.useState([]);
  const [flow, setFlow] = React.useState([]);
  const [availableTests, setAvailableTests] = React.useState([]);
  const [loadingCase, setLoadingCase] = React.useState(false);
  const [loadingReply, setLoadingReply] = React.useState(false);
  const [toast, setToast] = React.useState('');

  React.useEffect(() => {
    if (toast) {
      const id = setTimeout(() => setToast(''), 3000);
      return () => clearTimeout(id);
    }
  }, [toast]);

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
        setLoadingCase(true);
        const caseRes = await fetch('/case');
        if (caseRes.ok) {
          const caseData = await caseRes.json();
          setSummary(caseData.summary);
        } else {
          setSummary('Unable to load case summary.');
          setToast('Failed to load case data');
        }
      } catch (err) {
        console.error('Failed to fetch case:', err);
        setToast('Failed to load case data');
        setSummary('Unable to load case summary.');
      } finally {
        setLoadingCase(false);
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
          setStepCost(d.cost || 0);
          if (d.ordered_tests) setTests(d.ordered_tests);
          setFlow(f => [...f, {sender: 'Gatekeeper', text: msgText}]);
          setLoadingReply(false);
        }
      };
      socket.onclose = () => {
        setToast('WebSocket disconnected');
        setWs(null);
      };
      setWs(socket);
    } else {
      let text = 'Login failed';
      try {
        const data = await res.json();
        if (data.detail) text += `: ${data.detail}`;
      } catch {}
      setToast(text);
    }
  };

  const send = () => {
    if (!ws) return;
    const content = msg;
    setLog(l => [...l, {sender: 'You', text: content}]);
    setFlow(f => [...f, {sender: 'You', text: content}]);
    ws.send(JSON.stringify({action, content}));
    setMsg('');
    setStepCost(0);
    setLoadingReply(true);
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
        <div>{loadingCase ? <span className='spinner'></span> : summary}</div>
      </div>
      <div id='tests-panel'>
        <h3>Ordered Tests</h3>
        <ul>{tests.map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
      <div id='chat-panel'>
        <h2>SDBench Physician Chat</h2>
        <div>Step Cost: ${stepCost.toFixed(2)} | Total Cost: ${cost.toFixed(2)}</div>
        <div id='log'>
          {log.map((m, i) => <div key={i}><b>{m.sender}:</b> {m.text}</div>)}
          {loadingReply && <span className='spinner'></span>}
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
      {toast && <div id='toast'>{toast}</div>}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
