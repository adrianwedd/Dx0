function App() {
  const [token, setToken] = React.useState(null);
  const [summary, setSummary] = React.useState('');
  const [log, setLog] = React.useState([]);
  const [msg, setMsg] = React.useState('');
  const [action, setAction] = React.useState('question');
  const [ws, setWs] = React.useState(null);
  const [cost, setCost] = React.useState(0);
  const [stepCost, setStepCost] = React.useState(0);
  const [remaining, setRemaining] = React.useState(null);
  const [tests, setTests] = React.useState([]);
  const [flow, setFlow] = React.useState([]);
  const [availableTests, setAvailableTests] = React.useState([]);
  const [loadingCase, setLoadingCase] = React.useState(false);
  const [loadingReply, setLoadingReply] = React.useState(false);
  const [loadingLogin, setLoadingLogin] = React.useState(false);
  const [toast, setToast] = React.useState('');
  const [username, setUsername] = React.useState('');
  const [showCommands, setShowCommands] = React.useState(false);
  const chartRef = React.useRef(null);

  React.useEffect(() => {
    if (toast) {
      const id = setTimeout(() => setToast(''), 3000);
      return () => clearTimeout(id);
    }
  }, [toast]);

  React.useEffect(() => {
    if (token) {
      fetch('/api/v1/tests')
        .then(res => res.ok ? res.json() : {tests: []})
        .then(data => setAvailableTests(data.tests || []))
        .catch(() => setAvailableTests([]));
      if (!chartRef.current && window.Chart) {
        const ctx = document.getElementById('cost-chart');
        if (ctx) {
          chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
              labels: [0],
              datasets: [{
                label: 'Cumulative Cost',
                data: [0],
                borderColor: 'blue',
                fill: false,
                tension: 0.1,
              }]
            },
            options: {
              animation: false,
              scales: {
                x: { title: { display: true, text: 'Step' } },
                y: { title: { display: true, text: 'Cost ($)' } },
              }
            }
          });
        }
      }
    }
  }, [token]);

  const login = async (e) => {
    e.preventDefault();
    setLoadingLogin(true);
    const user = e.target.user.value;
    const pass = e.target.pass.value;
    try {
      const res = await fetch('/api/v1/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: user, password: pass})
      });
      if (res.ok) {
        const data = await res.json();
        setToken(data.token);
        setUsername(user);
      try {
        setLoadingCase(true);
        const caseRes = await fetch('/api/v1/case');
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
      const socket = new WebSocket(`ws://${location.host}/api/v1/ws?token=${data.token}`);
      socket.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        if (d.error) {
          setToast('Validation error');
          return;
        }
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
          if (typeof d.remaining_budget === 'number') {
            setRemaining(d.remaining_budget);
          }
          if (chartRef.current && typeof d.total_spent === 'number') {
            const chart = chartRef.current;
            chart.data.labels.push(chart.data.labels.length);
            chart.data.datasets[0].data.push(d.total_spent);
            chart.update();
          }
          if (d.ordered_tests) setTests(d.ordered_tests);
          setFlow(f => [...f, {sender: 'Gatekeeper', text: msgText}]);
          setLoadingReply(false);
        }
      };
      socket.onerror = () => {
        setToast('WebSocket error');
      };
      socket.onclose = (ev) => {
        const reason = ev.reason || 'connection closed';
        setToast(`WebSocket disconnected (${ev.code}): ${reason}`);
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
    } catch (err) {
      setToast(`Login error: ${err.message}`);
    } finally {
      setLoadingLogin(false);
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

  const logout = async () => {
    if (!token) return;
    try {
      await fetch('/api/v1/logout', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token})
      });
    } catch (err) {
      console.error('Logout failed', err);
    }
    if (ws) ws.close();
    setWs(null);
    setToken(null);
    setUsername('');
    setLog([]);
    setFlow([]);
    setTests([]);
    setCost(0);
    setRemaining(null);
  };

  if (!token) {
      return (
      <form onSubmit={login} className='m-3' role='form'>
        <div className='mb-2'>
          <label htmlFor='username' className='form-label'>Username</label>
          <input
            id='username'
            name='user'
            className='form-control'
            placeholder='Username'
            aria-label='Username'
          />
        </div>
        <div className='mb-2'>
          <label htmlFor='password' className='form-label'>Password</label>
          <input
            id='password'
            name='pass'
            type='password'
            className='form-control'
            placeholder='Password'
            aria-label='Password'
          />
        </div>
        <button type='submit' className='btn btn-primary'>
          {loadingLogin ? (
            <span className='spinner' role='status' aria-label='Loading'></span>
          ) : 'Login'}
        </button>
      </form>
    );
  }

  const limit = remaining !== null ? remaining + cost : null;
  const percent = limit ? (cost / limit) * 100 : 0;

  return (
    <div role='main'>
      <header className='d-flex justify-content-between align-items-center mb-2'>
        <h2>SDBench Physician Chat</h2>
        <div>
          <span className='me-2'>Logged in as {username}</span>
          <button onClick={logout} className='btn btn-secondary btn-sm'>Logout</button>
        </div>
      </header>
      <div id='layout'>
      <div id='summary-panel' className='panel' role='region' aria-label='Case Summary' tabIndex='0'>
        <h3>Case Summary</h3>
        <div>
          {loadingCase ? (
            <span className='skeleton' style={{width: '100%', height: '3em'}}></span>
          ) : (
            summary
          )}
        </div>
      </div>
      <div id='tests-panel' className='panel' role='region' aria-label='Ordered Tests' tabIndex='0'>
        <h3>Ordered Tests</h3>
        <ul>{tests.map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
      <div id='chat-panel' className='panel' role='region' aria-label='Chat Panel' tabIndex='0'>
        <h2>SDBench Physician Chat</h2>
        <div className='mb-2'>
          <strong>Step Cost:</strong> ${stepCost.toFixed(2)}<br/>
          <strong>Total Cost:</strong> ${cost.toFixed(2)}
        </div>
        {limit && (
          <div className='mb-2'>
            <div className='progress'>
              <div className='progress-bar' role='progressbar'
                style={{width: `${percent}%`}}
                aria-valuenow={percent}
                aria-valuemin='0'
                aria-valuemax='100'></div>
            </div>
            <small>Remaining Budget: ${remaining.toFixed(2)} of ${limit.toFixed(2)}</small>
          </div>
        )}
        <div id='log'>
          {log.map((m, i) => <div key={i}><b>{m.sender}:</b> {m.text}</div>)}
          {loadingReply && (
            <span className='skeleton' style={{width: '100%', height: '1em'}}></span>
          )}
        </div>
        <div>
          <select
            value={action}
            onChange={e => setAction(e.target.value)}
            className='form-select mb-2'
            aria-label='Action Select'
          >
            <option value='question'>Ask Question</option>
            <option value='test'>Order Test</option>
            <option value='diagnosis'>Provide Diagnosis</option>
          </select>
          <label htmlFor='msg-input' className='form-label visually-hidden'>Message</label>
          <input
            id='msg-input'
            value={msg}
            onChange={e => setMsg(e.target.value)}
            size='80'
            list={action === 'test' ? 'tests-list' : undefined}
            placeholder={action === 'test' ? 'Search tests' : ''}
            className='form-control mb-2'
            aria-label='Message Input'
          />
          <datalist id='tests-list'>
            {availableTests.map((t, i) => <option key={i} value={t} />)}
          </datalist>
          <button onClick={send} className='btn btn-primary'>Send</button>
        </div>
      </div>
      <div id='flow-panel' className='panel' role='region' aria-label='Diagnostic Flow' tabIndex='0'>
        <h3>Diagnostic Flow</h3>
        <ol>{flow.map((m, i) => <li key={i}>{m.sender}: {m.text}</li>)}</ol>
      </div>
      <div id='commands-panel' className='panel' role='region' aria-label='Supported Commands' tabIndex='0'>
        <h3>
          <button className='btn btn-link p-0' onClick={() => setShowCommands(!showCommands)} aria-expanded={showCommands} aria-controls='commands-list'>
            Chat Commands
          </button>
        </h3>
        {showCommands && (
          <ul id='commands-list'>
            <li><strong>Ask Question</strong> – general questions about the case</li>
            <li><strong>Order Test</strong> – request a medical test</li>
            <li><strong>Provide Diagnosis</strong> – propose a diagnosis</li>
          </ul>
        )}
      </div>
      {toast && <div id='toast' role='alert' aria-live='polite'>{toast}</div>}
      </div>
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
