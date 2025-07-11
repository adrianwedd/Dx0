import { useState } from 'react'

export default function CollapsiblePanel({title, children, defaultOpen = true}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="panel">
      <h3>
        <button className="toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
          {title}
        </button>
      </h3>
      {open && <div className="panel-body">{children}</div>}
    </div>
  )
}
