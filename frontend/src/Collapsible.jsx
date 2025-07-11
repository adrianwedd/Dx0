import { useState } from 'react'
import PropTypes from 'prop-types'

export default function Collapsible({ title, children }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="panel">
      <h3>
        <button className="toggle" onClick={() => setOpen(!open)}>
          {title}
        </button>
      </h3>
      {open && <div className="content">{children}</div>}
    </div>
  )
}

Collapsible.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired
}
