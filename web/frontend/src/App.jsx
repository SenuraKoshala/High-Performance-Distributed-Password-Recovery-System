import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import SingleRun from './pages/SingleRun'
import Benchmark from './pages/Benchmark'
import Architecture from './pages/Architecture'

function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <div>
            <div className="header-logo">⚡ HPC Cracker</div>
            <div className="header-subtitle">EE7218 / EC7207 · Group 30</div>
          </div>
        </div>
        <nav className="header-nav">
          <NavLink to="/" end className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Dashboard</NavLink>
          <NavLink to="/run" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Run</NavLink>
          <NavLink to="/benchmark" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Benchmark</NavLink>
          <NavLink to="/architecture" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Architecture</NavLink>
        </nav>
      </div>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/run" element={<SingleRun />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/architecture" element={<Architecture />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
