import { useState } from 'react'
import Dashboard from './components/Dashboard'
import Upcoming from './components/Upcoming'
import Matchup from './components/Matchup'
import './index.css'

const TABS = [
  { id: 'dashboard', label: 'Model Performance' },
  { id: 'upcoming',  label: 'Upcoming Fights' },
  { id: 'matchup',   label: 'Custom Matchup' },
]

export default function App() {
  const [tab, setTab] = useState('upcoming')

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 24px 60px' }}>
      {/* Header */}
      <header style={{ padding: '40px 0 32px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-disp)', fontSize: 48, letterSpacing: 2, lineHeight: 1, color: 'var(--text)' }}>
              OCTAGON
              <span style={{ color: 'var(--red)' }}> ANALYTICS</span>
            </div>
            <div style={{ color: 'var(--muted)', fontSize: 13, marginTop: 6, letterSpacing: '0.05em' }}>
              UFC fight prediction · powered by XGBoost · self-updating weekly
            </div>
          </div>
        </div>

        {/* Tabs */}
        <nav style={{ display: 'flex', gap: 4, marginTop: 28 }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                background: tab === t.id ? 'var(--red)' : 'transparent',
                color: tab === t.id ? '#fff' : 'var(--muted)',
                border: tab === t.id ? 'none' : '1px solid var(--border)',
                borderRadius: 8,
                padding: '8px 18px',
                fontSize: 13,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Content */}
      <main style={{ marginTop: 32 }}>
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'upcoming'  && <Upcoming />}
        {tab === 'matchup'   && <Matchup />}
      </main>
    </div>
  )
}
