import { useState, useEffect, useRef } from 'react'
import { searchFighters, predict } from '../api'

const WEIGHT_CLASSES = [
  'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
  'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight',
  "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
]

function FighterSearch({ label, value, onChange }) {
  const [query, setQuery] = useState(value?.name || '')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const ref = useRef()

  useEffect(() => {
    if (query.length < 2) { setResults([]); return }
    const t = setTimeout(() => {
      searchFighters(query).then(r => { setResults(r.data); setOpen(true) })
    }, 300)
    return () => clearTimeout(t)
  }, [query])

  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const select = (fighter) => {
    setQuery(fighter.name)
    onChange(fighter)
    setOpen(false)
    setResults([])
  }

  return (
    <div ref={ref} style={{ position: 'relative', flex: 1 }}>
      <div className="label" style={{ marginBottom: 8 }}>{label}</div>
      <input
        value={query}
        onChange={e => { setQuery(e.target.value); if (!e.target.value) onChange(null) }}
        placeholder="Search fighter..."
      />
      {open && results.length > 0 && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 100,
          background: 'var(--bg3)', border: '1px solid var(--border2)',
          borderRadius: 8, marginTop: 4, overflow: 'hidden'
        }}>
          {results.map((f, i) => (
            <div key={i} onClick={() => select(f)}
              style={{ padding: '10px 14px', cursor: 'pointer', fontSize: 14, transition: 'background 0.1s' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
              <div style={{ fontWeight: 500 }}>{f.name}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>{f.stance} · {f.height} · {f.reach}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Matchup() {
  const [fighter1, setFighter1] = useState(null)
  const [fighter2, setFighter2] = useState(null)
  const [weightClass, setWeightClass] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    if (!fighter1 || !fighter2) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await predict({
        fighter_1: fighter1.name,
        fighter_2: fighter2.name,
        weight_class: weightClass || null,
      })
      setResult(r.data)
    } catch (e) {
      setError('Prediction failed. Make sure both fighters are in the database.')
    } finally {
      setLoading(false)
    }
  }

  const f1Prob = result?.fighter_1_win_probability ?? 0
  const f2Prob = result?.fighter_2_win_probability ?? 0

  return (
    <div style={{ maxWidth: 700, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

      <div className="card">
        <div style={{ fontFamily: 'var(--font-disp)', fontSize: 22, letterSpacing: 1, marginBottom: 20 }}>
          FIGHT PREDICTOR
        </div>

        {/* Fighter selects */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <FighterSearch label="Fighter 1" value={fighter1} onChange={setFighter1} />
          <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: 10, color: 'var(--muted)', fontWeight: 600 }}>VS</div>
          <FighterSearch label="Fighter 2" value={fighter2} onChange={setFighter2} />
        </div>

        {/* Weight class */}
        <div style={{ marginTop: 16 }}>
          <div className="label" style={{ marginBottom: 8 }}>Weight class (optional)</div>
          <select value={weightClass} onChange={e => setWeightClass(e.target.value)}>
            <option value="">— Select weight class —</option>
            {WEIGHT_CLASSES.map(wc => <option key={wc} value={wc}>{wc}</option>)}
          </select>
        </div>

        {/* Predict button */}
        <button
          className="btn-primary"
          onClick={handlePredict}
          disabled={!fighter1 || !fighter2 || loading}
          style={{ marginTop: 20, width: '100%', padding: '14px', fontSize: 15, fontFamily: 'var(--font-disp)', letterSpacing: 1, opacity: (!fighter1 || !fighter2) ? 0.5 : 1 }}
        >
          {loading ? 'PREDICTING...' : 'PREDICT FIGHT'}
        </button>

        {error && <div style={{ color: 'var(--fail)', fontSize: 13, marginTop: 12 }}>{error}</div>}
      </div>

      {/* Result card */}
      {result && (
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-disp)', fontSize: 14, letterSpacing: 2, color: 'var(--muted)', marginBottom: 16 }}>
            PREDICTION
          </div>

          {/* Fighters + probs */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: 24, marginBottom: 24 }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 18, fontWeight: result.predicted_winner === result.fighter_1 ? 600 : 400, color: result.predicted_winner === result.fighter_1 ? 'var(--text)' : 'var(--muted)' }}>
                {result.fighter_1}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, color: 'var(--red)', marginTop: 4 }}>
                {(f1Prob * 100).toFixed(0)}%
              </div>
            </div>

            <div style={{ color: 'var(--muted)', fontWeight: 600, fontSize: 13 }}>VS</div>

            <div style={{ textAlign: 'left' }}>
              <div style={{ fontSize: 18, fontWeight: result.predicted_winner === result.fighter_2 ? 600 : 400, color: result.predicted_winner === result.fighter_2 ? 'var(--text)' : 'var(--muted)' }}>
                {result.fighter_2}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, color: 'var(--red)', marginTop: 4 }}>
                {(f2Prob * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Probability bar */}
          <div style={{ height: 8, borderRadius: 999, background: 'var(--bg3)', overflow: 'hidden', display: 'flex', marginBottom: 16 }}>
            <div style={{ width: `${f1Prob * 100}%`, background: 'var(--red)', transition: 'width 0.5s ease' }}/>
            <div style={{ flex: 1, background: 'var(--bg)' }}/>
          </div>

          {/* Winner + confidence */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className="tag gold">
              ▲ {result.predicted_winner} wins
            </span>
            <span className={`tag ${result.confidence === 'high' ? 'red' : ''}`}>
              {result.confidence} confidence
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
