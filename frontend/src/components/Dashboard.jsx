import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { getModelCurrent, getAccuracy, getAccuracyByWC, getAccuracyByVer, getHistory } from '../api'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-sm" style={{ fontSize: 12 }}>
      <div style={{ color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' && p.value < 1 ? (p.value * 100).toFixed(1) + '%' : p.value}</div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [model, setModel]     = useState(null)
  const [accuracy, setAcc]    = useState(null)
  const [byWC, setByWC]       = useState([])
  const [byVer, setByVer]     = useState([])
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getModelCurrent(),
      getAccuracy(),
      getAccuracyByWC(),
      getAccuracyByVer(),
      getHistory(),
    ]).then(([m, a, wc, v, h]) => {
      setModel(m.data)
      setAcc(a.data)
      setByWC(wc.data)
      setByVer(v.data)
      setHistory(h.data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ color: 'var(--muted)', padding: 40, textAlign: 'center' }}>Loading dashboard...</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Model stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
        {[
          { label: 'Model version',  value: model ? `v${model.version}` : '—' },
          { label: 'Test AUC',       value: model ? model.auc?.toFixed(4) : '—' },
          { label: 'Test accuracy',  value: model ? (model.accuracy * 100).toFixed(1) + '%' : '—' },
          { label: 'Live accuracy',  value: accuracy?.accuracy != null ? (accuracy.accuracy * 100).toFixed(1) + '%' : 'No data yet' },
          { label: 'Fights scored',  value: accuracy?.total_scored ?? '—' },
        ].map(({ label, value }) => (
          <div key={label} className="card">
            <div className="label">{label}</div>
            <div className="metric">{value}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>

        {/* Accuracy by weight class */}
        <div className="card">
          <div className="label" style={{ marginBottom: 16 }}>Accuracy by weight class</div>
          {byWC.length === 0
            ? <div style={{ color: 'var(--muted)', fontSize: 13 }}>No scored predictions yet</div>
            : <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byWC} layout="vertical" margin={{ left: 20 }}>
                  <XAxis type="number" domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false}/>
                  <YAxis type="category" dataKey="weight_class" tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} width={120}/>
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="accuracy" name="Accuracy" radius={[0, 4, 4, 0]}>
                    {byWC.map((_, i) => <Cell key={i} fill="var(--red)" fillOpacity={0.7 + i * 0.03}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
          }
        </div>

        {/* Accuracy by model version */}
        <div className="card">
          <div className="label" style={{ marginBottom: 16 }}>Accuracy by model version</div>
          {byVer.length === 0
            ? <div style={{ color: 'var(--muted)', fontSize: 13 }}>No scored predictions yet</div>
            : <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byVer} margin={{ left: 10 }}>
                  <XAxis dataKey="model_version" tickFormatter={v => `v${v}`} tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false}/>
                  <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false}/>
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="accuracy" name="Accuracy" fill="var(--gold)" radius={[4, 4, 0, 0]}/>
                </BarChart>
              </ResponsiveContainer>
          }
        </div>
      </div>

      {/* Recent predictions table */}
      <div className="card">
        <div className="label" style={{ marginBottom: 16 }}>Recent scored predictions</div>
        {history.length === 0
          ? <div style={{ color: 'var(--muted)', fontSize: 13 }}>No scored predictions yet — check back after the next event.</div>
          : <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                    {['Fighter 1', 'Fighter 2', 'Prediction', 'Confidence', 'Result', 'Weight class', 'Model'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 500, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  
                  {history.slice(0, 20).map((p, i) => {
                    const f1Wins = p.fighter_1_win_prob > p.fighter_2_win_prob
                    const conf = Math.abs(p.fighter_1_win_prob - 0.5)
                    const confLabel = conf > 0.15 ? 'High' : conf > 0.05 ? 'Medium' : 'Low'
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.1s' }}
                          onMouseEnter={e => e.currentTarget.style.background = 'var(--bg3)'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                        <td style={{ padding: '10px 12px' }}>{p.fighter_1_name}</td>
                        <td style={{ padding: '10px 12px' }}>{p.fighter_2_name}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--gold)' }}>
                          {f1Wins ? p.fighter_1_name : p.fighter_2_name}
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          <span className={`tag ${conf > 0.15 ? 'red' : ''}`}>{confLabel}</span>
                        </td>
                        <td style={{ padding: '10px 12px' }}>
                          <span className={`tag ${p.correct ? 'green' : 'red'}`}>
                            {p.correct ? '✓ Correct' : '✗ Wrong'}
                          </span>
                        </td>
                        <td style={{ padding: '10px 12px', color: 'var(--muted)' }}>{p.weight_class}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                          v{p.model_version}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
        }
      </div>
    </div>
  )
}
