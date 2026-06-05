import { useEffect, useState } from 'react'
import { getUpcoming } from '../api'

export default function Upcoming() {
  const [fights, setFights] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getUpcoming().then(r => {
      setFights(r.data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ color: 'var(--muted)', padding: 40, textAlign: 'center' }}>Loading predictions...</div>

  // Group by event
  const events = fights.reduce((acc, f) => {
    const key = f.event_name
    if (!acc[key]) acc[key] = { date: f.date, fights: [] }
    acc[key].fights.push(f)
    return acc
  }, {})

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
      {Object.entries(events).map(([eventName, { date, fights }]) => (
        <div key={eventName} className="card">
          {/* Event header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <div>
              <div style={{ fontFamily: 'var(--font-disp)', fontSize: 22, letterSpacing: 1 }}>{eventName}</div>
              <div style={{ color: 'var(--muted)', fontSize: 13, marginTop: 2 }}>
                {new Date(date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
              </div>
            </div>
            <span className="tag gold">{fights.length} fights predicted</span>
          </div>

          <div className="divider" style={{ margin: '0 0 20px' }}/>

          {/* Fight rows */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {fights.map((f, i) => {
              const f1Wins = f.fighter_1_win_prob > f.fighter_2_win_prob
              const winnerProb = Math.max(f.fighter_1_win_prob, f.fighter_2_win_prob)
              const loserProb = Math.min(f.fighter_1_win_prob, f.fighter_2_win_prob)

              return (
                <div key={i} className="card-sm" style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: 16 }}>
                  {/* Fighter 1 */}
                  <div style={{ textAlign: 'right' }}>
                    <div style={{
                      fontSize: 15, fontWeight: f1Wins ? 600 : 400,
                      color: f1Wins ? 'var(--text)' : 'var(--muted)'
                    }}>
                      {f.fighter_1_name}
                      {f1Wins && <span style={{ color: 'var(--red)', marginLeft: 6, fontSize: 12 }}>▲</span>}
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: f1Wins ? 'var(--red)' : 'var(--muted)', marginTop: 2 }}>
                      {(f.fighter_1_win_prob * 100).toFixed(0)}%
                    </div>
                  </div>

                  {/* VS + bar */}
                  <div style={{ textAlign: 'center', minWidth: 140 }}>
                    <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', marginBottom: 6 }}>VS</div>
                    <div style={{ height: 6, borderRadius: 999, background: 'var(--bg)', overflow: 'hidden', display: 'flex' }}>
                      <div style={{ width: `${f.fighter_1_win_prob * 100}%`, background: f1Wins ? 'var(--red)' : 'var(--muted)', borderRadius: '999px 0 0 999px', transition: 'width 0.4s' }}/>
                      <div style={{ flex: 1, background: !f1Wins ? 'var(--red)' : 'var(--muted)', borderRadius: '0 999px 999px 0' }}/>
                    </div>
                    <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 4 }}>{f.weight_class}</div>
                  </div>

                  {/* Fighter 2 */}
                  <div style={{ textAlign: 'left' }}>
                    <div style={{
                      fontSize: 15, fontWeight: !f1Wins ? 600 : 400,
                      color: !f1Wins ? 'var(--text)' : 'var(--muted)'
                    }}>
                      {!f1Wins && <span style={{ color: 'var(--red)', marginRight: 6, fontSize: 12 }}>▲</span>}
                      {f.fighter_2_name}
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: !f1Wins ? 'var(--red)' : 'var(--muted)', marginTop: 2 }}>
                      {(f.fighter_2_win_prob * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}

      {fights.length === 0 && (
        <div className="card" style={{ textAlign: 'center', color: 'var(--muted)', padding: 60 }}>
          No upcoming predictions stored yet.
        </div>
      )}
    </div>
  )
}
