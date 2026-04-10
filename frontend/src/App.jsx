import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:8000/api/v1'

function Viewer3D({ pdbId, hotspots = [] }) {
  const containerRef = useRef(null)
  const viewerRef = useRef(null)
  const [viewerStatus, setViewerStatus] = useState('initializing')
  const $3DmolRef = useRef(null)

  useEffect(() => {
    // Dynamically import 3dmol
    import('3dmol').then(module => {
      $3DmolRef.current = module
      setViewerStatus('ready to load')
    }).catch(err => {
      console.error('Failed to load 3Dmol:', err)
      setViewerStatus('error: failed to load 3Dmol')
    })
  }, [])

  useEffect(() => {
    if (!containerRef.current || !pdbId || !$3DmolRef.current) return

    let mounted = true
    const $3Dmol = $3DmolRef.current

    const loadViewer = async () => {
      setViewerStatus('loading')

      // Clear previous viewer
      if (viewerRef.current) {
        try {
          viewerRef.current.spin(false)
          viewerRef.current.clear()
        } catch (e) {}
        viewerRef.current = null
      }
      
      // Clear container
      containerRef.current.innerHTML = ''

      try {
        // Create viewer
        const viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: '#0f172a',
          antialias: true
        })
        
        if (!viewer) {
          setViewerStatus('error: viewer creation failed')
          return
        }
        
        viewerRef.current = viewer

        // Fetch PDB from RCSB
        setViewerStatus('fetching PDB...')
        const response = await fetch(`https://files.rcsb.org/download/${pdbId.toLowerCase()}.pdb`)
        
        if (!response.ok) {
          setViewerStatus(`error: PDB ${pdbId} not found`)
          return
        }
        
        const pdbData = await response.text()
        
        if (!mounted) return
        
        if (!pdbData || pdbData.length < 100) {
          setViewerStatus('error: invalid PDB data')
          return
        }

        setViewerStatus('rendering...')
        
        // Add model
        viewer.addModel(pdbData, 'pdb')
        
        // Base style: cartoon
        viewer.setStyle({}, { cartoon: { color: '#64748b' } })
        
        // Highlight hotspots
        if (hotspots && hotspots.length > 0) {
          hotspots.forEach((h, i) => {
            // Parse residue_id: "A:123_" -> chain=A, resi=123
            const parts = h.residue_id?.split(':') || []
            const chain = parts[0] || 'A'
            const resiPart = parts[1] || ''
            const resi = parseInt(resiPart.replace(/\D/g, '')) || 0
            
            if (resi > 0) {
              const color = i < 3 ? '#ef4444' : i < 6 ? '#f59e0b' : '#3b82f6'
              
              viewer.setStyle(
                { chain: chain, resi: resi },
                { 
                  cartoon: { color: color },
                  stick: { color: color, radius: 0.15 }
                }
              )
            }
          })
        }
        
        // Render
        viewer.zoomTo()
        viewer.render()
        viewer.spin('y', 0.5)
        
        setViewerStatus('ready')
        
      } catch (e) {
        console.error('Viewer error:', e)
        setViewerStatus('error: ' + e.message)
      }
    }

    loadViewer()

    return () => {
      mounted = false
      if (viewerRef.current) {
        try {
          viewerRef.current.spin(false)
          viewerRef.current.clear()
        } catch (e) {}
      }
    }
  }, [pdbId, hotspots, $3DmolRef.current])

  return (
    <div style={{ position: 'relative' }}>
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: 450, 
          borderRadius: 12,
          background: '#0f172a',
          position: 'relative'
        }} 
      />
      {viewerStatus !== 'ready' && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: viewerStatus.startsWith('error') ? '#ef4444' : '#64748b',
          fontSize: 13,
          textAlign: 'center',
          background: '#1e293b',
          padding: '12px 20px',
          borderRadius: 8
        }}>
          {viewerStatus}
        </div>
      )}
    </div>
  )
}

function App() {
  const [pdbId, setPdbId] = useState('1crn')
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [modelReady, setModelReady] = useState(false)
  const [trainingIds, setTrainingIds] = useState('1crn,4hhb,1ubq,2src,1tim')
  const [epochs, setEpochs] = useState(100)
  const [trainLoss, setTrainLoss] = useState(null)

  useEffect(() => {
    fetch(`${API}/model/status`)
      .then(r => r.json())
      .then(d => setModelReady(d.is_trained))
      .catch(() => {})
  }, [])

  const analyze = async () => {
    if (!pdbId.trim()) return
    setLoading(true)
    setError(null)
    
    try {
      const res = await fetch(`${API}/analyze/pdb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdb_id: pdbId.toLowerCase() })
      })
      
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed')
      }
      
      setResults(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const train = async () => {
    const ids = trainingIds.split(',').map(s => s.trim()).filter(Boolean)
    if (!ids.length) return
    
    setTraining(true)
    setError(null)
    
    try {
      const res = await fetch(`${API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdb_ids: ids, epochs })
      })
      
      if (!res.ok) throw new Error('Training failed')
      
      const data = await res.json()
      if (data.success) {
        setModelReady(true)
        setTrainLoss(data.loss)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setTraining(false)
    }
  }

  return (
    <div style={S.container}>
      <header style={S.header}>
        <div>
          <h1 style={S.title}>Protein Surface Analyzer</h1>
          <p style={S.sub}>GNN-powered binding site prediction</p>
        </div>
        <div style={S.headerRight}>
          {trainLoss !== null && <span style={S.loss}>Loss: {trainLoss.toFixed(4)}</span>}
          <span style={{...S.badge, background: modelReady ? '#10b981' : '#f59e0b'}}>
            {modelReady ? 'Model Ready' : 'Not Trained'}
          </span>
        </div>
      </header>

      <div style={S.controls}>
        <div style={S.card}>
          <h3 style={S.cardTitle}>Analyze</h3>
          <div style={S.row}>
            <input style={S.input} value={pdbId} onChange={e => setPdbId(e.target.value)} placeholder="PDB ID" onKeyDown={e => e.key === 'Enter' && analyze()} />
            <button style={S.btn} onClick={analyze} disabled={loading}>{loading ? '...' : 'Analyze'}</button>
          </div>
        </div>
        <div style={S.card}>
          <h3 style={S.cardTitle}>Train GNN</h3>
          <div style={S.row}>
            <input style={{...S.input, flex: 2}} value={trainingIds} onChange={e => setTrainingIds(e.target.value)} placeholder="PDB IDs" />
            <input style={{...S.input, width: 60}} type="number" value={epochs} onChange={e => setEpochs(+e.target.value || 50)} />
            <button style={{...S.btn, background: '#8b5cf6'}} onClick={train} disabled={training}>{training ? '...' : 'Train'}</button>
          </div>
        </div>
      </div>

      {error && <div style={S.error}>{error}</div>}

      {results && (
        <>
          <div style={S.stats}>
            <Stat label="Protein" value={results.protein_id?.toUpperCase()} />
            <Stat label="Surface" value={results.features?.summary?.total_surface_residues || 0} />
            <Stat label="Pockets" value={results.features?.pockets?.length || 0} />
            <Stat label="Hotspots" value={results.features?.hotspots?.length || 0} />
          </div>

          <div style={S.main}>
            <div style={S.viewerCard}>
              <h3 style={S.cardTitle}>3D Structure</h3>
              <Viewer3D pdbId={results.protein_id} hotspots={results.features?.hotspots || []} />
              <div style={S.legend}>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#ef4444'}}></span>Top 3</span>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#f59e0b'}}></span>Top 4-6</span>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#3b82f6'}}></span>Others</span>
              </div>
            </div>

            <div style={S.sidebar}>
              <div style={S.card}>
                <h3 style={S.cardTitle}>GNN Hotspots</h3>
                <div style={S.list}>
                  {(results.features?.hotspots || []).map((h, i) => (
                    <div key={i} style={S.hotspot}>
                      <span style={{...S.rank, color: i < 3 ? '#ef4444' : '#64748b'}}>#{i+1}</span>
                      <span style={S.resId}>{h.residue_id}</span>
                      <span style={S.resName}>{h.residue_name}</span>
                      <div style={S.bar}><div style={{...S.fill, width: `${h.gnn_score*100}%`, background: i < 3 ? '#ef4444' : i < 6 ? '#f59e0b' : '#3b82f6'}} /></div>
                      <span style={{...S.pct, color: i < 3 ? '#ef4444' : '#10b981'}}>{(h.gnn_score*100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>Pockets</h3>
                <div style={S.list}>
                  {(results.features?.pockets || []).slice(0,5).map((p, i) => (
                    <div key={i} style={S.pocket}>
                      <div style={S.pocketHead}>
                        <span style={S.pocketId}>Pocket {p.pocket_id}</span>
                        <span style={S.pocketVol}>{p.volume?.toFixed(0)} Å³</span>
                      </div>
                      <div style={S.pocketStats}>
                        <span>H: {p.hydrophobicity?.toFixed(2)}</span>
                        <span>C: {p.charge?.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

const Stat = ({ label, value }) => (
  <div style={S.stat}>
    <div style={S.statVal}>{value}</div>
    <div style={S.statLabel}>{label}</div>
  </div>
)

const S = {
  container: { maxWidth: 1300, margin: '0 auto', padding: 20, fontFamily: 'system-ui, sans-serif', background: '#0f172a', minHeight: '100vh', color: '#e2e8f0' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, paddingBottom: 16, borderBottom: '1px solid #334155' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 12 },
  title: { fontSize: 24, fontWeight: 700, color: '#f1f5f9' },
  sub: { fontSize: 13, color: '#64748b' },
  loss: { fontSize: 11, color: '#94a3b8', background: '#1e293b', padding: '4px 8px', borderRadius: 4 },
  badge: { padding: '6px 14px', borderRadius: 16, fontSize: 11, fontWeight: 600, color: '#fff' },
  controls: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 },
  card: { background: '#1e293b', borderRadius: 12, padding: 16 },
  cardTitle: { fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 },
  row: { display: 'flex', gap: 8 },
  input: { flex: 1, padding: '10px 14px', borderRadius: 8, border: '1px solid #334155', background: '#0f172a', color: '#e2e8f0', fontSize: 13, outline: 'none' },
  btn: { padding: '10px 20px', borderRadius: 8, border: 'none', background: '#3b82f6', color: '#fff', fontWeight: 600, cursor: 'pointer', fontSize: 13 },
  error: { background: '#7f1d1d', color: '#fca5a5', padding: 12, borderRadius: 8, marginBottom: 16, fontSize: 13 },
  stats: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20 },
  stat: { background: '#1e293b', borderRadius: 10, padding: 16, textAlign: 'center' },
  statVal: { fontSize: 22, fontWeight: 700, color: '#3b82f6' },
  statLabel: { fontSize: 10, color: '#64748b', marginTop: 4, textTransform: 'uppercase' },
  main: { display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16 },
  viewerCard: { background: '#1e293b', borderRadius: 12, padding: 16 },
  legend: { display: 'flex', justifyContent: 'center', gap: 20, marginTop: 12, paddingTop: 12, borderTop: '1px solid #334155' },
  legendItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' },
  legendDot: { width: 10, height: 10, borderRadius: '50%' },
  sidebar: { display: 'flex', flexDirection: 'column', gap: 16 },
  list: { maxHeight: 280, overflowY: 'auto' },
  hotspot: { display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0', borderBottom: '1px solid #334155' },
  rank: { fontSize: 11, fontWeight: 700, width: 24 },
  resId: { fontWeight: 600, color: '#f1f5f9', fontSize: 12, width: 60 },
  resName: { color: '#64748b', fontSize: 11, width: 36 },
  bar: { flex: 1, height: 6, background: '#334155', borderRadius: 3, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 3 },
  pct: { fontWeight: 700, width: 36, textAlign: 'right', fontSize: 12 },
  pocket: { padding: '10px 0', borderBottom: '1px solid #334155' },
  pocketHead: { display: 'flex', justifyContent: 'space-between', marginBottom: 4 },
  pocketId: { fontWeight: 600, color: '#f1f5f9', fontSize: 12 },
  pocketVol: { color: '#3b82f6', fontSize: 11 },
  pocketStats: { display: 'flex', gap: 12, fontSize: 11, color: '#64748b' },
}

export default App