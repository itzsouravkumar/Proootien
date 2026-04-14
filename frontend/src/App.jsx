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
        viewer.spin(false)
        
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
          <span style={{...S.badge, background: modelReady ? '#0a0a0a' : '#0a0a0a', color: modelReady ? '#ededed' : '#888888', border: `1px solid ${modelReady ? '#333333' : '#262626'}`}}>
            {modelReady ? 'Model Ready' : 'Not Trained'}
          </span>
        </div>
      </header>

      <div style={S.grid}>
        {/* Controls Card - Span 24 */}
        <div style={{...S.card, gridColumn: '1 / -1', flexDirection: 'row', flexWrap: 'wrap', gap: 24, padding: '24px 32px'}}>
          <div style={{flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: 16}}>
            <h3 style={S.cardTitle}>Analyze Protein</h3>
            <div style={S.row}>
              <input style={S.input} value={pdbId} onChange={e => setPdbId(e.target.value)} placeholder="PDB ID" onKeyDown={e => e.key === 'Enter' && analyze()} />
              <button style={{...S.btn, opacity: loading ? 0.7 : 1}} onClick={analyze} disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
            </div>
          </div>
          
          <div style={S.divider} />

          <div style={{flex: '1 1 400px', display: 'flex', flexDirection: 'column', gap: 16}}>
            <h3 style={S.cardTitle}>Train GNN</h3>
            <div style={S.row}>
              <input style={{...S.input, flex: 2}} value={trainingIds} onChange={e => setTrainingIds(e.target.value)} placeholder="PDB IDs (comma separated)" />
              <input style={{...S.input, width: 80, flex: 'none'}} type="number" value={epochs} onChange={e => setEpochs(+e.target.value || 50)} placeholder="Epochs" />
              <button style={{...S.btn, ...S.btnTrain, opacity: training ? 0.7 : 1}} onClick={train} disabled={training}>{training ? 'Training...' : 'Train Mode'}</button>
            </div>
          </div>
        </div>

        {error && <div style={{...S.error, gridColumn: '1 / -1'}}>{error}</div>}

        {results && (
          <>
            {/* Stats - Span 24 */}
            <div style={{...S.card, gridColumn: '1 / -1', padding: '20px 32px'}}>
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 20}}>
                <Stat label="Protein" value={results.protein_id?.toUpperCase()} />
                <Stat label="Total Surface" value={results.features?.summary?.total_surface_residues || 0} />
                <Stat label="Pockets Found" value={results.features?.pockets?.length || 0} />
                <Stat label="GNN Hotspots" value={results.features?.hotspots?.length || 0} />
              </div>
            </div>

            {/* Viewer - Span 13 */}
            <div style={{...S.card, gridColumn: 'span 13', minHeight: 500}}>
              <h3 style={S.cardTitle}>3D Structure Configuration</h3>
              <div style={{flex: 1, border: '1px solid #1a1a1a', borderRadius: 8, overflow: 'hidden'}}>
                <Viewer3D pdbId={results.protein_id} hotspots={results.features?.hotspots || []} />
              </div>
              <div style={S.legend}>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#ef4444'}}></span>Top 3</span>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#f59e0b'}}></span>Top 4-6</span>
                <span style={S.legendItem}><span style={{...S.legendDot, background: '#3b82f6'}}></span>Others</span>
              </div>
            </div>

            {/* Data Columns - Span 11 */}
            <div style={{gridColumn: 'span 11', display: 'flex', flexDirection: 'column', gap: 16}}>
              <div style={{...S.card, flex: 1, maxHeight: 350}}>
                <h3 style={S.cardTitle}>GNN Hotspots</h3>
                <div style={S.list}>
                  {(results.features?.hotspots || []).map((h, i) => (
                    <div key={i} style={S.hotspot}>
                      <span style={S.rank}>#{i+1}</span>
                      <span style={S.resId}>{h.residue_id}</span>
                      <span style={S.resName}>{h.residue_name}</span>
                      <div style={S.barWrap}><div style={{...S.fill, width: `${h.gnn_score*100}%`, background: i < 3 ? '#ef4444' : i < 6 ? '#f59e0b' : '#3b82f6'}} /></div>
                      <span style={S.pct}>{(h.gnn_score*100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{...S.card, flex: 1, maxHeight: 350}}>
                <h3 style={S.cardTitle}>Identified Pockets</h3>
                <div style={S.list}>
                  {(results.features?.pockets || []).slice(0,5).map((p, i) => (
                    <div key={i} style={S.pocket}>
                      <div style={S.pocketHead}>
                        <span style={S.pocketId}>Pocket {p.pocket_id}</span>
                        <span style={S.pocketVol}>{p.volume?.toFixed(0)} Å³</span>
                      </div>
                      <div style={S.pocketStats}>
                        <span>H: <span style={{color: '#ededed'}}>{p.hydrophobicity?.toFixed(2)}</span></span>
                        <span>C: <span style={{color: '#ededed'}}>{p.charge?.toFixed(2)}</span></span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
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
  container: { width: '100%', margin: '0 auto', padding: '40px 4vw', background: '#000000', minHeight: '100vh', color: '#ededed' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 40, paddingBottom: 24, borderBottom: '1px solid #262626' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 16 },
  title: { fontSize: '24px', fontWeight: 500, color: '#ededed', letterSpacing: '-0.02em', margin: '0 0 4px 0' },
  sub: { fontSize: '14px', color: '#888888', fontWeight: 400, margin: 0 },
  loss: { fontSize: 13, color: '#ededed', background: '#111111', padding: '4px 10px', borderRadius: 4, border: '1px solid #333333' },
  badge: { padding: '4px 10px', borderRadius: 4, fontSize: 12, fontWeight: 500 },
  
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(24, 1fr)',
    gap: 16,
    alignItems: 'start'
  },
  
  card: { 
    background: '#0a0a0a', 
    borderRadius: 8, 
    padding: 24, 
    border: '1px solid #262626',
    display: 'flex',
    flexDirection: 'column',
  },
  cardTitle: { 
    fontSize: 13, 
    fontWeight: 500, 
    color: '#888888', 
    marginBottom: 16, 
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    margin: 0
  },
  
  divider: { width: 1, background: '#262626', margin: '0 16px' },
  row: { display: 'flex', gap: 8 },
  input: { 
    flex: 1, 
    padding: '10px 14px', 
    borderRadius: 6, 
    border: '1px solid #333333', 
    background: '#000000', 
    color: '#ededed', 
    fontSize: 14, 
    outline: 'none',
    transition: 'border-color 0.2s',
    fontFamily: 'inherit'
  },
  btn: { 
    padding: '0 16px', 
    borderRadius: 6, 
    border: '1px solid #333333', 
    background: '#111111', 
    color: '#ededed', 
    fontWeight: 500, 
    cursor: 'pointer', 
    fontSize: 13,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: 'inherit'
  },
  btnTrain: {
    background: '#ededed',
    color: '#000000',
    border: '1px solid #ededed',
  },
  error: { background: '#1a0505', color: '#ff8888', padding: '16px', borderRadius: 8, border: '1px solid #3a1111', fontSize: 14 },
  
  statVal: { fontSize: 24, fontWeight: 500, color: '#ededed', marginBottom: 2, letterSpacing: '-0.02em' },
  statLabel: { fontSize: 11, color: '#888888', textTransform: 'uppercase', letterSpacing: '0.06em' },
  stat: { display: 'flex', flexDirection: 'column', alignItems: 'flex-start' },
  
  legend: { display: 'flex', justifyContent: 'center', gap: 24, marginTop: 24, paddingTop: 16, borderTop: '1px solid #262626' },
  legendItem: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#888888' },
  legendDot: { width: 8, height: 8, borderRadius: '50%' },
  
  list: { overflowY: 'auto', flex: 1, paddingRight: 8, marginTop: 16 },
  hotspot: { display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderBottom: '1px solid #1a1a1a' },
  rank: { fontSize: 12, fontWeight: 500, width: 24, color: '#888888' },
  resId: { fontWeight: 500, color: '#ededed', fontSize: 13, width: 60 },
  resName: { color: '#888888', fontSize: 12, width: 36 },
  barWrap: { flex: 1, height: 4, background: '#1a1a1a', borderRadius: 2, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 2 },
  pct: { fontWeight: 500, width: 36, textAlign: 'right', fontSize: 12, color: '#888888' },
  
  pocket: { padding: '10px 0', borderBottom: '1px solid #1a1a1a', marginTop: 8 },
  pocketHead: { display: 'flex', justifyContent: 'space-between', marginBottom: 4 },
  pocketId: { fontWeight: 500, color: '#ededed', fontSize: 13 },
  pocketVol: { color: '#888888', fontSize: 12 },
  pocketStats: { display: 'flex', gap: 16, fontSize: 12, color: '#666666' },
}

export default App