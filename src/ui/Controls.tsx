import { useSimStore } from '../state/simStore'
import { logger } from '../utils/logger'

export function Controls(props: { 
  onReset: () => void
  onSkipGeneration: () => void
  onReplayBest?: () => void
  onCompareGenerations?: () => void
  bestDistance?: number
}) {
  const playing = useSimStore((s) => s.playing)
  const setPlaying = useSimStore((s) => s.setPlaying)
  const speedMultiplier = useSimStore((s) => s.speedMultiplier)
  const setSpeedMultiplier = useSimStore((s) => s.setSpeedMultiplier)
  const view = useSimStore((s) => s.view)
  const setView = useSimStore((s) => s.setView)

  return (
    <div className="controls">
      <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
      <button onClick={props.onReset}>Reset</button>
      <button onClick={props.onSkipGeneration}>Skip Gen</button>

      <label className="controlRow">
        <span>Speed</span>
        <select
          value={speedMultiplier}
          onChange={(e) => setSpeedMultiplier(Number(e.target.value))}
        >
          {[0.5, 1, 2, 5, 10, 25, 50, 100].map((v) => (
            <option key={v} value={v}>
              {v}x
            </option>
          ))}
        </select>
      </label>

      <label className="controlRow">
        <span>Camera</span>
        <select
          value={view.cameraMode}
          onChange={(e) => setView({ cameraMode: e.target.value as 'follow' | 'fixed' })}
        >
          <option value="follow">Follow</option>
          <option value="fixed">Fixed</option>
        </select>
      </label>

      <label className="controlToggle">
        <input
          type="checkbox"
          checked={view.showNetwork}
          onChange={(e) => setView({ showNetwork: e.target.checked })}
        />
        <span>Network</span>
      </label>
      <label className="controlToggle">
        <input
          type="checkbox"
          checked={view.showMuscleActivation}
          onChange={(e) => setView({ showMuscleActivation: e.target.checked })}
        />
        <span>Activations</span>
      </label>
      <label className="controlToggle">
        <input
          type="checkbox"
          checked={view.showPowerGraph}
          onChange={(e) => setView({ showPowerGraph: e.target.checked })}
        />
        <span>Power</span>
      </label>
      
      <label className="controlToggle" title="Only show the best genome from each generation">
        <input
          type="checkbox"
          checked={view.showBestOnly}
          onChange={(e) => setView({ showBestOnly: e.target.checked })}
        />
        <span>Best Only</span>
      </label>
      
      <button 
        onClick={() => logger.downloadLogs()}
        style={{ marginTop: '8px' }}
      >
        Download Logs
      </button>
      
      {props.onReplayBest && (
        <button 
          onClick={props.onReplayBest}
          style={{ marginTop: '8px', backgroundColor: '#4CAF50', color: 'white' }}
          title={props.bestDistance ? `Replay the best walker: ${props.bestDistance.toFixed(2)}m` : 'No best recorded yet'}
        >
          Replay Best ({props.bestDistance?.toFixed(1) ?? 0}m)
        </button>
      )}

      {props.onCompareGenerations && (
        <button onClick={props.onCompareGenerations} style={{ marginTop: '8px' }}>
          Compare Gens (1/10/25/50/100/200)
        </button>
      )}
    </div>
  )
}
