import './App.css'
import { useEffect, useMemo, useRef } from 'react'
import { SimulationRunner } from './simulation/runner'
import { Controls } from './ui/Controls'
import { Hud } from './ui/Hud'
import { useSimStore } from './state/simStore'

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const runnerRef = useRef<SimulationRunner | null>(null)
  
  // Track best distance from store to update button
  const bestDistanceAllTime = useSimStore((s) => s.metrics.bestDistanceAllTimeM)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const runner = new SimulationRunner(canvas)
    runnerRef.current = runner
    runner.start()
    return () => {
      runner.stop()
      runnerRef.current = null
    }
  }, [])

  const handlers = useMemo(
    () => ({
      onReset: () => runnerRef.current?.reset(),
      onSkipGeneration: () => runnerRef.current?.skipToNextGeneration(),
      onReplayBest: () => runnerRef.current?.replayAllTimeBest(),
      onCompareGenerations: () => runnerRef.current?.toggleCompareGenerations(),
    }),
    [],
  )

  return (
    <div className="appRoot">
      <canvas ref={canvasRef} className="simCanvas" />
      <Controls 
        onReset={handlers.onReset} 
        onSkipGeneration={handlers.onSkipGeneration}
        onReplayBest={handlers.onReplayBest}
        onCompareGenerations={handlers.onCompareGenerations}
        bestDistance={bestDistanceAllTime}
      />
      <Hud />
    </div>
  )
}

export default App
