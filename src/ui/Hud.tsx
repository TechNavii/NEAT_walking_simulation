import { useEffect, useMemo, useRef } from 'react'
import { JOINT_ORDER, type JointName } from '../physics/human'
import { useSimStore } from '../state/simStore'
import { Panel } from './Panel'
import { NetworkView } from './NetworkView'

const JOINT_LABEL: Record<JointName, string> = {
  neck: 'Neck',
  shoulderL: 'Shoulder L',
  elbowL: 'Elbow L',
  shoulderR: 'Shoulder R',
  elbowR: 'Elbow R',
  hipL: 'Hip L',
  kneeL: 'Knee L',
  ankleL: 'Ankle L',
  hipR: 'Hip R',
  kneeR: 'Knee R',
  ankleR: 'Ankle R',
}

function energyColor(ratio: number): string {
  if (ratio >= 0.66) return '#34d399'
  if (ratio >= 0.33) return '#fbbf24'
  return '#f87171'
}

export function Hud() {
  const metrics = useSimStore((s) => s.metrics)
  const config = useSimStore((s) => s.config)
  const view = useSimStore((s) => s.view)

  const powerHistory = useRef<number[]>([])
  const powerCanvas = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (!view.showPowerGraph) return
    powerHistory.current.push(metrics.currentPowerW)
    if (powerHistory.current.length > 200) powerHistory.current.shift()
    drawSparkline(powerCanvas.current, powerHistory.current)
  }, [metrics.currentPowerW, view.showPowerGraph])

  const activationRows = useMemo(() => {
    const acts = metrics.muscleActivations
    return JOINT_ORDER.map((j, i) => ({ joint: j, value: acts[i] ?? 0 }))
  }, [metrics.muscleActivations])

  const energyRows = useMemo(() => {
    const es = metrics.muscleEnergies
    return JOINT_ORDER.map((j, i) => ({ joint: j, value: es[i] ?? 1 }))
  }, [metrics.muscleEnergies])

  return (
    <>
      <div className="hudTopLeft">
        <Panel title="Muscle Energy">
          <div className="barList">
            {energyRows.map((r) => (
              <div key={r.joint} className="barRow">
                <div className="barLabel">{JOINT_LABEL[r.joint]}</div>
                <div className="barTrack">
                  <div
                    className="barFill"
                    style={{ width: `${Math.round(r.value * 100)}%`, background: energyColor(r.value) }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="hudTopCenter">
        <Panel title={`Generation ${metrics.generation}`}>
          <div className="kv">
            <div>Genome</div>
            <div>
              {metrics.genomeIndex}/{metrics.genomeCount}
            </div>
            <div>Time</div>
            <div>
              {metrics.evalTimeSeconds.toFixed(1)} / {config.evaluation.maxSeconds}s
            </div>
            <div>Best (gen)</div>
            <div>{metrics.bestDistanceThisGenM.toFixed(1)} m</div>
            <div>Best (all)</div>
            <div>{metrics.bestDistanceAllTimeM.toFixed(1)} m</div>
          </div>
        </Panel>
      </div>

      {view.showNetwork && (
        <div className="hudTopRight">
          <Panel title="Network">
            <NetworkView graph={metrics.networkGraph} />
          </Panel>
        </div>
      )}

      {view.showMuscleActivation && (
        <div className="hudRightMiddle">
          <Panel title="Muscle Activation">
            <div className="barList">
              {activationRows.map((r) => (
                <div key={r.joint} className="barRow">
                  <div className="barLabel">{JOINT_LABEL[r.joint]}</div>
                  <ActivationBar value={r.value} />
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}

      {view.showPowerGraph && (
        <div className="hudRightBottom">
          <Panel
            title="Total Muscle Power"
            right={<div className="mono">{metrics.currentPowerW.toFixed(1)} W</div>}
          >
            <canvas ref={powerCanvas} className="sparkline" width={240} height={64} />
          </Panel>
        </div>
      )}

      <div className="hudBottom">
        <div className="bottomStrip">
          <div className="mono">{metrics.currentDistanceM.toFixed(1)} m</div>
          <div className="mono">{metrics.currentSpeedKmh.toFixed(1)} km/h</div>
          <div className="mono">
            Contact L:{metrics.footContacts.left ? '1' : '0'} R:{metrics.footContacts.right ? '1' : '0'}
          </div>
        </div>
      </div>
    </>
  )
}

function ActivationBar(props: { value: number }) {
  const v = Math.max(-1, Math.min(1, props.value))
  const left = Math.round(((v + 1) / 2) * 100)
  return (
    <div className="activationTrack">
      <div className="activationMid" />
      <div className="activationDot" style={{ left: `${left}%` }} />
    </div>
  )
}

function drawSparkline(canvas: HTMLCanvasElement | null, values: number[]): void {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)

  if (values.length < 2) return
  const max = Math.max(...values, 1)
  const min = 0
  const scaleY = (v: number) => h - ((v - min) / (max - min || 1)) * h

  ctx.strokeStyle = 'rgba(255,255,255,0.9)'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < values.length; i++) {
    const x = (i / (values.length - 1)) * w
    const y = scaleY(values[i])
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}
