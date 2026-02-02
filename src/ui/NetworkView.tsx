import type { NetworkGraph } from '../simulation/types'

const nodeColor = (t: NetworkGraph['nodes'][number]['type']) => {
  switch (t) {
    case 'input':
    case 'bias':
      return '#60a5fa'
    case 'output':
      return '#f87171'
    case 'hidden':
    default:
      return '#9ca3af'
  }
}

export function NetworkView(props: { graph?: NetworkGraph }) {
  const graph = props.graph
  if (!graph || graph.nodes.length === 0) return <div className="muted">(no network)</div>

  const width = 260
  const height = 160
  const padX = 12
  const padY = 10

  const layers = [...new Set(graph.nodes.map((n) => n.layer))].sort((a, b) => a - b)
  const layerX = new Map<number, number>()
  layers.forEach((l, i) => {
    const t = layers.length === 1 ? 0.5 : i / (layers.length - 1)
    layerX.set(l, padX + t * (width - padX * 2))
  })

  const nodesByLayer = new Map<number, NetworkGraph['nodes']>()
  for (const n of graph.nodes) {
    const arr = nodesByLayer.get(n.layer) ?? []
    arr.push(n)
    nodesByLayer.set(n.layer, arr)
  }

  const posById = new Map<number, { x: number; y: number; type: NetworkGraph['nodes'][number]['type'] }>()
  for (const l of layers) {
    const xs = layerX.get(l) ?? 0
    const nodes = (nodesByLayer.get(l) ?? []).slice().sort((a, b) => a.type.localeCompare(b.type) || a.id - b.id)
    const count = nodes.length
    for (let i = 0; i < count; i++) {
      const t = (i + 1) / (count + 1)
      const y = padY + t * (height - padY * 2)
      posById.set(nodes[i].id, { x: xs, y, type: nodes[i].type })
    }
  }

  const connections = graph.connections.filter((c) => c.enabled)

  return (
    <svg
      className="networkSvg"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
    >
      {connections.map((c, idx) => {
        const a = posById.get(c.from)
        const b = posById.get(c.to)
        if (!a || !b) return null
        const w = Math.min(4, Math.max(0.5, Math.abs(c.weight) * 1.2))
        const col = c.weight >= 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)'
        return (
          <line
            key={idx}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={col}
            strokeWidth={w}
            strokeLinecap="round"
          />
        )
      })}

      {[...posById.entries()].map(([id, p]) => (
        <circle key={id} cx={p.x} cy={p.y} r={4} fill={nodeColor(p.type)} stroke="#111827" />
      ))}
    </svg>
  )
}
