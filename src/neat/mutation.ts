import { Genome, type ConnectionGene, type NodeGene } from './genome'
import { InnovationTracker } from './innovation'

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))

const hasConnection = (connections: ConnectionGene[], from: number, to: number) =>
  connections.some((c) => c.from === from && c.to === to)

export function mutateWeights(genome: Genome, mutationRate: number, rng = Math.random): void {
  mutateWeightsWithOptions(genome, mutationRate, rng)
}

export type WeightMutationOptions = {
  // Probability of perturbing the existing weight vs. resetting it.
  // Higher values preserve existing behavior and allow fine-tuning.
  perturbChance?: number
  // Max uniform delta applied when perturbing.
  perturbScale?: number
  // Reset scale (new weight sampled uniformly in [-resetScale, +resetScale]).
  resetScale?: number
  // If provided, overrides the per-connection mutation probability so that the
  // expected number of mutated weights stays roughly constant as networks grow.
  // This prevents mutations from becoming more destructive as connection count increases.
  mutationsPerGenome?: number
}

export function mutateWeightsWithOptions(
  genome: Genome,
  mutationRate: number,
  rng = Math.random,
  opts: WeightMutationOptions = {},
): void {
  // Defaults are intentionally conservative for continuous control.
  // Large weight jumps under high mutation rates destroy promising gaits.
  const perturbChance = opts.perturbChance ?? 0.97
  const perturbScale = opts.perturbScale ?? 0.25
  const resetScale = opts.resetScale ?? 0.9

  const effectiveMutationRate =
    opts.mutationsPerGenome != null
      ? genome.connections.length === 0
        ? 0
        : Math.min(1, Math.max(0, opts.mutationsPerGenome) / genome.connections.length)
      : mutationRate

  for (const c of genome.connections) {
    if (rng() > effectiveMutationRate) continue
    const perturb = rng() < perturbChance
    if (perturb) {
      c.weight = clamp(c.weight + (rng() * 2 - 1) * perturbScale, -5, 5)
    } else {
      c.weight = clamp((rng() * 2 - 1) * resetScale, -5, 5)
    }
  }
}

export function mutateToggleConnection(genome: Genome, rng = Math.random): void {
  if (genome.connections.length === 0) return
  const idx = Math.floor(rng() * genome.connections.length)
  genome.connections[idx].enabled = !genome.connections[idx].enabled
}

export function mutateAddConnection(genome: Genome, tracker: InnovationTracker, rng = Math.random): void {
  const fromCandidates = genome.nodes.filter((n) => n.type !== 'output')
  const toCandidates = genome.nodes.filter((n) => n.type !== 'input' && n.type !== 'bias')
  if (fromCandidates.length === 0 || toCandidates.length === 0) return

  for (let attempt = 0; attempt < 30; attempt++) {
    const from = fromCandidates[Math.floor(rng() * fromCandidates.length)]
    const to = toCandidates[Math.floor(rng() * toCandidates.length)]
    if (from.id === to.id) continue
    if (from.layer >= to.layer) continue
    if (hasConnection(genome.connections, from.id, to.id)) continue
    genome.connections.push({
      innovation: tracker.getInnovation(from.id, to.id),
      from: from.id,
      to: to.id,
      weight: clamp((rng() * 2 - 1) * 1.5, -5, 5),
      enabled: true,
    })
    return
  }
}

export function mutateAddNode(genome: Genome, tracker: InnovationTracker, rng = Math.random): void {
  const enabledConnections = genome.connections.filter((c) => c.enabled)
  if (enabledConnections.length === 0) return
  const connection = enabledConnections[Math.floor(rng() * enabledConnections.length)]
  connection.enabled = false

  const fromNode = genome.nodes.find((n) => n.id === connection.from)
  const toNode = genome.nodes.find((n) => n.id === connection.to)
  if (!fromNode || !toNode) return

  const split = tracker.getSplitRecord(connection.innovation, connection.from, connection.to)
  const layer = (fromNode.layer + toNode.layer) / 2

  if (!genome.nodes.some((n) => n.id === split.newNodeId)) {
    const newNode: NodeGene = { id: split.newNodeId, type: 'hidden', layer }
    genome.nodes.push(newNode)
  }

  if (!genome.connections.some((c) => c.innovation === split.inInnovation)) {
    genome.connections.push({
      innovation: split.inInnovation,
      from: connection.from,
      to: split.newNodeId,
      weight: 1,
      enabled: true,
    })
  }
  if (!genome.connections.some((c) => c.innovation === split.outInnovation)) {
    genome.connections.push({
      innovation: split.outInnovation,
      from: split.newNodeId,
      to: connection.to,
      weight: connection.weight,
      enabled: true,
    })
  }
}

// Mirror pairs for symmetric walking: (leftOutput, rightOutput) indices
// Output order: neck, shoulderL, elbowL, shoulderR, elbowR, hipL, kneeL, ankleL, hipR, kneeR, ankleR
const LEG_MIRROR_PAIRS: [number, number][] = [
  [5, 8],   // hipL <-> hipR
  [6, 9],   // kneeL <-> kneeR
  [7, 10],  // ankleL <-> ankleR
]

// Arm mirror pairs (for future use in arm swing coordination)
// const ARM_MIRROR_PAIRS: [number, number][] = [
//   [1, 3],   // shoulderL <-> shoulderR
//   [2, 4],   // elbowL <-> elbowR
// ]

export function mutateMirrorWeights(genome: Genome, rng = Math.random): void {
  // With some probability, make leg weights more symmetric (opposing phase)
  // This helps discover alternating gait patterns
  const outputIdStart = genome.inputCount + 1
  
  for (const [leftIdx, rightIdx] of LEG_MIRROR_PAIRS) {
    const leftOutputId = outputIdStart + leftIdx
    const rightOutputId = outputIdStart + rightIdx
    
    // Find connections to these outputs
    const leftConns = genome.connections.filter(c => c.to === leftOutputId && c.enabled)
    const rightConns = genome.connections.filter(c => c.to === rightOutputId && c.enabled)
    
    // For each left connection, try to find/create matching right connection with opposite sign
    for (const lc of leftConns) {
      const matchingRight = rightConns.find(rc => rc.from === lc.from)
      if (matchingRight) {
        // Nudge toward opposite values (anti-phase for walking)
        const target = -lc.weight
        matchingRight.weight = clamp(
          matchingRight.weight + (target - matchingRight.weight) * 0.3 * rng(),
          -5, 5
        )
      }
    }
  }
}

export function mutateGenome(
  genome: Genome,
  tracker: InnovationTracker,
  opts: {
    mutationRate: number
    addNodeProbability: number
    addConnectionProbability: number
    weightMutation?: WeightMutationOptions
    allowStructural?: boolean
    allowToggle?: boolean
    allowMirror?: boolean
  },
  rng = Math.random,
): void {
  mutateWeightsWithOptions(genome, opts.mutationRate, rng, opts.weightMutation)

  const allowStructural = opts.allowStructural ?? true
  const allowToggle = opts.allowToggle ?? true
  const allowMirror = opts.allowMirror ?? true

  if (allowStructural && rng() < opts.addConnectionProbability) mutateAddConnection(genome, tracker, rng)
  if (allowStructural && rng() < opts.addNodeProbability) mutateAddNode(genome, tracker, rng)
  if (allowToggle && rng() < 0.02) mutateToggleConnection(genome, rng)
  // Occasionally apply mirror mutation to encourage symmetric gait
  if (allowMirror && rng() < 0.08) mutateMirrorWeights(genome, rng)
}
