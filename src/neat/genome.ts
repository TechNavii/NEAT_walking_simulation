import { InnovationTracker } from './innovation'

export type NodeType = 'input' | 'bias' | 'hidden' | 'output'

export type NodeGene = {
  id: number
  type: NodeType
  layer: number
}

export type ConnectionGene = {
  innovation: number
  from: number
  to: number
  weight: number
  enabled: boolean
}

export type GenomeInit = {
  inputCount: number
  outputCount: number
}

export class Genome {
  readonly inputCount: number
  readonly outputCount: number
  readonly inputIds: number[]
  readonly outputIds: number[]
  readonly biasId: number
  nodes: NodeGene[]
  connections: ConnectionGene[]
  fitness = 0

  constructor(args: {
    inputCount: number
    outputCount: number
    inputIds: number[]
    outputIds: number[]
    biasId: number
    nodes: NodeGene[]
    connections: ConnectionGene[]
  }) {
    this.inputCount = args.inputCount
    this.outputCount = args.outputCount
    this.inputIds = args.inputIds
    this.outputIds = args.outputIds
    this.biasId = args.biasId
    this.nodes = args.nodes
    this.connections = args.connections
  }

  static minimal(init: GenomeInit, tracker: InnovationTracker, rng = Math.random): Genome {
    const inputIds = Array.from({ length: init.inputCount }, (_, i) => i)
    const biasId = init.inputCount
    const outputIds = Array.from({ length: init.outputCount }, (_, i) => init.inputCount + 1 + i)
    const nodes: NodeGene[] = [
      ...inputIds.map((id) => ({ id, type: 'input' as const, layer: 0 })),
      { id: biasId, type: 'bias' as const, layer: 0 },
      ...outputIds.map((id) => ({ id, type: 'output' as const, layer: 1 })),
    ]

    // CRITICAL FIX: Very sparse initialization following original NEAT paper
    // Start with minimal connections - let NEAT discover structure through evolution
    // This creates room for meaningful structural exploration
    const connections: ConnectionGene[] = []
    
    // Bias to all outputs (baseline activation)
    for (const to of outputIds) {
      connections.push({
        innovation: tracker.getInnovation(biasId, to),
        from: biasId,
        to,
        weight: (rng() * 2 - 1) * 0.3,
        enabled: true,
      })
    }
    
    // Phase inputs (last 2: sin/cos) - only connect to LEG outputs (5-10)
    // This is the core timing signal for walking
    const sinPhaseId = inputIds[inputIds.length - 2]
    const cosPhaseId = inputIds[inputIds.length - 1]
    const legOutputIds = outputIds.slice(5) // hipL, kneeL, ankleL, hipR, kneeR, ankleR
    
    for (const to of legOutputIds) {
      connections.push({
        innovation: tracker.getInnovation(sinPhaseId, to),
        from: sinPhaseId,
        to,
        weight: (rng() * 2 - 1) * 1.2,
        enabled: true,
      })
    }
    
    // cos(phase) only to knee outputs for phase offset
    const kneeOutputIds = [outputIds[6], outputIds[9]] // kneeL, kneeR
    for (const to of kneeOutputIds) {
      connections.push({
        innovation: tracker.getInnovation(cosPhaseId, to),
        from: cosPhaseId,
        to,
        weight: (rng() * 2 - 1) * 0.8,
        enabled: true,
      })
    }
    
    // Torso angle (input 22) to hip outputs for balance
    const torsoAngleId = inputIds[22]
    const hipOutputIds = [outputIds[5], outputIds[8]] // hipL, hipR
    for (const to of hipOutputIds) {
      connections.push({
        innovation: tracker.getInnovation(torsoAngleId, to),
        from: torsoAngleId,
        to,
        weight: (rng() * 2 - 1) * 0.6,
        enabled: true,
      })
    }
    
    // Total: ~11 (bias) + 6 (sin) + 2 (cos) + 2 (torso) = ~21 connections
    // Much sparser than before (~115), leaves room for evolution

    return new Genome({
      inputCount: init.inputCount,
      outputCount: init.outputCount,
      inputIds,
      outputIds,
      biasId,
      nodes,
      connections,
    })
  }
  
  // Create genome with hidden nodes pre-seeded
  static withHiddenNodes(
    init: GenomeInit, 
    tracker: InnovationTracker, 
    hiddenCount: number,
    rng = Math.random
  ): Genome {
    const genome = Genome.minimal(init, tracker, rng)
    
    // Add hidden nodes by splitting random connections
    for (let i = 0; i < hiddenCount; i++) {
      const enabledConns = genome.connections.filter(c => c.enabled)
      if (enabledConns.length === 0) break
      
      const conn = enabledConns[Math.floor(rng() * enabledConns.length)]
      conn.enabled = false
      
      const fromNode = genome.nodes.find(n => n.id === conn.from)
      const toNode = genome.nodes.find(n => n.id === conn.to)
      if (!fromNode || !toNode) continue
      
      const split = tracker.getSplitRecord(conn.innovation, conn.from, conn.to)
      const layer = (fromNode.layer + toNode.layer) / 2
      
      if (!genome.nodes.some(n => n.id === split.newNodeId)) {
        genome.nodes.push({ id: split.newNodeId, type: 'hidden', layer })
      }
      
      if (!genome.connections.some(c => c.innovation === split.inInnovation)) {
        genome.connections.push({
          innovation: split.inInnovation,
          from: conn.from,
          to: split.newNodeId,
          weight: 1,
          enabled: true,
        })
      }
      if (!genome.connections.some(c => c.innovation === split.outInnovation)) {
        genome.connections.push({
          innovation: split.outInnovation,
          from: split.newNodeId,
          to: conn.to,
          weight: conn.weight,
          enabled: true,
        })
      }
    }
    
    return genome
  }

  clone(): Genome {
    return new Genome({
      inputCount: this.inputCount,
      outputCount: this.outputCount,
      inputIds: [...this.inputIds],
      outputIds: [...this.outputIds],
      biasId: this.biasId,
      nodes: this.nodes.map((n) => ({ ...n })),
      connections: this.connections.map((c) => ({ ...c })),
    })
  }

  buildNetwork(): Network {
    return Network.fromGenome(this)
  }
}

export class Network {
  private orderedNodes: NodeGene[]
  private incoming: Array<Array<{ fromIndex: number; weight: number }>>
  private outputNodeIndices: number[]
  private biasIndex: number
  private inputIndices: number[]
  private values: number[]

  private constructor(args: {
    orderedNodes: NodeGene[]
    incoming: Array<Array<{ fromIndex: number; weight: number }>>
    outputNodeIndices: number[]
    biasIndex: number
    inputIndices: number[]
  }) {
    this.orderedNodes = args.orderedNodes
    this.incoming = args.incoming
    this.outputNodeIndices = args.outputNodeIndices
    this.biasIndex = args.biasIndex
    this.inputIndices = args.inputIndices
    this.values = new Array(args.orderedNodes.length).fill(0)
  }

  static fromGenome(genome: Genome): Network {
    const orderedNodes = [...genome.nodes].sort((a, b) => a.layer - b.layer || a.id - b.id)
    const nodeIndexById = new Map<number, number>()
    orderedNodes.forEach((n, i) => nodeIndexById.set(n.id, i))

    const incoming: Array<Array<{ fromIndex: number; weight: number }>> = Array.from(
      { length: orderedNodes.length },
      () => [],
    )

    for (const c of genome.connections) {
      if (!c.enabled) continue
      const toIndex = nodeIndexById.get(c.to)
      const fromIndex = nodeIndexById.get(c.from)
      if (toIndex == null || fromIndex == null) continue
      incoming[toIndex].push({ fromIndex, weight: c.weight })
    }

    const outputNodeIndices = genome.outputIds
      .map((id) => nodeIndexById.get(id))
      .filter((v): v is number => v != null)
    const biasIndex = nodeIndexById.get(genome.biasId) ?? -1
    const inputIndices = genome.inputIds
      .map((id) => nodeIndexById.get(id))
      .filter((v): v is number => v != null)

    return new Network({
      orderedNodes,
      incoming,
      outputNodeIndices,
      biasIndex,
      inputIndices,
    })
  }

  activate(inputs: number[]): number[] {
    if (inputs.length !== this.inputIndices.length) {
      throw new Error(`Expected ${this.inputIndices.length} inputs, got ${inputs.length}`)
    }

    this.values.fill(0)
    for (let i = 0; i < this.inputIndices.length; i++) {
      this.values[this.inputIndices[i]] = inputs[i]
    }
    if (this.biasIndex >= 0) this.values[this.biasIndex] = 1

    for (let i = 0; i < this.orderedNodes.length; i++) {
      const node = this.orderedNodes[i]
      if (node.type === 'input' || node.type === 'bias') continue
      let sum = 0
      for (const edge of this.incoming[i]) {
        sum += this.values[edge.fromIndex] * edge.weight
      }
      this.values[i] = Math.tanh(sum)
    }

    return this.outputNodeIndices.map((i) => this.values[i])
  }
}
