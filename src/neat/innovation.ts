export type ConnectionKey = `${number}->${number}`

export type SplitRecord = {
  newNodeId: number
  inInnovation: number
  outInnovation: number
}

export class InnovationTracker {
  private nextInnovation = 1
  private nextNodeId: number
  private connInnovations = new Map<ConnectionKey, number>()
  private splitByInnovation = new Map<number, SplitRecord>()

  constructor(opts: { nextNodeId: number }) {
    this.nextNodeId = opts.nextNodeId
  }

  getInnovation(from: number, to: number): number {
    const key: ConnectionKey = `${from}->${to}`
    const existing = this.connInnovations.get(key)
    if (existing != null) return existing
    const created = this.nextInnovation++
    this.connInnovations.set(key, created)
    return created
  }

  getSplitRecord(connectionInnovation: number, from: number, to: number): SplitRecord {
    const existing = this.splitByInnovation.get(connectionInnovation)
    if (existing) return existing
    const record: SplitRecord = {
      newNodeId: this.nextNodeId++,
      inInnovation: this.getInnovation(from, this.nextNodeId - 1),
      outInnovation: this.getInnovation(this.nextNodeId - 1, to),
    }
    this.splitByInnovation.set(connectionInnovation, record)
    return record
  }

  allocateNodeId(): number {
    return this.nextNodeId++
  }
}
