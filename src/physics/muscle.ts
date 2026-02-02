const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

export class Muscle {
  energy: number
  activation = 0

  constructor(energyReserve: number) {
    this.energy = energyReserve
  }

  setActivation(v: number): void {
    this.activation = clamp01(v)
  }
}

export class MusclePair {
  readonly flexor: Muscle
  readonly extensor: Muscle
  output = 0

  constructor(energyReserve: number) {
    this.flexor = new Muscle(energyReserve)
    this.extensor = new Muscle(energyReserve)
  }

  setOutput(output: number): void {
    this.output = Math.max(-1, Math.min(1, output))
    this.flexor.setActivation(Math.max(0, this.output))
    this.extensor.setActivation(Math.max(0, -this.output))
  }

  step(dtSeconds: number, energyReserve: number, recoveryRate: number, costScale = 1): void {
    for (const m of [this.flexor, this.extensor]) {
      const use = (m.activation * m.activation) * costScale * dtSeconds
      const recover = (1 - m.activation) * recoveryRate * dtSeconds
      m.energy = Math.max(0, Math.min(energyReserve, m.energy - use + recover))
    }
  }

  energyRatioForCurrentDirection(energyReserve: number): number {
    if (energyReserve <= 0) return 0
    const m = this.output >= 0 ? this.flexor : this.extensor
    return Math.max(0, Math.min(1, m.energy / energyReserve))
  }

  combinedEnergyRatio(energyReserve: number): number {
    if (energyReserve <= 0) return 0
    return Math.max(
      0,
      Math.min(1, (this.flexor.energy + this.extensor.energy) / (2 * energyReserve)),
    )
  }
}
