import { Genome, type ConnectionGene } from './genome'

const sortByInnovation = (a: ConnectionGene, b: ConnectionGene) => a.innovation - b.innovation

export function compatibilityDistance(a: Genome, b: Genome): number {
  const aGenes = [...a.connections].sort(sortByInnovation)
  const bGenes = [...b.connections].sort(sortByInnovation)

  let i = 0
  let j = 0
  let matching = 0
  let weightDiff = 0
  let disjoint = 0
  let excess = 0

  const aMax = aGenes.at(-1)?.innovation ?? 0
  const bMax = bGenes.at(-1)?.innovation ?? 0

  while (i < aGenes.length && j < bGenes.length) {
    const ga = aGenes[i]
    const gb = bGenes[j]
    if (ga.innovation === gb.innovation) {
      matching++
      weightDiff += Math.abs(ga.weight - gb.weight)
      i++
      j++
      continue
    }
    if (ga.innovation < gb.innovation) {
      if (ga.innovation > bMax) excess++
      else disjoint++
      i++
      continue
    }
    if (gb.innovation > aMax) excess++
    else disjoint++
    j++
  }

  if (i < aGenes.length) {
    const remaining = aGenes.length - i
    excess += remaining
  }
  if (j < bGenes.length) {
    const remaining = bGenes.length - j
    excess += remaining
  }

  // Normalize by genome size - standard NEAT approach
  const n = Math.max(aGenes.length, bGenes.length)
  const normalizer = n < 20 ? 1 : n
  const avgWeightDiff = matching === 0 ? 0 : weightDiff / matching

  // Count hidden node differences (smaller weight to avoid species explosion)
  const aHidden = a.nodes.filter(n => n.type === 'hidden').length
  const bHidden = b.nodes.filter(n => n.type === 'hidden').length
  const hiddenNodeDiff = Math.abs(aHidden - bHidden)
  
  // Standard NEAT coefficients - less aggressive to reduce species count
  const c1 = 1.0   // excess weight
  const c2 = 1.0   // disjoint weight
  const c3 = 0.4   // weight diff
  const c4 = 0.3   // hidden node difference (reduced from 2.0 - was causing species explosion)
  
  const baseDistance = (c1 * excess) / normalizer + (c2 * disjoint) / normalizer + c3 * avgWeightDiff
  const structuralDistance = c4 * hiddenNodeDiff
  
  return baseDistance + structuralDistance
}

export class Species {
  readonly id: number
  representative: Genome
  genomes: Genome[] = []
  adjustedFitnessSum = 0

  constructor(id: number, representative: Genome) {
    this.id = id
    this.representative = representative
  }

  resetForNextGen(): void {
    this.genomes = []
    this.adjustedFitnessSum = 0
  }
}
