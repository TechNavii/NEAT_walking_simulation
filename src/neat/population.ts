import { Genome } from './genome'
import { InnovationTracker } from './innovation'
import { mutateGenome } from './mutation'
import { Species, compatibilityDistance } from './species'

export type PopulationConfig = {
  populationSize: number
  mutationRate: number
  addNodeProbability: number
  addConnectionProbability: number
  compatibilityThreshold: number
  elitism: number
  injectDiversity?: boolean
  stagnationLevel?: number  // 0-3: how severely stagnated we are
  hallOfFameGenome?: Genome
}

const byFitnessDesc = (a: Genome, b: Genome) => b.fitness - a.fitness

export class Population {
  readonly inputCount: number
  readonly outputCount: number
  readonly tracker: InnovationTracker
  generation = 1
  genomes: Genome[]
  species: Species[] = []
  private nextSpeciesId = 1
  
  // Dynamic compatibility threshold adjustment
  private compatibilityThreshold: number = 3.0  // Start higher to avoid species explosion
  private readonly minSpeciesTarget = 5
  private readonly maxSpeciesTarget = 12

  constructor(args: { inputCount: number; outputCount: number; config: PopulationConfig; rng?: () => number }) {
    this.inputCount = args.inputCount
    this.outputCount = args.outputCount
    const nextNodeId = args.inputCount + 1 + args.outputCount
    this.tracker = new InnovationTracker({ nextNodeId })
    const rng = args.rng ?? Math.random
    const popSize = args.config.populationSize
    
    // CRITICAL: Seed population with diverse topologies
    // This ensures hidden nodes exist from the start and creates multiple species
    const minimalCount = Math.floor(popSize * 0.35)      // 35% minimal (no hidden)
    const oneHiddenCount = Math.floor(popSize * 0.30)    // 30% with 1 hidden node
    const twoHiddenCount = Math.floor(popSize * 0.20)    // 20% with 2 hidden nodes
    const threeHiddenCount = popSize - minimalCount - oneHiddenCount - twoHiddenCount // 15% with 3 hidden
    
    const init = { inputCount: args.inputCount, outputCount: args.outputCount }
    
    const minimalGenomes = Array.from({ length: minimalCount }, () =>
      Genome.minimal(init, this.tracker, rng)
    )
    
    const oneHiddenGenomes = Array.from({ length: oneHiddenCount }, () =>
      Genome.withHiddenNodes(init, this.tracker, 1, rng)
    )
    
    const twoHiddenGenomes = Array.from({ length: twoHiddenCount }, () =>
      Genome.withHiddenNodes(init, this.tracker, 2, rng)
    )
    
    const threeHiddenGenomes = Array.from({ length: threeHiddenCount }, () =>
      Genome.withHiddenNodes(init, this.tracker, 3, rng)
    )
    
    this.genomes = [
      ...minimalGenomes,
      ...oneHiddenGenomes,
      ...twoHiddenGenomes,
      ...threeHiddenGenomes,
    ]
    
    // Apply walking pattern bias to some genomes
    const walkingBiasCount = Math.floor(popSize * 0.15)
    for (let i = 0; i < walkingBiasCount; i++) {
      const idx = Math.floor(rng() * this.genomes.length)
      this.seedWalkingPattern(this.genomes[idx], rng)
    }
    
    console.log(`[NEAT] Initialized population: ${minimalCount} minimal, ${oneHiddenCount} 1-hidden, ${twoHiddenCount} 2-hidden, ${threeHiddenCount} 3-hidden`)
  }
  
  private seedWalkingPattern(genome: Genome, rng: () => number): void {
    const outputIdStart = this.inputCount + 1
    const sinPhaseId = this.inputCount - 2
    
    // Set up anti-phase weights for left/right legs
    const legPairs: [number, number][] = [
      [5, 8],   // hipL, hipR
      [6, 9],   // kneeL, kneeR
      [7, 10],  // ankleL, ankleR
    ]
    
    for (const conn of genome.connections) {
      for (const [leftIdx, rightIdx] of legPairs) {
        const leftOutputId = outputIdStart + leftIdx
        const rightOutputId = outputIdStart + rightIdx
        
        if (conn.from === sinPhaseId) {
          if (conn.to === leftOutputId) {
            conn.weight = 0.8 + rng() * 0.4
          } else if (conn.to === rightOutputId) {
            conn.weight = -(0.8 + rng() * 0.4)
          }
        }
      }
    }
  }

  speciate(): void {
    for (const s of this.species) s.resetForNextGen()
    const aliveSpecies: Species[] = []

    // Use dynamic threshold that adjusts to maintain species diversity
    const threshold = this.compatibilityThreshold

    for (const g of this.genomes) {
      let placed = false
      for (const s of aliveSpecies) {
        if (compatibilityDistance(g, s.representative) < threshold) {
          s.genomes.push(g)
          placed = true
          break
        }
      }
      if (!placed) {
        const sp = new Species(this.nextSpeciesId++, g.clone())
        sp.genomes.push(g)
        aliveSpecies.push(sp)
      }
    }

    for (const s of aliveSpecies) {
      const size = s.genomes.length
      s.adjustedFitnessSum = s.genomes.reduce((sum, g) => sum + g.fitness / size, 0)
      const rep = s.genomes[Math.floor(Math.random() * s.genomes.length)]
      s.representative = rep.clone()
    }

    this.species = aliveSpecies
    
    // CRITICAL: Dynamically adjust threshold to maintain target species count
    // This ensures we always have diversity for innovation
    const speciesCount = this.species.length
    if (speciesCount < this.minSpeciesTarget) {
      // Too few species - lower threshold to create more separation
      this.compatibilityThreshold = Math.max(0.5, this.compatibilityThreshold - 0.3)
      console.log(`[NEAT] Species: ${speciesCount} < ${this.minSpeciesTarget}, threshold -> ${this.compatibilityThreshold.toFixed(2)}`)
    } else if (speciesCount > this.maxSpeciesTarget) {
      // Too many species - raise threshold AGGRESSIVELY to consolidate
      const overshoot = speciesCount - this.maxSpeciesTarget
      const adjustment = Math.min(0.5, 0.15 + overshoot * 0.02)  // More aggressive when way over
      this.compatibilityThreshold = Math.min(5.0, this.compatibilityThreshold + adjustment)
      console.log(`[NEAT] Species: ${speciesCount} > ${this.maxSpeciesTarget}, threshold -> ${this.compatibilityThreshold.toFixed(2)}`)
    }
  }

  evolve(config: PopulationConfig, rng = Math.random): void {
    this.genomes.sort(byFitnessDesc)

    // NOTE: Fitness sharing can prevent incremental local search if too aggressive.
    // Only enable it at high stagnation levels to push diversity without collapsing
    // exploitation around a promising gait.
    const stagnation = config.stagnationLevel ?? 0
    if (stagnation >= 4) {
      this.applyFitnessSharing(stagnation)
    }
    
    // Re-sort after fitness sharing
    this.genomes.sort(byFitnessDesc)

    this.speciate()

    // =============================================================
    // NUCLEAR RESET: Level 5 stagnation (50+ generations)
    // Keep only top 5 genomes, regenerate entire rest of population
    // =============================================================
    if (stagnation >= 5) {
      console.log(`%c[NEAT] ☢️ NUCLEAR RESET ☢️ - Keeping top 5, regenerating ${config.populationSize - 5} genomes!`, 
        'color: #e91e63; font-weight: bold; font-size: 16px')
      
      const top5 = this.genomes.slice(0, 5).map(g => g.clone())
      const next: Genome[] = top5
      if (config.hallOfFameGenome) {
        next.push(config.hallOfFameGenome.clone())
      }
      
      const init = { inputCount: this.inputCount, outputCount: this.outputCount }
      const remaining = Math.max(0, config.populationSize - next.length)
      
      for (let i = 0; i < remaining; i++) {
        // Create diverse topologies
        const hiddenCount = 1 + Math.floor(rng() * 5)  // 1-5 hidden nodes
        const fresh = Genome.withHiddenNodes(init, this.tracker, hiddenCount, rng)
        
        // 50% get walking pattern bias
        if (rng() < 0.5) {
          this.seedWalkingPattern(fresh, rng)
        }
        
        // 30% are mutated clones of top 5 (to explore nearby solutions)
        if (rng() < 0.3) {
          const elite = top5[Math.floor(rng() * top5.length)]
          const mutated = elite.clone()
          // Exploration-heavy mutation to escape local optimum.
          // Keep it strong, but avoid "randomizing everything" (gait control is brittle).
          for (let m = 0; m < 3; m++) {
            mutateGenome(mutated, this.tracker, {
              mutationRate: 0.6,
              addNodeProbability: 0.25,
              addConnectionProbability: 0.35,
              weightMutation: { perturbChance: 0.9, perturbScale: 0.35, resetScale: 1.2 },
            }, rng)
          }
          next.push(mutated)
        } else {
          next.push(fresh)
        }
      }
      
      this.genomes = next
      this.generation++
      return  // Skip normal evolution for nuclear reset
    }
    
    // CRITICAL FIX: ALWAYS preserve at least 2 elites to prevent catastrophic forgetting
    // Even during "extinction events", keep the best solutions alive
    const minElitism = 2  // NEVER go below this!
    const effectiveElitism = Math.max(minElitism, config.elitism)
    
    const elites = this.genomes.slice(0, Math.max(0, Math.min(effectiveElitism, config.populationSize)))
    const next: Genome[] = elites.map((g) => g.clone())
    if (config.hallOfFameGenome) {
      next.push(config.hallOfFameGenome.clone())
    }

    // =============================================================
    // LOCAL SEARCH LANE: Fine-tune around current elites (weight-only)
    // Continuous control improvements are tiny; global mutation destroys them.
    // =============================================================
    const localSearchFraction = stagnation >= 4 ? 0.25 :
                               stagnation >= 3 ? 0.22 :
                               stagnation >= 2 ? 0.18 :
                               stagnation >= 1 ? 0.12 : 0.06
    const localSearchTarget = Math.floor(config.populationSize * localSearchFraction)
    const localSearchCount = Math.min(localSearchTarget, Math.max(0, config.populationSize - next.length))
    if (localSearchCount > 0) {
      const parents = this.genomes.slice(0, Math.min(5, this.genomes.length))
      for (let i = 0; i < localSearchCount; i++) {
        const parent = parents[Math.floor(rng() * parents.length)]
        const child = parent.clone()
        mutateGenome(child, this.tracker, {
          // Fine-tune a SMALL subset of weights so mutations stay local even as
          // networks gain more connections.
          mutationRate: 1.0,
          addNodeProbability: 0,
          addConnectionProbability: 0,
          allowStructural: false,
          allowToggle: false,
          allowMirror: false,
          weightMutation: {
            mutationsPerGenome: 6,
            perturbChance: 1.0,
            perturbScale: 0.03,
            resetScale: 0.03,
          },
        }, rng)
        next.push(child)
      }
      console.log(`[NEAT] Local search: +${localSearchCount} fine-tune clones (${(localSearchFraction * 100).toFixed(0)}%)`)
    }

    // DIVERSITY INJECTION: Keep modest.
    // Too much injection collapses average fitness and prevents incremental refinement.
    // Stagnation 1: 8% injection
    // Stagnation 2: 12% injection
    // Stagnation 3: 18% injection
    // Stagnation 4: 25% injection
    const injectFraction = stagnation >= 4 ? 0.25 :
                          stagnation >= 3 ? 0.18 :
                          stagnation >= 2 ? 0.12 :
                          stagnation >= 1 ? 0.08 :
                          config.injectDiversity ? 0.06 : 0
    const injectCount = Math.floor(config.populationSize * injectFraction)
    
    if (injectCount > 0) {
      console.log(`%c[NEAT] Diversity injection (level ${stagnation}): Adding ${injectCount} fresh genomes (${(injectFraction*100).toFixed(0)}%) - elites preserved!`, 
        'color: #E91E63; font-weight: bold')
      
      const init = { inputCount: this.inputCount, outputCount: this.outputCount }
      for (let i = 0; i < injectCount; i++) {
        // Moderate complexity for fresh genomes
        const minHidden = stagnation >= 3 ? 2 : 1
        const maxHidden = stagnation >= 3 ? 5 : 4
        const hiddenCount = minHidden + Math.floor(rng() * (maxHidden - minHidden + 1))
        const fresh = Genome.withHiddenNodes(init, this.tracker, hiddenCount, rng)
        // Apply walking bias to help fresh genomes
        if (rng() < 0.4) {
          this.seedWalkingPattern(fresh, rng)
        }
        next.push(fresh)
      }
    }

    // =============================================================
    // MUTATION LANES
    // - Refinement: weight-only, small moves (good for gait fine-tuning)
    // - Exploration: structural + larger weight moves (new strategies)
    // =============================================================
    const explorationChance = stagnation >= 4 ? 0.35 :
                             stagnation >= 3 ? 0.30 :
                             stagnation >= 2 ? 0.25 :
                             stagnation >= 1 ? 0.20 : 0.15

    const structuralBoost = stagnation >= 4 ? 2.2 :
                           stagnation >= 3 ? 1.8 :
                           stagnation >= 2 ? 1.5 :
                           stagnation >= 1 ? 1.3 : 1.2

    const refinementMutation = {
      mutationRate: 1.0,
      addNodeProbability: 0,
      addConnectionProbability: 0,
      allowStructural: false,
      allowToggle: false,
      allowMirror: false,
      weightMutation: {
        mutationsPerGenome: 8,
        perturbChance: 1.0,
        perturbScale: 0.08,
        resetScale: 0.08,
      },
    } satisfies Parameters<typeof mutateGenome>[2]

    const explorationMutation = {
      mutationRate: Math.min(0.65, config.mutationRate * 1.25),
      addNodeProbability: Math.min(0.35, config.addNodeProbability * structuralBoost),
      addConnectionProbability: Math.min(0.45, config.addConnectionProbability * structuralBoost),
      allowStructural: true,
      allowToggle: true,
      allowMirror: true,
      weightMutation: { perturbChance: 0.9, perturbScale: 0.35, resetScale: 1.2 },
    } satisfies Parameters<typeof mutateGenome>[2]

    const lanes = {
      explorationChance,
      refinement: refinementMutation,
      exploration: explorationMutation,
    }

    const totalAdjusted = this.species.reduce((sum, s) => sum + s.adjustedFitnessSum, 0) || 1
    const desired = config.populationSize - next.length

    const allocations = this.species.map((s) => {
      const raw = (s.adjustedFitnessSum / totalAdjusted) * desired
      return { species: s, raw, count: Math.floor(raw), frac: raw - Math.floor(raw) }
    })

    let allocated = allocations.reduce((sum, a) => sum + a.count, 0)
    allocations.sort((a, b) => b.frac - a.frac)
    for (let k = 0; allocated < desired && k < allocations.length; k++) {
      allocations[k].count++
      allocated++
    }

    for (const a of allocations) {
      if (a.count <= 0) continue
      const kids = this.makeOffspring(a.species, a.count, lanes, rng)
      next.push(...kids)
    }

    while (next.length < config.populationSize) {
      const g = this.genomes[Math.floor(rng() * this.genomes.length)]
      const child = g.clone()
      const explore = rng() < lanes.explorationChance
      mutateGenome(child, this.tracker, explore ? lanes.exploration : lanes.refinement, rng)
      next.push(child)
    }
    if (next.length > config.populationSize) next.length = config.populationSize

    this.genomes = next
    this.generation++
  }
  
  // Fitness sharing: penalize genomes that are too similar to the best
  // This creates selection pressure for diversity
  private applyFitnessSharing(stagnationLevel: number): void {
    if (this.genomes.length < 2) return
    
    const best = this.genomes[0]
    // Keep sharing mild; we primarily want diversity from explicit exploration lanes.
    const sharingRadius = stagnationLevel >= 5 ? 3.0 : 2.5
    const sharingStrength = stagnationLevel >= 5 ? 0.45 : 0.35
    
    let sharedCount = 0
    for (let i = 1; i < this.genomes.length; i++) {
      const g = this.genomes[i]
      const dist = compatibilityDistance(g, best)
      if (dist < sharingRadius) {
        // This genome is too similar to the best - reduce its fitness
        const proximity = 1 - (dist / sharingRadius)  // 0-1, higher = more similar
        const penalty = proximity * sharingStrength
        g.fitness *= (1 - penalty)
        sharedCount++
      }
    }
    
    if (sharedCount > 0) {
      console.log(`%c[NEAT] Fitness sharing (level ${stagnationLevel}): penalized ${sharedCount} genomes (radius=${sharingRadius.toFixed(1)}, strength=${(sharingStrength*100).toFixed(0)}%)`, 'color: #FF9800')
    }
  }

  private makeOffspring(
    species: Species,
    count: number,
    lanes: {
      explorationChance: number
      refinement: Parameters<typeof mutateGenome>[2]
      exploration: Parameters<typeof mutateGenome>[2]
    },
    rng: () => number,
  ): Genome[] {
    const pool = [...species.genomes].sort(byFitnessDesc)
    const survivors = pool.slice(0, Math.max(2, Math.ceil(pool.length * 0.5)))
    const children: Genome[] = []

    for (let i = 0; i < count; i++) {
      const parentA = this.tournamentSelect(survivors, rng)
      const parentB = this.tournamentSelect(survivors, rng)
      const [fitter, other] = parentA.fitness >= parentB.fitness ? [parentA, parentB] : [parentB, parentA]
      const child = crossover(fitter, other, rng)
      const explore = rng() < lanes.explorationChance
      mutateGenome(child, this.tracker, explore ? lanes.exploration : lanes.refinement, rng)
      children.push(child)
    }

    return children
  }

  private tournamentSelect(genomes: Genome[], rng: () => number): Genome {
    const k = Math.min(3, genomes.length)
    let best = genomes[Math.floor(rng() * genomes.length)]
    for (let i = 1; i < k; i++) {
      const g = genomes[Math.floor(rng() * genomes.length)]
      if (g.fitness > best.fitness) best = g
    }
    return best
  }
}

function crossover(fitter: Genome, other: Genome, rng: () => number): Genome {
  const child = fitter.clone()
  const otherByInnov = new Map(other.connections.map((c) => [c.innovation, c]))
  const fitterByInnov = new Map(fitter.connections.map((c) => [c.innovation, c]))

  // Handle matching genes - take from either parent
  child.connections = fitter.connections.map((c) => {
    const match = otherByInnov.get(c.innovation)
    if (!match) return { ...c }
    const chosen = rng() < 0.5 ? c : match
    const enabled = c.enabled && match.enabled ? true : rng() < 0.25
    return { ...chosen, enabled }
  })
  
  // CRITICAL FIX: Sometimes inherit disjoint/excess genes from less fit parent
  // This allows structural innovations to spread through the population
  const inheritFromOtherProb = 0.2  // 20% chance to inherit each non-matching gene
  for (const c of other.connections) {
    if (!fitterByInnov.has(c.innovation) && rng() < inheritFromOtherProb) {
      // This is a structural innovation in the less fit parent
      child.connections.push({ ...c, enabled: c.enabled && rng() < 0.75 })
    }
  }

  // Collect all required nodes
  const nodesById = new Map(child.nodes.map((n) => [n.id, n]))
  for (const c of child.connections) {
    if (!nodesById.has(c.from)) {
      const fromNode = fitter.nodes.find((n) => n.id === c.from) ?? other.nodes.find((n) => n.id === c.from)
      if (fromNode) nodesById.set(fromNode.id, { ...fromNode })
    }
    if (!nodesById.has(c.to)) {
      const toNode = fitter.nodes.find((n) => n.id === c.to) ?? other.nodes.find((n) => n.id === c.to)
      if (toNode) nodesById.set(toNode.id, { ...toNode })
    }
  }
  child.nodes = [...nodesById.values()]
  child.fitness = 0
  return child
}
