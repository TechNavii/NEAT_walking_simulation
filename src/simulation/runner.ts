import * as planck from 'planck'
import {
  applyHumanActions,
  createHuman,
  getHumanActivations,
  getHumanCOM,
  getHumanEnergies,
  getHumanInputs,
  isHumanFallen,
  JOINT_ORDER,
} from '../physics/human'
import { createWorld } from '../physics/world'
import { Genome, Network } from '../neat/genome'
import { Population } from '../neat/population'
import { useSimStore } from '../state/simStore'
import { Renderer } from '../visualization/renderer'
import { logger, type GenerationLog } from '../utils/logger'

const ARM_JOINTS = ['neck', 'shoulderL', 'elbowL', 'shoulderR', 'elbowR'] as const
const ARM_INDICES = ARM_JOINTS.map((j) => JOINT_ORDER.indexOf(j)).filter((i) => i >= 0)

export class SimulationRunner {
  private renderer: Renderer
  private rafId: number | null = null

  private population: Population | null = null
  private currentGenomeIndex = 0
  private currentNetwork: Network | null = null

  private world: planck.World | null = null
  private human: ReturnType<typeof createHuman> | null = null
  private evalTime = 0
  private energyCost = 0
  private armActuationCost = 0
  private jointSpeedCost = 0
  private jointLimitCost = 0
  private torsoTiltCost = 0
  private torsoSpinCost = 0
  private legGroundTime = 0
  private airTime = 0
  private upwardVelCost = 0
  private jumpHeightCost = 0
  private baseComY = 0

  private prevLeftFootContact = false
  private prevRightFootContact = false
  private leftSwingTime = 0
  private rightSwingTime = 0
  private lastStrikeFoot: 'left' | 'right' | null = null
  private lastStrikeTime = -1
  private stepCount = 0
  private stepLengthSum = 0
  private slipCost = 0
  private swingClearanceScore = 0
  private bestThisGen = 0
  private bestAllTime = 0
  private bestDistanceThisGenFinal = -Infinity
  private bestDistanceGenomeThisGen: Genome | null = null
  private bestDistanceEverFinal = 0
  private bestDistanceGenomeEver: Genome | null = null
  
  // Intermediate gait metrics for curriculum learning
  private singleSupportTime = 0
  private footLiftCount = 0
  private prevBothFeetDown = true
  private contactAlternations = 0
  private lastSingleSupportFoot: 'left' | 'right' | null = null
  
  // NEW: Track position at each step to verify forward progress
  private lastStepTorsoX = 0
  private effectiveStepCount = 0  // Steps that actually moved forward
  private effectiveStepLengthSum = 0
  
  // NEW: Track traction (stationary feet during contact)
  private tractionTime = 0  // Time feet are stationary while in contact

  private stepRemainder = 0
  private currentPowerW = 0

  private lastUiUpdateMs = 0
  
  // Logging state for current evaluation
  private currentEvalStats: GenerationLog['bestGenomeStats'] | null = null
  private bestEvalStatsThisGen: GenerationLog['bestGenomeStats'] | null = null
  private bestFitnessThisGen = -Infinity
  private bestGenomeThisGen: Genome | null = null
  
  // Stagnation detection
  private bestDistanceEver = 0
  private generationsWithoutImprovement = 0
  
  // Performance-based curriculum tracking
  private avgDistanceLastGen = 0
  private avgVelocityLastGen = 0
  private bestFitnessLastGen = 0
  private avgStepCoverageLastGen = 0
  
  // Best-only visualization mode
  private isReplayingBest = false  // True when replaying the best genome

  // Manual visualization modes (pause evolution)
  private mode: 'train' | 'replayBest' | 'compare' = 'train'
  private resumeGenomeIndex: number | null = null
  private replayBestDone = false

  private compareMode:
    | {
        items: Array<
          | {
              label: string
              session: {
                world: planck.World
                human: ReturnType<typeof createHuman>
                network: Network
                evalTime: number
                done: boolean
              }
            }
          | { label: string; session?: undefined }
        >
      }
    | null = null

  private readonly comparisonGenerations = [1, 10, 25, 50, 100, 200] as const
  private generationSnapshots = new Map<number, Genome>()

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new Renderer(canvas)
    this.reset()
  }

  start(): void {
    if (this.rafId != null) return
    const loop = (t: number) => {
      this.rafId = requestAnimationFrame(loop)
      this.tick(t)
    }
    this.rafId = requestAnimationFrame(loop)
  }

  stop(): void {
    if (this.rafId != null) {
      cancelAnimationFrame(this.rafId)
      this.rafId = null
    }
  }

  reset(): void {
    const { config } = useSimStore.getState()
    const inputCount = JOINT_ORDER.length * 2 + 9
    const outputCount = JOINT_ORDER.length

    this.population = new Population({
      inputCount,
      outputCount,
      config: {
        populationSize: config.neat.populationSize,
        mutationRate: config.neat.mutationRate,
        addNodeProbability: config.neat.addNodeProbability,
        addConnectionProbability: config.neat.addConnectionProbability,
        compatibilityThreshold: config.neat.compatibilityThreshold,
        elitism: config.neat.elitism,
      },
    })

    this.currentGenomeIndex = 0
    this.bestThisGen = 0
    this.bestAllTime = 0
    this.bestDistanceThisGenFinal = -Infinity
    this.bestDistanceGenomeThisGen = null
    this.bestDistanceEverFinal = 0
    this.bestDistanceGenomeEver = null
    this.stepRemainder = 0
    this.currentPowerW = 0

    this.mode = 'train'
    this.resumeGenomeIndex = null
    this.replayBestDone = false
    this.compareMode = null
    this.generationSnapshots.clear()
    
    // Reset logging state
    this.bestEvalStatsThisGen = null
    this.bestFitnessThisGen = -Infinity
    this.bestGenomeThisGen = null
    this.bestDistanceEver = 0
    this.generationsWithoutImprovement = 0
    this.avgDistanceLastGen = 0
    this.avgVelocityLastGen = 0
    this.bestFitnessLastGen = 0
    this.avgStepCoverageLastGen = 0
    logger.clear()
    console.log('%c[Walking Sim] Reset - Starting fresh with curriculum learning', 'color: #4CAF50; font-weight: bold')

    useSimStore.getState().setMetrics({
      generation: this.population.generation,
      genomeIndex: 1,
      genomeCount: config.neat.populationSize,
      bestDistanceAllTimeM: 0,
      bestDistanceThisGenM: 0,
    })

    this.startEvaluation()
  }

  skipToNextGeneration(): void {
    if (!this.population) return

    // Ensure we're in training mode when skipping.
    this.mode = 'train'
    this.compareMode = null
    this.replayBestDone = false
    this.resumeGenomeIndex = null

    // Mark remaining genomes with minimal fitness and advance.
    for (let i = this.currentGenomeIndex; i < this.population.genomes.length; i++) {
      this.population.genomes[i].fitness = -1
    }
    this.isReplayingBest = false  // Skip replay when manually skipping
    this.actuallyFinishGeneration()
  }

  toggleCompareGenerations(): void {
    if (!this.population) return

    if (this.mode === 'compare') {
      this.resumeTraining()
      return
    }

    // Pause evolution and show snapshot replays.
    this.resumeGenomeIndex = this.currentGenomeIndex
    this.mode = 'compare'
    this.replayBestDone = false
    this.isReplayingBest = false

    const { config } = useSimStore.getState()
    const items: NonNullable<typeof this.compareMode>['items'] = []

    for (const gen of this.comparisonGenerations) {
      const snap = this.generationSnapshots.get(gen)
      if (!snap) {
        items.push({ label: `Gen ${gen} (not recorded)` })
        continue
      }

      const { world } = createWorld(config)
      const human = createHuman(world, config, 0)
      const network = snap.buildNetwork()
      items.push({ label: `Gen ${gen}`, session: { world, human, network, evalTime: 0, done: false } })
    }

    this.compareMode = { items }
  }

  private resumeTraining(): void {
    this.mode = 'train'
    this.compareMode = null
    this.replayBestDone = false
    this.isReplayingBest = false

    if (this.resumeGenomeIndex != null) {
      this.currentGenomeIndex = this.resumeGenomeIndex
      this.resumeGenomeIndex = null
    }

    this.startEvaluation()
  }

  private startEvaluation(): void {
    const { config } = useSimStore.getState()
    if (!this.population) return

    const { world } = createWorld(config)
    const human = createHuman(world, config, 0)

    this.world = world
    this.human = human
    this.evalTime = 0
    this.energyCost = 0
    this.armActuationCost = 0
    this.jointSpeedCost = 0
    this.jointLimitCost = 0
    this.torsoTiltCost = 0
    this.torsoSpinCost = 0
    this.legGroundTime = 0
    this.airTime = 0
    this.upwardVelCost = 0
    this.jumpHeightCost = 0
    this.baseComY = getHumanCOM(human).com.y
    this.currentPowerW = 0

    this.prevLeftFootContact = human.contacts.leftFoot
    this.prevRightFootContact = human.contacts.rightFoot
    this.leftSwingTime = 0
    this.rightSwingTime = 0
    this.lastStrikeFoot = null
    this.lastStrikeTime = -1
    this.stepCount = 0
    this.stepLengthSum = 0
    this.slipCost = 0
    this.swingClearanceScore = 0
    
    // Reset intermediate gait metrics
    this.singleSupportTime = 0
    this.footLiftCount = 0
    this.prevBothFeetDown = true
    this.contactAlternations = 0
    this.lastSingleSupportFoot = null
    
    // Reset effective step tracking
    this.lastStepTorsoX = human.bodies.torso.getPosition().x
    this.effectiveStepCount = 0
    this.effectiveStepLengthSum = 0
    this.tractionTime = 0

    const genome = this.population.genomes[this.currentGenomeIndex]
    this.currentNetwork = genome.buildNetwork()

    useSimStore.getState().setMetrics({
      generation: this.population.generation,
      genomeIndex: this.currentGenomeIndex + 1,
      genomeCount: this.population.genomes.length,
      evalTimeSeconds: 0,
      currentDistanceM: 0,
      currentSpeedKmh: 0,
      currentPowerW: 0,
      muscleEnergies: getHumanEnergies(human, config.muscles.energyReserve),
      muscleActivations: getHumanActivations(human),
      footContacts: { left: human.contacts.leftFoot, right: human.contacts.rightFoot },
      fallen: false,
      networkGraph: {
        nodes: genome.nodes.map((n) => ({ id: n.id, type: n.type, layer: n.layer })),
        connections: genome.connections.map((c) => ({ from: c.from, to: c.to, weight: c.weight, enabled: c.enabled })),
      },
    })
  }

  private tick(nowMs: number): void {
    const state = useSimStore.getState()
    if (!state.playing) {
      if (this.mode === 'compare' && this.compareMode) {
        this.renderer.renderGrid(
          this.compareMode.items.map((it) => ({ human: it.session?.human, label: it.label })),
          state.view,
        )
      } else if (this.human) {
        this.renderer.render(this.human, state.view)
      }
      return
    }

    const config = state.config
    const showBestOnly = state.view.showBestOnly

    const baseSpeed = Math.max(0, state.speedMultiplier)
    // In "Best Only" mode during evaluation (not replay), run at max speed
    const isHiddenEvaluation = this.mode === 'train' && showBestOnly && !this.isReplayingBest
    const speed = isHiddenEvaluation ? 50 : baseSpeed
    
    this.stepRemainder += speed
    const steps = Math.floor(this.stepRemainder)
    if (steps > 0) {
      for (let i = 0; i < steps; i++) {
        if (this.mode === 'compare') {
          this.stepCompareOnce(config)
        } else if (this.mode === 'replayBest') {
          this.stepReplayBestOnce(config)
        } else {
          this.stepOnce(config)
        }

        if (this.mode !== 'compare' && (!this.world || !this.human)) break
      }
      this.stepRemainder -= steps
    }

    if (this.mode === 'compare' && this.compareMode) {
      this.renderer.renderGrid(
        this.compareMode.items.map((it) => ({ human: it.session?.human, label: it.label })),
        state.view,
      )
    } else {
      // Only render during replay or when not in best-only mode
      if (this.human && (this.mode !== 'train' || !showBestOnly || this.isReplayingBest)) {
        this.renderer.render(this.human, state.view)
      }
    }

    if (this.mode !== 'compare') {
      // Update metrics less frequently during hidden evaluation
      const uiUpdateInterval = isHiddenEvaluation ? 200 : 80
      if (nowMs - this.lastUiUpdateMs > uiUpdateInterval) {
        this.lastUiUpdateMs = nowMs
        this.pushMetrics(config)
      }
    }
  }

  private stepReplayBestOnce(config: ReturnType<typeof useSimStore.getState>['config']): void {
    if (this.replayBestDone) return
    if (!this.world || !this.human || !this.currentNetwork) return

    this.world.setGravity(planck.Vec2(0, config.physics.gravityY))

    const inputs = getHumanInputs(this.human, this.evalTime)
    const outputs = this.currentNetwork.activate(inputs)
    const { powerW } = applyHumanActions(this.human, outputs, config.evaluation.dtSeconds, config)
    this.currentPowerW = powerW

    this.world.step(config.evaluation.dtSeconds)
    this.evalTime += config.evaluation.dtSeconds

    if (isHumanFallen(this.human)) {
      this.replayBestDone = true
    }
  }

  private stepCompareOnce(config: ReturnType<typeof useSimStore.getState>['config']): void {
    if (!this.compareMode) return

    for (const it of this.compareMode.items) {
      const s = it.session
      if (!s || s.done) continue

      s.world.setGravity(planck.Vec2(0, config.physics.gravityY))
      const inputs = getHumanInputs(s.human, s.evalTime)
      const outputs = s.network.activate(inputs)
      applyHumanActions(s.human, outputs, config.evaluation.dtSeconds, config)
      s.world.step(config.evaluation.dtSeconds)
      s.evalTime += config.evaluation.dtSeconds

      if (isHumanFallen(s.human)) {
        s.done = true
      }
    }
  }

  private stepOnce(config: ReturnType<typeof useSimStore.getState>['config']): void {
    if (!this.population || !this.world || !this.human || !this.currentNetwork) return

    this.world.setGravity(planck.Vec2(0, config.physics.gravityY))

    const inputs = getHumanInputs(this.human, this.evalTime)
    const outputs = this.currentNetwork.activate(inputs)

    this.energyCost += outputs.reduce((s, v) => s + v * v, 0) * config.evaluation.dtSeconds
    this.armActuationCost +=
      ARM_INDICES.reduce((s, idx) => {
        const v = outputs[idx] ?? 0
        return s + v * v
      }, 0) * config.evaluation.dtSeconds
    const { powerW } = applyHumanActions(this.human, outputs, config.evaluation.dtSeconds, config)
    this.currentPowerW = powerW

    this.world.step(config.evaluation.dtSeconds)
    this.evalTime += config.evaluation.dtSeconds

    const { com, vel } = getHumanCOM(this.human)
    const up = Math.max(0, vel.y)
    this.upwardVelCost += up * up * config.evaluation.dtSeconds
    this.jumpHeightCost += Math.max(0, com.y - this.baseComY - 0.15) * config.evaluation.dtSeconds
    if (!this.human.contacts.leftFoot && !this.human.contacts.rightFoot) {
      this.airTime += config.evaluation.dtSeconds
    }

    this.stepGaitShaping(com.y, config.evaluation.dtSeconds)

    if (this.human.contacts.legGround) {
      this.legGroundTime += config.evaluation.dtSeconds
    }

    const torsoAngleFromVertical = normalizeAngle(this.human.bodies.torso.getAngle())
    this.torsoTiltCost += Math.abs(torsoAngleFromVertical) * config.evaluation.dtSeconds
    this.torsoSpinCost += Math.abs(this.human.bodies.torso.getAngularVelocity()) * config.evaluation.dtSeconds

    for (const name of JOINT_ORDER) {
      const joint = this.human.joints[name]
      this.jointSpeedCost += Math.abs(joint.getJointSpeed()) * config.evaluation.dtSeconds

      const spec = this.human.jointSpecs[name]
      const angle = joint.getJointAngle()
      const range = spec.upper - spec.lower
      if (range > 0) {
        const distToEdge = Math.min(angle - spec.lower, spec.upper - angle)
        const edge = range * 0.12
        if (edge > 0) {
          const prox = Math.max(0, edge - distToEdge) / edge
          this.jointLimitCost += prox * prox * config.evaluation.dtSeconds
        }
      }
    }

    const torsoX = this.human.bodies.torso.getPosition().x
    const distance = torsoX - this.human.startX
    this.bestThisGen = Math.max(this.bestThisGen, distance)
    this.bestAllTime = Math.max(this.bestAllTime, distance)

    const fallen = isHumanFallen(this.human)
    const timeLimitReached = this.mode === 'train' && this.evalTime >= config.evaluation.maxSeconds

    if (fallen || timeLimitReached) {
      // Safety: if stepOnce ever gets called outside training mode, do not
      // advance evolution; just stop stepping.
      if (this.mode !== 'train') {
        this.replayBestDone = true
        return
      }

      const genome = this.population.genomes[this.currentGenomeIndex]
      const hiddenNodeCount = genome.nodes.filter(n => n.type === 'hidden').length

      const avgVelocity = this.evalTime > 0 ? distance / this.evalTime : 0
      const slipPerMeter = distance > 0 ? this.slipCost / distance : this.slipCost
      const distClamped = Math.max(0, distance)
      const cappedStepLen = Math.min(Math.max(0, this.effectiveStepLengthSum), distClamped)
      const stepCoverage = distClamped > 0 ? cappedStepLen / distClamped : 0
      
      const fitness = computeFitness({
        distance,
        // Raw step metrics (for analysis)
        stepCount: this.stepCount,
        stepLengthSum: this.stepLengthSum,
        // CRITICAL: Effective steps that actually moved body forward
        effectiveStepCount: this.effectiveStepCount,
        effectiveStepLengthSum: this.effectiveStepLengthSum,
        slipCost: this.slipCost,
        swingClearanceScore: this.swingClearanceScore,
        survived: this.evalTime,
        maxSurvivalTime: config.evaluation.maxSeconds,
        fallen,
        energyCost: this.energyCost,
        armActuationCost: this.armActuationCost,
        jointSpeedCost: this.jointSpeedCost,
        jointLimitCost: this.jointLimitCost,
        torsoTiltCost: this.torsoTiltCost,
        torsoSpinCost: this.torsoSpinCost,
        legGroundTime: this.legGroundTime,
        airTime: this.airTime,
        upwardVelCost: this.upwardVelCost,
        jumpHeightCost: this.jumpHeightCost,
        // Curriculum learning metrics - NOW PERFORMANCE-BASED
        generation: this.population.generation,
        singleSupportTime: this.singleSupportTime,
        footLiftCount: this.footLiftCount,
        contactAlternations: this.contactAlternations,
        // Network complexity
        hiddenNodeCount,
        // Performance-based curriculum inputs
        avgDistanceLastGen: this.avgDistanceLastGen,
        avgVelocityLastGen: this.avgVelocityLastGen,
        bestFitnessLastGen: this.bestFitnessLastGen,
        avgStepCoverageLastGen: this.avgStepCoverageLastGen,
        // NEW: Traction time for anti-skating reward
        tractionTime: this.tractionTime,
      })
      this.population.genomes[this.currentGenomeIndex].fitness = fitness

      if (distance > this.bestDistanceThisGenFinal) {
        this.bestDistanceThisGenFinal = distance
        this.bestDistanceGenomeThisGen = genome
      }
      
      // Track evaluation stats for logging
      this.currentEvalStats = {
        distance,
        avgVelocity,
        stepCount: this.stepCount,
        stepLengthSum: this.stepLengthSum,
        effectiveStepCount: this.effectiveStepCount,
        effectiveStepLengthSum: this.effectiveStepLengthSum,
        stepCoverage,
        singleSupportTime: this.singleSupportTime,
        footLiftCount: this.footLiftCount,
        contactAlternations: this.contactAlternations,
        survivalTime: this.evalTime,
        energyCost: this.energyCost,
        slipCost: this.slipCost,
        slipPerMeter,
        fallen,
      }
      
      // Track best this generation
      if (fitness > this.bestFitnessThisGen) {
        this.bestFitnessThisGen = fitness
        this.bestEvalStatsThisGen = { ...this.currentEvalStats }
        this.bestGenomeThisGen = this.population.genomes[this.currentGenomeIndex]
      }
      
      // Log individual evaluation
      logger.logEvaluation({
        genomeIndex: this.currentGenomeIndex,
        fitness,
        distance,
        avgVelocity,
        stepCount: this.stepCount,
        stepLengthSum: this.stepLengthSum,
        stepCoverage,
        singleSupportTime: this.singleSupportTime,
        footLiftCount: this.footLiftCount,
        contactAlternations: this.contactAlternations,
        survivalTime: this.evalTime,
        energyCost: this.energyCost,
        slipCost: this.slipCost,
        slipPerMeter,
        fallen,
        hiddenNodes: genome.nodes.filter(n => n.type === 'hidden').length,
        enabledConnections: genome.connections.filter(c => c.enabled).length,
      })
      
      this.advanceGenomeOrGeneration()
      return
    }
  }

  private stepGaitShaping(comY: number, dtSeconds: number): void {
    const human = this.human
    if (!human) return

    const left = human.contacts.leftFoot
    const right = human.contacts.rightFoot

    const leftSwingTime = this.leftSwingTime
    const rightSwingTime = this.rightSwingTime

    // Track single-support time (exactly one foot on ground) - intermediate reward
    if ((left && !right) || (!left && right)) {
      this.singleSupportTime += dtSeconds
    }

    // Track foot lifts - count when going from both-feet-down to one-foot-up
    const bothDown = left && right
    if (this.prevBothFeetDown && !bothDown && (left || right)) {
      this.footLiftCount += 1
    }
    this.prevBothFeetDown = bothDown

    // Track alternation between LEFT-only and RIGHT-only single-support.
    // (Normal gait goes left-only -> both -> right-only -> both -> ...)
    const singleSupportFoot: 'left' | 'right' | null =
      left && !right ? 'left' : right && !left ? 'right' : null
    if (singleSupportFoot && this.lastSingleSupportFoot && singleSupportFoot !== this.lastSingleSupportFoot) {
      this.contactAlternations += 1
    }
    if (singleSupportFoot) this.lastSingleSupportFoot = singleSupportFoot

    // Slip penalty: discourage skating while in contact.
    const vLx = human.bodies.footL.getLinearVelocity().x
    const vRx = human.bodies.footR.getLinearVelocity().x
    if (left) this.slipCost += Math.abs(vLx) * dtSeconds
    if (right) this.slipCost += Math.abs(vRx) * dtSeconds
    
    // TRACTION REWARD: Track time feet are stationary during contact
    // A foot is "planted" if velocity < 0.1 m/s while in contact
    const plantedThreshold = 0.1
    const leftPlanted = left && Math.abs(vLx) < plantedThreshold
    const rightPlanted = right && Math.abs(vRx) < plantedThreshold
    if (leftPlanted || rightPlanted) {
      this.tractionTime += dtSeconds
    }

    // Swing clearance reward: only in single-support and not during high jumps.
    const clearanceMin = 0.04  // Relaxed from 0.06
    const clearanceMax = 0.18
    const comOk = comY - this.baseComY < 0.25
    if (comOk && left !== right) {
      const swingFoot = left ? human.bodies.footR : human.bodies.footL
      const y = swingFoot.getPosition().y
      this.swingClearanceScore += Math.max(0, Math.min(clearanceMax, y) - clearanceMin) * dtSeconds
    }

    // Heel-strike step reward (alternation + stride length).
    // CRITICAL FIX: Only count steps that actually move the body forward
    const now = this.evalTime
    const minSwing = 0.08
    const minStrideTime = 0.14
    const currentTorsoX = human.bodies.torso.getPosition().x
    
    const tryStrike = (foot: 'left' | 'right') => {
      const isLeft = foot === 'left'
      const contact = isLeft ? left : right
      const wasContact = isLeft ? this.prevLeftFootContact : this.prevRightFootContact
      const swingTime = isLeft ? leftSwingTime : rightSwingTime
      const otherContact = isLeft ? right : left
      if (!contact || wasContact) return
      if (swingTime < minSwing) return
      if (this.lastStrikeTime >= 0 && now - this.lastStrikeTime < minStrideTime) return

      const strikeX = (isLeft ? human.bodies.footL : human.bodies.footR).getPosition().x
      const stanceX = (isLeft ? human.bodies.footR : human.bodies.footL).getPosition().x
      
      // CRITICAL: Check if body actually moved forward since last step
      const forwardProgressSinceLastStep = currentTorsoX - this.lastStepTorsoX
      const minForwardProgress = 0.05  // Must move at least 5cm forward per step
      const isEffectiveStep = forwardProgressSinceLastStep >= minForwardProgress

      // Step length estimation:
      // - If the other foot is in contact, use foot-to-foot separation (classic walking).
      // - If not (e.g. flight phase / running), fall back to torso progress so we don't
      //   misclassify fast gaits as "shuffling" due to under-counted steps.
      const rawStepLen = otherContact ? strikeX - stanceX : forwardProgressSinceLastStep
      const stepLen = Math.max(0, Math.min(0.6, rawStepLen - 0.02))

      const alternates = this.lastStrikeFoot == null || this.lastStrikeFoot !== foot
      if (alternates) {
        // Always count raw steps (for analysis)
        this.stepCount += 1
        this.stepLengthSum += stepLen
        
        // Only count effective steps that moved body forward
        if (isEffectiveStep) {
          this.effectiveStepCount += 1
          this.effectiveStepLengthSum += Math.min(stepLen, forwardProgressSinceLastStep)
          this.lastStepTorsoX = currentTorsoX  // Update reference position
        }
      }
      this.lastStrikeFoot = foot
      this.lastStrikeTime = now
    }

    tryStrike('left')
    tryStrike('right')

    // Swing timers for next step.
    this.leftSwingTime = left ? 0 : leftSwingTime + dtSeconds
    this.rightSwingTime = right ? 0 : rightSwingTime + dtSeconds

    this.prevLeftFootContact = left
    this.prevRightFootContact = right
  }

  private advanceGenomeOrGeneration(): void {
    if (!this.population) return

    // If we just finished replaying the best, move to actual finish
    if (this.isReplayingBest) {
      this.isReplayingBest = false
      this.actuallyFinishGeneration()
      return
    }

    this.currentGenomeIndex++
    if (this.currentGenomeIndex >= this.population.genomes.length) {
      // All genomes evaluated - check if we need to replay best
      const { view } = useSimStore.getState()
      if (view.showBestOnly && this.bestGenomeThisGen) {
        this.startBestReplay()
      } else {
        this.actuallyFinishGeneration()
      }
      return
    }
    this.startEvaluation()
  }
  
  private startBestReplay(): void {
    if (!this.population || !this.bestGenomeThisGen) return
    
    this.isReplayingBest = true
    
    // Find index of best genome
    const bestIndex = this.population.genomes.indexOf(this.bestGenomeThisGen)
    this.currentGenomeIndex = bestIndex >= 0 ? bestIndex : 0
    
    console.log(`%c[Best Only] Replaying best genome (index ${this.currentGenomeIndex + 1}, fitness ${this.bestFitnessThisGen.toFixed(1)})`, 'color: #4CAF50')
    
    // Re-run the evaluation for visualization
    this.startEvaluationForReplay(this.bestGenomeThisGen)
  }
  
  // Public method to replay the all-time best genome
  replayAllTimeBest(): void {
    if (!this.population || !this.bestDistanceGenomeEver) {
      console.log('%c[Replay] No all-time best genome recorded yet', 'color: #ff9800')
      return
    }

    // Toggle off if already replaying.
    if (this.mode === 'replayBest') {
      this.resumeTraining()
      return
    }
    
    // Pause evolution and replay without the time limit (run until fall).
    this.resumeGenomeIndex = this.currentGenomeIndex
    this.mode = 'replayBest'
    this.compareMode = null
    this.replayBestDone = false
    this.isReplayingBest = true
    
    console.log(
      `%c[Replay] Playing all-time best (no time limit): ${this.bestDistanceEverFinal.toFixed(2)}m`,
      'color: #4CAF50; font-weight: bold',
    )
    
    // Start replay with the all-time best genome
    this.startEvaluationForReplay(this.bestDistanceGenomeEver)
  }
  
  // Check if we have an all-time best to replay
  hasAllTimeBest(): boolean {
    return this.bestDistanceGenomeEver !== null && this.bestDistanceEverFinal > 0
  }
  
  // Get the all-time best distance
  getAllTimeBestDistance(): number {
    return this.bestDistanceEverFinal
  }
  
  private startEvaluationForReplay(genome: Genome): void {
    const { config } = useSimStore.getState()
    if (!this.population) return

    const { world } = createWorld(config)
    const human = createHuman(world, config, 0)

    this.world = world
    this.human = human
    this.evalTime = 0
    this.energyCost = 0
    this.armActuationCost = 0
    this.jointSpeedCost = 0
    this.jointLimitCost = 0
    this.torsoTiltCost = 0
    this.torsoSpinCost = 0
    this.legGroundTime = 0
    this.airTime = 0
    this.upwardVelCost = 0
    this.jumpHeightCost = 0
    this.baseComY = getHumanCOM(human).com.y
    this.currentPowerW = 0

    this.prevLeftFootContact = human.contacts.leftFoot
    this.prevRightFootContact = human.contacts.rightFoot
    this.leftSwingTime = 0
    this.rightSwingTime = 0
    this.lastStrikeFoot = null
    this.lastStrikeTime = -1
    this.stepCount = 0
    this.stepLengthSum = 0
    this.slipCost = 0
    this.swingClearanceScore = 0
    
    this.singleSupportTime = 0
    this.footLiftCount = 0
    this.prevBothFeetDown = true
    this.contactAlternations = 0
    this.lastSingleSupportFoot = null
    
    this.lastStepTorsoX = human.bodies.torso.getPosition().x
    this.effectiveStepCount = 0
    this.effectiveStepLengthSum = 0
    this.tractionTime = 0

    this.currentNetwork = genome.buildNetwork()

    useSimStore.getState().setMetrics({
      generation: this.population.generation,
      genomeIndex: this.currentGenomeIndex + 1,
      genomeCount: this.population.genomes.length,
      evalTimeSeconds: 0,
      currentDistanceM: 0,
      currentSpeedKmh: 0,
      currentPowerW: 0,
      muscleEnergies: getHumanEnergies(human, config.muscles.energyReserve),
      muscleActivations: getHumanActivations(human),
      footContacts: { left: human.contacts.leftFoot, right: human.contacts.rightFoot },
      fallen: false,
      networkGraph: {
        nodes: genome.nodes.map((n) => ({ id: n.id, type: n.type, layer: n.layer })),
        connections: genome.connections.map((c) => ({ from: c.from, to: c.to, weight: c.weight, enabled: c.enabled })),
      },
    })
  }

  private actuallyFinishGeneration(): void {
    if (!this.population) return
    const { config } = useSimStore.getState()
    
    // Log generation stats before evolving
    const allFitnesses = this.population.genomes.map(g => g.fitness)
    const bestGenome = this.bestGenomeThisGen ?? this.population.genomes[0]
    const bestStats = this.bestEvalStatsThisGen ?? {
      distance: 0,
      avgVelocity: 0,
      stepCount: 0,
      stepLengthSum: 0,
      effectiveStepCount: 0,
      effectiveStepLengthSum: 0,
      stepCoverage: 0,
      singleSupportTime: 0,
      footLiftCount: 0,
      contactAlternations: 0,
      survivalTime: 0,
      energyCost: 0,
      slipCost: 0,
      slipPerMeter: 0,
      fallen: true,
    }
    
    // UPDATE PERFORMANCE-BASED CURRICULUM METRICS
    // These are used by computeFitness to determine curriculum phases
    this.avgDistanceLastGen = bestStats.distance
    this.avgVelocityLastGen = bestStats.avgVelocity
    this.bestFitnessLastGen = Math.max(...allFitnesses, 0)
    this.avgStepCoverageLastGen = bestStats.stepCoverage
    
    // Stagnation detection
    // NOTE: fitness changes over time due to curriculum (stepPhase/penaltyPhase),
    // so use best distance (stable across generations) as the improvement signal.
    const currentBestDist = this.bestThisGen
    // Small improvements matter at the top-end (e.g. 6.80m -> 6.86m).
    const improvementThresholdM = 0.05
    if (currentBestDist > this.bestDistanceEver + improvementThresholdM) {
      this.bestDistanceEver = currentBestDist
      this.generationsWithoutImprovement = 0
    } else {
      this.generationsWithoutImprovement++
    }

    if (this.bestDistanceThisGenFinal > this.bestDistanceEverFinal && this.bestDistanceGenomeThisGen) {
      this.bestDistanceEverFinal = this.bestDistanceThisGenFinal
      this.bestDistanceGenomeEver = this.bestDistanceGenomeThisGen.clone()
    }
    
    // Log stagnation status
    if (this.generationsWithoutImprovement > 0) {
      console.log(
        `%c[Stagnation] No distance improvement for ${this.generationsWithoutImprovement} generations (best=${this.bestDistanceEver.toFixed(2)}m)`,
        this.generationsWithoutImprovement > 5 ? 'color: #f44336' : 'color: #ff9800'
      )
    }
    
    logger.logGenerationEnd(
      this.population.generation,
      this.population.species.length,
      bestStats,
      {
        hiddenNodes: bestGenome.nodes.filter(n => n.type === 'hidden').length,
        totalConnections: bestGenome.connections.length,
        enabledConnections: bestGenome.connections.filter(c => c.enabled).length,
      },
      allFitnesses
    )

    // Snapshot champions for the comparison grid.
    if ((this.comparisonGenerations as readonly number[]).includes(this.population.generation)) {
      this.generationSnapshots.set(this.population.generation, bestGenome.clone())
    }

    // BALANCED STAGNATION LEVELS (give population time to converge):
    // 0: Normal (< 5 gens) - let evolution work naturally
    // 1: Mild (5-10 gens) - slight mutation boost
    // 2: Moderate (10-20 gens) - fitness sharing + diversity injection
    // 3: Severe (20-40 gens) - aggressive exploration
    // 4: Critical (40-70 gens) - massive diversity injection
    // 5: NUCLEAR (70+ gens) - keep only top 5, regenerate rest
    // IMPORTANT: Nuclear reset should be a one-shot intervention, not a permanent mode.
    const doNuclearReset = this.generationsWithoutImprovement >= 70
    const stagnationLevel = doNuclearReset ? 5 :
                           this.generationsWithoutImprovement >= 40 ? 4 :
                           this.generationsWithoutImprovement >= 20 ? 3 :
                           this.generationsWithoutImprovement >= 10 ? 2 :
                           this.generationsWithoutImprovement >= 5 ? 1 : 0

    if (stagnationLevel > 0) {
      const colors = ['#ff9800', '#ff9800', '#f44336', '#f44336', '#9c27b0', '#e91e63']
      const labels = ['Mild', 'Mild', 'Moderate', 'Severe', 'Critical', 'NUCLEAR']
      console.log(`%c[Stagnation] Level ${stagnationLevel} (${labels[stagnationLevel]}): ${this.generationsWithoutImprovement} generations without improvement`, 
        `color: ${colors[stagnationLevel]}; font-weight: bold`)
    }

    // Inject diversity starting at level 1 (earlier than before!)
    const injectDiversity = stagnationLevel >= 1

    this.population.evolve({
      populationSize: config.neat.populationSize,
      // IMPORTANT: Do NOT globally boost weight mutation rates under stagnation.
      // For gait control, high mutation destroys incremental improvements.
      // Stagnation handling is implemented inside Population.evolve via explicit
      // local-search + exploration lanes.
      mutationRate: config.neat.mutationRate,
      addNodeProbability: config.neat.addNodeProbability,
      addConnectionProbability: config.neat.addConnectionProbability,
      compatibilityThreshold: config.neat.compatibilityThreshold,
      elitism: config.neat.elitism,
      injectDiversity,
      stagnationLevel,
      hallOfFameGenome: this.bestDistanceGenomeEver ?? undefined,
    })

    if (doNuclearReset) {
      // Give the reset population a fresh window before escalating interventions again.
      this.generationsWithoutImprovement = 0
    }

    this.currentGenomeIndex = 0
    this.bestThisGen = 0
    this.bestDistanceThisGenFinal = -Infinity
    this.bestDistanceGenomeThisGen = null
    
    // Reset generation logging state
    this.bestFitnessThisGen = -Infinity
    this.bestEvalStatsThisGen = null
    this.bestGenomeThisGen = null

    useSimStore.getState().setMetrics({
      generation: this.population.generation,
      genomeIndex: 1,
      genomeCount: this.population.genomes.length,
      bestDistanceThisGenM: 0,
      bestDistanceAllTimeM: this.bestAllTime,
    })

    this.startEvaluation()
  }

  private pushMetrics(config: ReturnType<typeof useSimStore.getState>['config']): void {
    if (!this.human || !this.population) return

    const torsoX = this.human.bodies.torso.getPosition().x
    const distance = torsoX - this.human.startX
    const comVelX = this.human.bodies.torso.getLinearVelocity().x
    const speedKmh = comVelX * 3.6

    useSimStore.getState().setMetrics({
      generation: this.population.generation,
      genomeIndex: this.currentGenomeIndex + 1,
      genomeCount: this.population.genomes.length,
      evalTimeSeconds: this.evalTime,
      currentDistanceM: distance,
      currentSpeedKmh: speedKmh,
      currentPowerW: this.currentPowerW,
      bestDistanceThisGenM: this.bestThisGen,
      bestDistanceAllTimeM: this.bestAllTime,
      muscleEnergies: getHumanEnergies(this.human, config.muscles.energyReserve),
      muscleActivations: getHumanActivations(this.human),
      footContacts: { left: this.human.contacts.leftFoot, right: this.human.contacts.rightFoot },
      fallen: isHumanFallen(this.human),
    })
  }
}

function computeFitness(args: {
  distance: number
  stepCount: number
  stepLengthSum: number
  effectiveStepCount: number
  effectiveStepLengthSum: number
  slipCost: number
  swingClearanceScore: number
  survived: number
  maxSurvivalTime: number
  fallen: boolean
  energyCost: number
  armActuationCost: number
  jointSpeedCost: number
  jointLimitCost: number
  torsoTiltCost: number
  torsoSpinCost: number
  legGroundTime: number
  airTime: number
  upwardVelCost: number
  jumpHeightCost: number
  generation: number
  singleSupportTime: number
  footLiftCount: number
  contactAlternations: number
  hiddenNodeCount: number
  // NEW: Performance-based curriculum inputs
  avgDistanceLastGen: number
  avgVelocityLastGen: number
  bestFitnessLastGen: number
  avgStepCoverageLastGen: number
  // NEW: Traction time for anti-skating
  tractionTime: number
}): number {
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
  
  // =============================================================
  // RADICALLY SIMPLIFIED FITNESS FUNCTION
  // 
  // LESSON LEARNED: Complex fitness functions with competing objectives
  // cause the model to optimize for the wrong thing. The previous version
  // regressed from 5.63m to 4.04m because "good steps over 4m" scored
  // better than "bad steps over 5.6m".
  // 
  // NEW PHILOSOPHY: DISTANCE IS KING
  // 1. Distance is 85% of fitness - no gates, no penalties reducing it
  // 2. Survival is 10% - stay upright longer = more distance potential
  // 3. Velocity bonus 5% - reward speed without penalizing slow progress
  // 4. NO penalties that compete with distance (no shuffling penalty!)
  // 5. Only penalize: falling, going backwards
  // =============================================================
  
  const rawDist = args.distance
  const dist = Math.max(0, rawDist)
  const survival = args.survived
  const maxSurvival = args.maxSurvivalTime
  const didFall = args.fallen
  const completedFullTime = !didFall && survival >= maxSurvival - 0.1
  const avgVelocity = survival > 0 ? rawDist / survival : 0
  const survivalFrac = maxSurvival > 0 ? clamp01(survival / maxSurvival) : 0
  
  // =============================================================
  // STEP COVERAGE (computed first - used to scale distance!)
  // This is the ratio of distance covered by actual steps vs total distance
  // A shuffler might have 15% step coverage, a proper walker 50%+
  // =============================================================
  const distClamped = Math.max(0, rawDist)
  const cappedStepLen = Math.min(Math.max(0, args.effectiveStepLengthSum), distClamped)
  const stepCoverage = distClamped > 0 ? cappedStepLen / distClamped : 0
  
  // =============================================================
  // PRIMARY: DISTANCE - NOW SCALED BY STEP COVERAGE!
  // Shuffling gives you LESS reward per meter than proper stepping
  // This is the KEY FIX to break the shuffling local optimum
  // =============================================================
  
  // =============================================================
  // GAIT-GATED DISTANCE REWARD
  // 
  // KEY INSIGHT: Shuffling is physically easier than stepping, so shufflers
  // will ALWAYS cover more distance. The only way to incentivize proper
  // walking is to HARD GATE the distance reward based on step coverage.
  //
  // Below 35% step coverage: You're shuffling. Distance reward is CAPPED.
  // Above 35% step coverage: Full distance reward scales with coverage.
  // =============================================================
  
  const STEP_COVERAGE_THRESHOLD = 0.35  // Below this = shuffling
  
  let gaitQualityMultiplier: number
  let effectiveDistance: number
  
  if (stepCoverage < STEP_COVERAGE_THRESHOLD) {
    // SHUFFLING: Hard cap on how much distance credit you get
    // Even if you shuffle 7m, you only get credit for ~3m worth
    // Plus a penalty multiplier based on how bad your coverage is
    const cappedDist = Math.min(dist, 3.0)  // Cap at 3m for shufflers
    const shufflePenalty = stepCoverage / STEP_COVERAGE_THRESHOLD  // 0.27/0.35 = 0.77
    effectiveDistance = cappedDist * shufflePenalty
    gaitQualityMultiplier = 0.5  // Additional 50% penalty
  } else {
    // PROPER WALKING: Full distance credit, scaled by coverage quality
    effectiveDistance = dist
    // Bonus multiplier for good step coverage (1.0 at 35%, up to 1.3 at 70%+)
    gaitQualityMultiplier = 1.0 + Math.min(0.3, (stepCoverage - STEP_COVERAGE_THRESHOLD) * 0.86)
  }
  
  // Base: 300 points per EFFECTIVE meter
  // Shuffler at 6.80m, 27% coverage: effectiveDist=2.31m * 300 * 0.5 = 347 points
  // Walker at 4.00m, 50% coverage: effectiveDist=4m * 300 * 1.13 = 1356 points
  // NOW proper walking clearly dominates!
  const distanceReward = effectiveDistance * 300 * gaitQualityMultiplier * (0.6 + 0.4 * survivalFrac)
  
  // Distance milestones - REQUIRE PROPER WALKING (35%+ step coverage)
  // Shufflers get NO distance milestones - they're capped at 3m effective anyway
  // This creates a huge incentive to cross the 35% step coverage threshold
  const distVelocityMult = 1.0 + Math.max(0, avgVelocity - 0.2) * 1.5
  let distanceMilestoneBonus = 0
  if (stepCoverage >= STEP_COVERAGE_THRESHOLD) {  // MUST be proper walking!
    if (dist >= 4) distanceMilestoneBonus += 100 * distVelocityMult
    if (dist >= 5) distanceMilestoneBonus += 100 * distVelocityMult
    if (dist >= 6) distanceMilestoneBonus += 150 * distVelocityMult
    if (dist >= 7) distanceMilestoneBonus += 150 * distVelocityMult
    if (dist >= 8) distanceMilestoneBonus += 200 * distVelocityMult
    if (dist >= 10) distanceMilestoneBonus += 250 * distVelocityMult
    if (dist >= 12) distanceMilestoneBonus += 300 * distVelocityMult
  }
  
  // =============================================================
  // SECONDARY: SURVIVAL - NOW MORE IMPORTANT!
  // The model keeps falling at 12s. We need to make survival valuable.
  // =============================================================
  
  const survivalBonus = survival * 15  // Increased from 12
  
  // MASSIVE completion bonus - this is key!
  // Current local optimum: 6.92m in 12.1s, falls, fitness ~2388
  // To make NOT falling attractive, completion bonus must be significant
  // If you survive 15s at 0.46 m/s (same velocity), you'd get 6.92m + this bonus
  // Bonus of 400 makes stable walking competitive with fast-then-fall
  const completionBonus = completedFullTime ? 400 : 0
  
  // =============================================================
  // TERTIARY: VELOCITY - MASSIVE PUSH FOR FASTER WALKING
  // Model stuck at 0.2 m/s with good step coverage. Need to reward SPEED!
  // =============================================================
  
  // Quadratic velocity bonus - DRAMATICALLY INCREASED for proper walkers
  // For shufflers (< 35% coverage): minimal bonus
  // For proper walkers (>= 35% coverage): huge bonus to walk faster
  // 
  // At 0.2 m/s, proper walker: 0.2^2 * 1500 * 1.0 = 60
  // At 0.3 m/s, proper walker: 0.3^2 * 1500 * 1.0 = 135
  // At 0.4 m/s, proper walker: 0.4^2 * 1500 * 1.0 = 240
  // At 0.5 m/s, proper walker: 0.5^2 * 1500 * 1.0 = 375
  const velBonusMultiplier = stepCoverage >= STEP_COVERAGE_THRESHOLD ? 1500 : 200
  const velocityBonus = avgVelocity > 0 ? avgVelocity * avgVelocity * velBonusMultiplier * survivalFrac : 0
  
  // Velocity milestones - REQUIRE MINIMUM STEP COVERAGE!
  // MUCH LARGER bonuses to incentivize faster walking
  // Must have 35%+ step coverage (the shuffling threshold) to earn velocity bonuses
  let velocityMilestoneBonus = 0
  if (completedFullTime && stepCoverage >= STEP_COVERAGE_THRESHOLD) {
    if (avgVelocity >= 0.15) velocityMilestoneBonus += 100   // Even slow proper walking gets rewarded
    if (avgVelocity >= 0.25) velocityMilestoneBonus += 200   // Increased from 100
    if (avgVelocity >= 0.35) velocityMilestoneBonus += 300   // Increased from 150
    if (avgVelocity >= 0.45) velocityMilestoneBonus += 400   // Big jump for faster walking
    if (avgVelocity >= 0.55) velocityMilestoneBonus += 500   
    if (avgVelocity >= 0.70) velocityMilestoneBonus += 600   // Really fast walking
  }
  
  // =============================================================
  // GAIT QUALITY BONUS - ADDITIONAL REWARDS FOR PROPER STEPPING
  // Note: Step coverage is now ALSO baked into distance reward above
  // These are extra incentives for good gait
  // =============================================================
  
  const effectiveSteps = args.effectiveStepCount
  const singleSupport = args.singleSupportTime
  
  // MASSIVE step coverage bonus - scales with BOTH coverage AND distance
  // This creates a strong incentive: "walk far WITH good steps"
  // 
  // Formula: stepCoverage * distance * 80
  // Shuffler at 6.80m, 27% coverage: 0.27 * 6.80 * 80 = 147 points
  // Walker at 4.00m, 50% coverage: 0.50 * 4.00 * 80 = 160 points (beats shuffler!)
  // Walker at 5.00m, 50% coverage: 0.50 * 5.00 * 80 = 200 points
  // Walker at 5.00m, 70% coverage: 0.70 * 5.00 * 80 = 280 points
  //
  // This creates a MULTIPLICATIVE incentive: every meter you walk properly
  // gives you more bonus than a meter shuffled
  const stepCoverageBonus = completedFullTime 
    ? stepCoverage * dist * 80  // Scales with distance!
    : stepCoverage * dist * 30  // Partial bonus if fell
  
  // Bonus for taking more steps (encourages leg movement)
  // Current shuffler: 4 steps = 20 points
  // Proper walker: 14 steps = 70 points
  const stepBonus = effectiveSteps * 5
  
  // Bonus for single-support time (one foot off ground = actually stepping)
  // Shufflers have LOW single-support time, proper walkers have HIGH
  const singleSupportBonus = singleSupport * 4  // Increased from 3
  
  // =============================================================
  // PENALTIES: THE KEY FIX FOR LOCAL OPTIMA
  // =============================================================
  
  // CRITICAL INSIGHT: The model is stuck at 6.92m in 12.1s (then FALLS)
  // Fitness: ~2388. Any mutation that improves stability results in
  // slower initial movement -> less distance -> LOWER FITNESS
  // So the model NEVER learns to stay upright!
  //
  // THE FIX: Make falling after going far VERY expensive
  // This creates pressure to learn stability WHILE maintaining distance
  
  // FALLING PENALTY scales with how much potential was wasted
  // If you fall early (2s), small penalty - you didn't have momentum
  // If you fall LATE (12s after going 7m), BIG penalty - you wasted potential!
  // Formula: penalty = (distance * survivalFrac * 80)
  // At 6.92m, 12.1s/15s = 0.807 survival: penalty = 6.92 * 0.807 * 80 = 447
  let fallingPenalty = 0
  if (didFall) {
    // Base penalty
    fallingPenalty = 50
    // Additional penalty for "wasted potential" - you were doing well then fell!
    // The further you got and longer you survived, the more you "lost"
    const wastedPotential = dist * survivalFrac * 80
    fallingPenalty += wastedPotential
  }
  
  // Going backwards is bad
  const backwardsPenalty = rawDist < 0 ? Math.abs(rawDist) * 150 : 0
  
  // Slow movement penalty (only for non-falling runs to not double-penalize)
  const minVelocityTarget = 0.3
  let slowMovementPenalty = 0
  if (completedFullTime && avgVelocity < minVelocityTarget) {
    slowMovementPenalty = (minVelocityTarget - avgVelocity) * 800
  }
  
  // Standing completely still is even worse
  const standingStillPenalty = completedFullTime && avgVelocity < 0.05
    ? (0.05 - avgVelocity) * 500
    : 0
  
  // =============================================================
  // TOTAL: Simple sum, distance dominates
  // =============================================================
  
  const total =
    // PRIMARY: Distance (should be ~70% of score)
    distanceReward +
    distanceMilestoneBonus +
    // SECONDARY: Survival (~10%)
    survivalBonus +
    completionBonus +
    // TERTIARY: Velocity (~10%)
    velocityBonus +
    velocityMilestoneBonus +
    // GAIT QUALITY (~10%) - KEY to break shuffling habit
    stepCoverageBonus +
    stepBonus +
    singleSupportBonus -
    // Penalties
    fallingPenalty -
    backwardsPenalty -
    slowMovementPenalty -
    standingStillPenalty

  // Minimum fitness floor
  const minFitness = 1 + survival * 2
  return Math.max(minFitness, total)
}

function normalizeAngle(a: number): number {
  let x = a
  while (x > Math.PI) x -= Math.PI * 2
  while (x < -Math.PI) x += Math.PI * 2
  return x
}
