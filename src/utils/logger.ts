export type GenerationLog = {
  generation: number
  timestamp: number
  
  // Population stats
  populationSize: number
  speciesCount: number
  
  // Fitness stats
  bestFitness: number
  avgFitness: number
  worstFitness: number
  
  // Best genome breakdown
  bestGenomeStats: {
    distance: number
    avgVelocity: number
    stepCount: number
    stepLengthSum: number
    effectiveStepCount: number
    effectiveStepLengthSum: number
    stepCoverage: number
    singleSupportTime: number
    footLiftCount: number
    contactAlternations: number
    survivalTime: number
    energyCost: number
    slipCost: number
    slipPerMeter: number
    fallen: boolean
  }
  
  // Network complexity
  bestNetworkComplexity: {
    hiddenNodes: number
    totalConnections: number
    enabledConnections: number
  }
  
  // Curriculum phase info
  curriculumPhase: {
    stepPhase: number
    intermediatePhase: number
    penaltyPhase: number
  }
}

export type EvaluationLog = {
  genomeIndex: number
  fitness: number
  distance: number
  avgVelocity: number
  stepCount: number
  stepLengthSum: number
  stepCoverage: number
  singleSupportTime: number
  footLiftCount: number
  contactAlternations: number
  survivalTime: number
  energyCost: number
  slipCost: number
  slipPerMeter: number
  fallen: boolean
  hiddenNodes: number
  enabledConnections: number
}

class SimulationLogger {
  private generationLogs: GenerationLog[] = []
  private currentGenEvaluations: EvaluationLog[] = []
  private enabled = true
  private consoleLogging = true
  
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
  }
  
  setConsoleLogging(enabled: boolean): void {
    this.consoleLogging = enabled
  }
  
  logEvaluation(log: EvaluationLog): void {
    if (!this.enabled) return
    this.currentGenEvaluations.push(log)
  }
  
  logGenerationEnd(
    generation: number,
    speciesCount: number,
    bestGenomeStats: GenerationLog['bestGenomeStats'],
    bestNetworkComplexity: GenerationLog['bestNetworkComplexity'],
    allFitnesses: number[]
  ): void {
    if (!this.enabled) return
    
    const sorted = [...allFitnesses].sort((a, b) => b - a)
    const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
    const bestFitness = sorted[0] ?? 0
    
    // PERFORMANCE-BASED CURRICULUM (matches computeFitness logic)
    const lastDist = bestGenomeStats.distance
    const lastStepCoverage = bestGenomeStats.stepCoverage
    const isDistanceStage = lastDist < 4
    const isGaitStage = !isDistanceStage && lastStepCoverage < 0.35
    const stepPhase = isDistanceStage ? 0 : isGaitStage ? 0.6 : 1
    const penaltyPhase = isDistanceStage ? 0 : clamp01((lastDist - 4) / 6)
    const intermediatePhase = isDistanceStage ? 1 : isGaitStage ? 0.7 : 0.4
    
    const log: GenerationLog = {
      generation,
      timestamp: Date.now(),
      populationSize: allFitnesses.length,
      speciesCount,
      bestFitness,
      avgFitness: allFitnesses.reduce((a, b) => a + b, 0) / allFitnesses.length || 0,
      worstFitness: sorted[sorted.length - 1] ?? 0,
      bestGenomeStats,
      bestNetworkComplexity,
      curriculumPhase: {
        stepPhase,
        intermediatePhase,
        penaltyPhase,
      },
    }
    
    this.generationLogs.push(log)
    this.currentGenEvaluations = []
    
    if (this.consoleLogging) {
      this.printGenerationSummary(log)
    }
  }
  
  private printGenerationSummary(log: GenerationLog): void {
    const phase = log.curriculumPhase
    const stats = log.bestGenomeStats
    const net = log.bestNetworkComplexity
    
    console.group(`%c[Gen ${log.generation}] Summary`, 'color: #4CAF50; font-weight: bold')
    
    console.log(
      `%cFitness: best=${log.bestFitness.toFixed(1)}, avg=${log.avgFitness.toFixed(1)}, worst=${log.worstFitness.toFixed(1)}`,
      'color: #2196F3'
    )
    
    console.log(
      `%cProgress: dist=${stats.distance.toFixed(2)}m, vel=${stats.avgVelocity.toFixed(2)}m/s, survived=${stats.survivalTime.toFixed(1)}s${stats.fallen ? ' (FELL)' : ''}`,
      'color: #FF9800'
    )
    
    const shuffleRatio = stats.stepCount > 0 ? (stats.effectiveStepCount / stats.stepCount * 100).toFixed(0) : '100'
    console.log(
      `%cSteps: raw=${stats.stepCount}, effective=${stats.effectiveStepCount} (${shuffleRatio}% effective), stepLen=${stats.stepLengthSum.toFixed(2)}m→${stats.effectiveStepLengthSum.toFixed(2)}m, coverage=${(stats.stepCoverage * 100).toFixed(0)}%`,
      stats.effectiveStepCount < stats.stepCount * 0.5 ? 'color: #F44336' : 'color: #4CAF50'
    )

    console.log(
      `%cSlip: ${stats.slipCost.toFixed(2)} (per m=${stats.slipPerMeter.toFixed(2)})`,
      stats.slipPerMeter > 1.2 ? 'color: #F44336' : 'color: #607D8B'
    )
    
    console.log(
      `%cGait: singleSupport=${stats.singleSupportTime.toFixed(2)}s, footLifts=${stats.footLiftCount}, alternations=${stats.contactAlternations}`,
      'color: #9C27B0'
    )
    
    console.log(
      `%cNetwork: ${net.hiddenNodes} hidden, ${net.enabledConnections}/${net.totalConnections} connections`,
      'color: #607D8B'
    )
    
    console.log(
      `%cCurriculum: stepPhase=${(phase.stepPhase * 100).toFixed(0)}%, intermediatePhase=${(phase.intermediatePhase * 100).toFixed(0)}%, penaltyPhase=${(phase.penaltyPhase * 100).toFixed(0)}%`,
      'color: #795548'
    )
    
    console.log(`%cSpecies: ${log.speciesCount}`, 'color: #009688')
    
    console.groupEnd()
  }
  
  getGenerationLogs(): GenerationLog[] {
    return [...this.generationLogs]
  }
  
  exportToJSON(): string {
    return JSON.stringify({
      exportedAt: new Date().toISOString(),
      totalGenerations: this.generationLogs.length,
      logs: this.generationLogs,
    }, null, 2)
  }
  
  downloadLogs(): void {
    const data = this.exportToJSON()
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `walking-sim-logs-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  clear(): void {
    this.generationLogs = []
    this.currentGenEvaluations = []
  }
  
  getLatestStats(): { generation: number; bestFitness: number; avgFitness: number } | null {
    if (this.generationLogs.length === 0) return null
    const latest = this.generationLogs[this.generationLogs.length - 1]
    return {
      generation: latest.generation,
      bestFitness: latest.bestFitness,
      avgFitness: latest.avgFitness,
    }
  }
}

export const logger = new SimulationLogger()

// Expose to window for debugging
if (typeof window !== 'undefined') {
  (window as unknown as { simLogger: SimulationLogger }).simLogger = logger
}
