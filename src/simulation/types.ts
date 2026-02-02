export type CameraMode = 'follow' | 'fixed'

export type SimConfig = {
  physics: {
    gravityY: number
    groundFriction: number
  }
  muscles: {
    maxTorque: number
    contractionSpeed: number
    energyReserve: number
    recoveryRate: number
  }
  evaluation: {
    dtSeconds: number
    maxSeconds: number
  }
  neat: {
    populationSize: number
    mutationRate: number
    addNodeProbability: number
    addConnectionProbability: number
    compatibilityThreshold: number
    elitism: number
  }
}

export type NetworkGraph = {
  nodes: Array<{ id: number; type: 'input' | 'bias' | 'hidden' | 'output'; layer: number }>
  connections: Array<{ from: number; to: number; weight: number; enabled: boolean }>
}

export type SimMetrics = {
  generation: number
  genomeIndex: number
  genomeCount: number
  evalTimeSeconds: number
  currentDistanceM: number
  currentSpeedKmh: number
  currentPowerW: number
  bestDistanceThisGenM: number
  bestDistanceAllTimeM: number
  muscleEnergies: number[]
  muscleActivations: number[]
  networkGraph?: NetworkGraph
  footContacts: { left: boolean; right: boolean }
  fallen: boolean
}

export type ViewOptions = {
  showNetwork: boolean
  showMuscleActivation: boolean
  showPowerGraph: boolean
  cameraMode: CameraMode
  showBestOnly: boolean  // When true, only visualize the best genome per generation
}
