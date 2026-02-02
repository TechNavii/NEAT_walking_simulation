import { create } from 'zustand'
import type { SimConfig, SimMetrics, ViewOptions } from '../simulation/types'

const defaultConfig: SimConfig = {
  physics: {
    gravityY: -9.81,
    groundFriction: 1.0,  // Balanced friction
  },
  muscles: {
    maxTorque: 120,
    contractionSpeed: 8,
    // Reduced energy to make fatigue meaningful (~4s continuous full activation)
    // Recovery allows sustained 50% duty cycle at moderate activation
    energyReserve: 400,
    recoveryRate: 35,
  },
  evaluation: {
    dtSeconds: 1 / 60,
    // INCREASED: Model learned stable walking (0.44 m/s, survives 15s)
    // But 15s caps distance at ~6.6m. Increase to 25s for ~11m potential
    // The falling penalty prevents lazy standing - model will try to go faster
    maxSeconds: 25,
  },
  neat: {
    // Population size for evolution
    // Increased to improve exploration / reduce premature convergence
    populationSize: 150,
    // Moderate mutation rate for weight changes
    mutationRate: 0.4,
    // MODERATE structural mutation - previous 0.30/0.40 caused network bloat
    // Networks grew from 5 to 16 hidden nodes without improving
    addNodeProbability: 0.12,
    addConnectionProbability: 0.20,
    // Compatibility threshold for speciation
    compatibilityThreshold: 1.5,
    // Keep 3 elites to preserve good solutions
    elitism: 3,
  },
}

const defaultMetrics: SimMetrics = {
  generation: 1,
  genomeIndex: 1,
  genomeCount: defaultConfig.neat.populationSize,
  evalTimeSeconds: 0,
  currentDistanceM: 0,
  currentSpeedKmh: 0,
  currentPowerW: 0,
  bestDistanceThisGenM: 0,
  bestDistanceAllTimeM: 0,
  muscleEnergies: [],
  muscleActivations: [],
  footContacts: { left: false, right: false },
  fallen: false,
}

const defaultView: ViewOptions = {
  showNetwork: true,
  showMuscleActivation: true,
  showPowerGraph: true,
  cameraMode: 'follow',
  showBestOnly: false,  // When true, only visualize the best genome per generation
}

type SimStore = {
  config: SimConfig
  setConfig: (patch: Partial<SimConfig>) => void

  playing: boolean
  setPlaying: (v: boolean) => void

  speedMultiplier: number
  setSpeedMultiplier: (v: number) => void

  view: ViewOptions
  setView: (patch: Partial<ViewOptions>) => void

  metrics: SimMetrics
  setMetrics: (patch: Partial<SimMetrics>) => void
}

export const useSimStore = create<SimStore>((set) => ({
  config: defaultConfig,
  setConfig: (patch) =>
    set((s) => ({
      config: {
        ...s.config,
        ...patch,
        physics: { ...s.config.physics, ...patch.physics },
        muscles: { ...s.config.muscles, ...patch.muscles },
        evaluation: { ...s.config.evaluation, ...patch.evaluation },
        neat: { ...s.config.neat, ...patch.neat },
      },
    })),

  playing: true,
  setPlaying: (v) => set({ playing: v }),

  speedMultiplier: 1,
  setSpeedMultiplier: (v) => set({ speedMultiplier: v }),

  view: defaultView,
  setView: (patch) => set((s) => ({ view: { ...s.view, ...patch } })),

  metrics: defaultMetrics,
  setMetrics: (patch) => set((s) => ({ metrics: { ...s.metrics, ...patch } })),
}))

export const getDefaultConfig = () => structuredClone(defaultConfig)
