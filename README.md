# Walking Simulation

A browser-based walking creature simulation using neuroevolution (NEAT algorithm) and 2D physics.

## Features

- NEAT (NeuroEvolution of Augmenting Topologies) for evolving walking behaviors
- Real-time 2D physics simulation with Planck.js
- Interactive visualization with generation comparison
- Replay best performers across generations

## Tech Stack

- React 19 + TypeScript
- Vite
- Planck.js (Box2D physics)
- Zustand (state management)

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Controls

- **Reset** - Restart the simulation
- **Skip Generation** - Fast-forward to next generation
- **Replay Best** - Watch the all-time best performer
- **Compare Generations** - Toggle generation comparison view

