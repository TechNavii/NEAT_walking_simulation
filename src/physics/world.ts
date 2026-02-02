import * as planck from 'planck'
import type { SimConfig } from '../simulation/types'

export type WorldBundle = {
  world: planck.World
  ground: planck.Body
}

export function createWorld(config: SimConfig): WorldBundle {
  const world = new planck.World(planck.Vec2(0, config.physics.gravityY))

  const ground = world.createBody({ type: 'static', position: planck.Vec2(0, 0) })
  const groundShape = planck.Box(200, 0.1, planck.Vec2(0, -0.1), 0)
  ground.createFixture(groundShape, { friction: config.physics.groundFriction, userData: { kind: 'ground' } })

  return { world, ground }
}
