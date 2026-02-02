import * as planck from 'planck'
import type { SimConfig } from '../simulation/types'
import { MusclePair } from './muscle'

export type JointName =
  | 'neck'
  | 'shoulderL'
  | 'elbowL'
  | 'shoulderR'
  | 'elbowR'
  | 'hipL'
  | 'kneeL'
  | 'ankleL'
  | 'hipR'
  | 'kneeR'
  | 'ankleR'

export const JOINT_ORDER: JointName[] = [
  'neck',
  'shoulderL',
  'elbowL',
  'shoulderR',
  'elbowR',
  'hipL',
  'kneeL',
  'ankleL',
  'hipR',
  'kneeR',
  'ankleR',
]

const JOINT_TORQUE_SCALE: Record<JointName, number> = {
  neck: 0.08,
  shoulderL: 0.15,
  elbowL: 0.15,
  shoulderR: 0.15,
  elbowR: 0.15,
  hipL: 1.0,
  kneeL: 1.0,
  ankleL: 1.0,
  hipR: 1.0,
  kneeR: 1.0,
  ankleR: 1.0,
}

const JOINT_NEUTRAL_ANGLE: Record<JointName, number> = {
  neck: 0,
  shoulderL: 0,
  elbowL: 0.35,
  shoulderR: 0,
  elbowR: 0.35,
  hipL: 0,
  kneeL: -0.12,
  ankleL: 0.08,
  hipR: 0,
  kneeR: -0.12,
  ankleR: 0.08,
}

const JOINT_TARGET_AMPLITUDE: Record<JointName, number> = {
  neck: 0.25,
  shoulderL: 0.9,
  elbowL: 0.9,
  shoulderR: 0.9,
  elbowR: 0.9,
  hipL: 0.95,
  kneeL: 1.15,
  ankleL: 0.55,
  hipR: 0.95,
  kneeR: 1.15,
  ankleR: 0.55,
}

type JointSpec = { lower: number; upper: number }

type ContactFlags = {
  leftFoot: boolean
  rightFoot: boolean
  headGround: boolean
  torsoGround: boolean
  armGround: boolean
  legGround: boolean
}

type FixtureUserData =
  | { kind: 'ground' }
  | { kind: 'foot'; side: 'left' | 'right' }
  | { kind: 'head' }
  | { kind: 'torso' }
  | { kind: 'limb'; group: 'arm' | 'leg' }
  | undefined

export type HumanModel = {
  bodies: {
    head: planck.Body
    torso: planck.Body
    upperArmL: planck.Body
    lowerArmL: planck.Body
    upperArmR: planck.Body
    lowerArmR: planck.Body
    upperLegL: planck.Body
    lowerLegL: planck.Body
    footL: planck.Body
    upperLegR: planck.Body
    lowerLegR: planck.Body
    footR: planck.Body
  }
  joints: Record<JointName, planck.RevoluteJoint>
  jointSpecs: Record<JointName, JointSpec>
  muscles: Record<JointName, MusclePair>
  contacts: ContactFlags
  startX: number
  startComY: number
}

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))
const normalizeAngle = (a: number) => {
  let x = a
  while (x > Math.PI) x -= Math.PI * 2
  while (x < -Math.PI) x += Math.PI * 2
  return x
}

export function createHuman(world: planck.World, config: SimConfig, startX = 0): HumanModel {
  const contacts: ContactFlags = {
    leftFoot: false,
    rightFoot: false,
    headGround: false,
    torsoGround: false,
    armGround: false,
    legGround: false,
  }

  const groundCounts = {
    leftFoot: 0,
    rightFoot: 0,
    head: 0,
    torso: 0,
    arm: 0,
    leg: 0,
  }

  const syncContactBooleans = () => {
    contacts.leftFoot = groundCounts.leftFoot > 0
    contacts.rightFoot = groundCounts.rightFoot > 0
    contacts.headGround = groundCounts.head > 0
    contacts.torsoGround = groundCounts.torso > 0
    contacts.armGround = groundCounts.arm > 0
    contacts.legGround = groundCounts.leg > 0
  }

  const makeBody = (pos: planck.Vec2, opts?: Partial<planck.BodyDef>) =>
    world.createBody({
      type: 'dynamic',
      position: pos,
      linearDamping: 0.2,
      angularDamping: 0.3,
      ...opts,
    })

  const yOffset = 0.09

  const torso = makeBody(planck.Vec2(startX, 1.4 - yOffset))
  torso.createFixture(planck.Box(0.15, 0.35), {
    density: 2.0,
    friction: 0.3,
    userData: { kind: 'torso' },
  })

  const head = makeBody(planck.Vec2(startX, 1.89 - yOffset))
  head.createFixture(planck.Circle(0.12), {
    density: 1.0,
    friction: 0.2,
    userData: { kind: 'head' },
  })

  const upperArmL = makeBody(planck.Vec2(startX - 0.31, 1.57 - yOffset))
  upperArmL.createFixture(planck.Box(0.07, 0.18), {
    density: 0.8,
    friction: 0.2,
    userData: { kind: 'limb', group: 'arm' },
  })
  const lowerArmL = makeBody(planck.Vec2(startX - 0.31, 1.21 - yOffset))
  lowerArmL.createFixture(planck.Box(0.06, 0.18), {
    density: 0.7,
    friction: 0.2,
    userData: { kind: 'limb', group: 'arm' },
  })

  const upperArmR = makeBody(planck.Vec2(startX + 0.31, 1.57 - yOffset))
  upperArmR.createFixture(planck.Box(0.07, 0.18), {
    density: 0.8,
    friction: 0.2,
    userData: { kind: 'limb', group: 'arm' },
  })
  const lowerArmR = makeBody(planck.Vec2(startX + 0.31, 1.21 - yOffset))
  lowerArmR.createFixture(planck.Box(0.06, 0.18), {
    density: 0.7,
    friction: 0.2,
    userData: { kind: 'limb', group: 'arm' },
  })

  const upperLegL = makeBody(planck.Vec2(startX - 0.11, 0.83 - yOffset))
  upperLegL.createFixture(planck.Box(0.09, 0.22), {
    density: 1.2,
    friction: 0.3,
    userData: { kind: 'limb', group: 'leg' },
  })
  const lowerLegL = makeBody(planck.Vec2(startX - 0.11, 0.39 - yOffset))
  lowerLegL.createFixture(planck.Box(0.08, 0.22), {
    density: 1.0,
    friction: 0.3,
    userData: { kind: 'limb', group: 'leg' },
  })
  // Feet are forward-biased rectangles (ankle attaches near the heel, toe points +X).
  const footL = makeBody(planck.Vec2(startX - 0.11, 0.13 - yOffset))
  footL.createFixture(planck.Box(0.13, 0.04, planck.Vec2(0.08, 0), 0), {
    density: 0.8,
    // REVERTED: 1.0 was too low - model couldn't grip ground properly
    // 1.2 provides good grip while still requiring proper stepping
    friction: 1.2,
    userData: { kind: 'foot', side: 'left' },
  })

  const upperLegR = makeBody(planck.Vec2(startX + 0.11, 0.83 - yOffset))
  upperLegR.createFixture(planck.Box(0.09, 0.22), {
    density: 1.2,
    friction: 0.3,
    userData: { kind: 'limb', group: 'leg' },
  })
  const lowerLegR = makeBody(planck.Vec2(startX + 0.11, 0.39 - yOffset))
  lowerLegR.createFixture(planck.Box(0.08, 0.22), {
    density: 1.0,
    friction: 0.3,
    userData: { kind: 'limb', group: 'leg' },
  })
  const footR = makeBody(planck.Vec2(startX + 0.11, 0.13 - yOffset))
  footR.createFixture(planck.Box(0.13, 0.04, planck.Vec2(0.08, 0), 0), {
    density: 0.8,
    // REVERTED: 1.2 provides good grip
    friction: 1.2,
    userData: { kind: 'foot', side: 'right' },
  })

  const jointSpecs: Record<JointName, JointSpec> = {
    neck: { lower: -0.4, upper: 0.4 },
    shoulderL: { lower: -1.2, upper: 1.2 },
    elbowL: { lower: 0.0, upper: 1.6 },
    shoulderR: { lower: -1.2, upper: 1.2 },
    elbowR: { lower: 0.0, upper: 1.6 },
    hipL: { lower: -1.1, upper: 1.1 },
    // Note: knee flexion is clockwise in our setup, so it lives in negative joint angles.
    kneeL: { lower: -1.8, upper: 0.05 },
    ankleL: { lower: -0.35, upper: 0.7 },
    hipR: { lower: -1.1, upper: 1.1 },
    kneeR: { lower: -1.8, upper: 0.05 },
    ankleR: { lower: -0.35, upper: 0.7 },
  }

  const createJoint = (
    name: JointName,
    a: planck.Body,
    b: planck.Body,
    anchor: planck.Vec2,
  ): planck.RevoluteJoint => {
    const spec = jointSpecs[name]
    return world.createJoint(
      planck.RevoluteJoint(
        {
          collideConnected: false,
          enableLimit: true,
          lowerAngle: spec.lower,
          upperAngle: spec.upper,
          enableMotor: true,
          maxMotorTorque: config.muscles.maxTorque,
          motorSpeed: 0,
        },
        a,
        b,
        anchor,
      ),
    ) as planck.RevoluteJoint
  }

  const joints: Record<JointName, planck.RevoluteJoint> = {
    neck: createJoint('neck', torso, head, planck.Vec2(startX, 1.75 - yOffset)),
    shoulderL: createJoint('shoulderL', torso, upperArmL, planck.Vec2(startX - 0.16, 1.75 - yOffset)),
    elbowL: createJoint('elbowL', upperArmL, lowerArmL, planck.Vec2(startX - 0.31, 1.39 - yOffset)),
    shoulderR: createJoint('shoulderR', torso, upperArmR, planck.Vec2(startX + 0.16, 1.75 - yOffset)),
    elbowR: createJoint('elbowR', upperArmR, lowerArmR, planck.Vec2(startX + 0.31, 1.39 - yOffset)),
    hipL: createJoint('hipL', torso, upperLegL, planck.Vec2(startX - 0.11, 1.05 - yOffset)),
    kneeL: createJoint('kneeL', upperLegL, lowerLegL, planck.Vec2(startX - 0.11, 0.61 - yOffset)),
    ankleL: createJoint('ankleL', lowerLegL, footL, planck.Vec2(startX - 0.11, 0.17 - yOffset)),
    hipR: createJoint('hipR', torso, upperLegR, planck.Vec2(startX + 0.11, 1.05 - yOffset)),
    kneeR: createJoint('kneeR', upperLegR, lowerLegR, planck.Vec2(startX + 0.11, 0.61 - yOffset)),
    ankleR: createJoint('ankleR', lowerLegR, footR, planck.Vec2(startX + 0.11, 0.17 - yOffset)),
  }

  const bodies = {
    head,
    torso,
    upperArmL,
    lowerArmL,
    upperArmR,
    lowerArmR,
    upperLegL,
    lowerLegL,
    footL,
    upperLegR,
    lowerLegR,
    footR,
  }

  const allBodies = Object.values(bodies)
  let totalMass = 0
  let sumY = 0
  for (const b of allBodies) {
    const m = b.getMass()
    const p = b.getWorldCenter()
    totalMass += m
    sumY += p.y * m
  }
  const startComY = totalMass > 0 ? sumY / totalMass : torso.getPosition().y

  const muscles = Object.fromEntries(
    JOINT_ORDER.map((j) => [j, new MusclePair(config.muscles.energyReserve)]),
  ) as Record<JointName, MusclePair>

  world.on('begin-contact', (contact) => {
    const a = contact.getFixtureA()
    const b = contact.getFixtureB()
    const au = a.getUserData() as FixtureUserData
    const bu = b.getUserData() as FixtureUserData
    const pair = [au, bu]

    const hasGround = pair.some((u) => u?.kind === 'ground')
    if (!hasGround) return

    for (const u of pair) {
      if (u?.kind === 'foot' && u?.side === 'left') groundCounts.leftFoot++
      if (u?.kind === 'foot' && u?.side === 'right') groundCounts.rightFoot++
      if (u?.kind === 'head') groundCounts.head++
      if (u?.kind === 'torso') groundCounts.torso++
      if (u?.kind === 'limb' && u?.group === 'arm') groundCounts.arm++
      if (u?.kind === 'limb' && u?.group === 'leg') groundCounts.leg++
    }

    syncContactBooleans()
  })

  world.on('end-contact', (contact) => {
    const a = contact.getFixtureA()
    const b = contact.getFixtureB()
    const au = a.getUserData() as FixtureUserData
    const bu = b.getUserData() as FixtureUserData
    const pair = [au, bu]

    const hasGround = pair.some((u) => u?.kind === 'ground')
    if (!hasGround) return

    for (const u of pair) {
      if (u?.kind === 'foot' && u?.side === 'left') groundCounts.leftFoot = Math.max(0, groundCounts.leftFoot - 1)
      if (u?.kind === 'foot' && u?.side === 'right') groundCounts.rightFoot = Math.max(0, groundCounts.rightFoot - 1)
      if (u?.kind === 'head') groundCounts.head = Math.max(0, groundCounts.head - 1)
      if (u?.kind === 'torso') groundCounts.torso = Math.max(0, groundCounts.torso - 1)
      if (u?.kind === 'limb' && u?.group === 'arm') groundCounts.arm = Math.max(0, groundCounts.arm - 1)
      if (u?.kind === 'limb' && u?.group === 'leg') groundCounts.leg = Math.max(0, groundCounts.leg - 1)
    }

    syncContactBooleans()
  })

  return {
    bodies,
    joints,
    jointSpecs,
    muscles,
    contacts,
    startX,
    startComY,
  }
}

export function getHumanInputs(h: HumanModel, tSeconds: number): number[] {
  const inputs: number[] = []
  for (const name of JOINT_ORDER) {
    const joint = h.joints[name]
    const spec = h.jointSpecs[name]
    const angle = joint.getJointAngle()
    const speed = joint.getJointSpeed()

    const half = (spec.upper - spec.lower) / 2
    const neutral = JOINT_NEUTRAL_ANGLE[name] ?? 0
    const angleNorm = half === 0 ? 0 : clamp((angle - neutral) / half, -1, 1)
    const speedNorm = clamp(speed / 10, -1, 1)
    inputs.push(angleNorm, speedNorm)
  }

  const torsoAngleFromVertical = normalizeAngle(h.bodies.torso.getAngle())
  inputs.push(clamp(torsoAngleFromVertical / (Math.PI / 2), -1, 1))
  inputs.push(clamp(h.bodies.torso.getAngularVelocity() / 8, -1, 1))

  inputs.push(h.contacts.leftFoot ? 1 : 0)
  inputs.push(h.contacts.rightFoot ? 1 : 0)

  const { com, vel } = getHumanCOM(h)
  inputs.push(clamp((com.y - h.startComY) / 0.6, -1, 1))
  inputs.push(clamp(vel.x / 3, -1, 1))
  inputs.push(clamp(vel.y / 3, -1, 1))

  const phase = (tSeconds * Math.PI * 2) / 1.2
  inputs.push(Math.sin(phase))
  inputs.push(Math.cos(phase))

  return inputs
}

export function applyHumanActions(
  h: HumanModel,
  outputs: number[],
  dtSeconds: number,
  config: SimConfig,
): { powerW: number } {
  let powerW = 0
  for (let i = 0; i < JOINT_ORDER.length; i++) {
    const name = JOINT_ORDER[i]
    const rawOut = outputs[i] ?? 0
    const joint = h.joints[name]
    const muscle = h.muscles[name]
    const torqueScale = JOINT_TORQUE_SCALE[name] ?? 1

    const spec = h.jointSpecs[name]
    const angle = joint.getJointAngle()
    const range = spec.upper - spec.lower
    const margin = Math.max(0.06, range * 0.06)

    const neutral = JOINT_NEUTRAL_ANGLE[name] ?? 0
    const amp = JOINT_TARGET_AMPLITUDE[name] ?? (range > 0 ? range * 0.35 : 0.5)
    const target = clamp(neutral + rawOut * amp, spec.lower + margin, spec.upper - margin)
    const error = target - angle

    const kP = 10
    const kD = 2.2
    const desiredSpeed = kP * error - kD * joint.getJointSpeed()
    const maxSpeed = Math.max(0.1, config.muscles.contractionSpeed)
    const motorSpeed = clamp(desiredSpeed, -maxSpeed, maxSpeed)
    const out = motorSpeed / maxSpeed

    muscle.setOutput(out)
    muscle.step(
      dtSeconds,
      config.muscles.energyReserve,
      config.muscles.recoveryRate,
      config.muscles.maxTorque * torqueScale,
    )
    const energyRatio = muscle.energyRatioForCurrentDirection(config.muscles.energyReserve)

    const maxTorque = config.muscles.maxTorque * torqueScale * energyRatio
    joint.setMaxMotorTorque(maxTorque)
    joint.setMotorSpeed(motorSpeed)

    powerW += Math.abs(out) * maxTorque * Math.abs(joint.getJointSpeed())
  }
  return { powerW }
}

export function getHumanEnergies(h: HumanModel, energyReserve: number): number[] {
  return JOINT_ORDER.map((j) => h.muscles[j].combinedEnergyRatio(energyReserve))
}

export function getHumanActivations(h: HumanModel): number[] {
  return JOINT_ORDER.map((j) => h.muscles[j].output)
}

export function getHumanCOM(h: HumanModel): { com: planck.Vec2; vel: planck.Vec2 } {
  const bodies = Object.values(h.bodies)
  let total = 0
  let cx = 0
  let cy = 0
  let vx = 0
  let vy = 0
  for (const b of bodies) {
    const m = b.getMass()
    const p = b.getWorldCenter()
    const v = b.getLinearVelocity()
    total += m
    cx += p.x * m
    cy += p.y * m
    vx += v.x * m
    vy += v.y * m
  }
  if (total <= 0) return { com: planck.Vec2(0, 0), vel: planck.Vec2(0, 0) }
  return { com: planck.Vec2(cx / total, cy / total), vel: planck.Vec2(vx / total, vy / total) }
}

export function isHumanFallen(h: HumanModel): boolean {
  if (h.contacts.headGround || h.contacts.torsoGround || h.contacts.armGround) return true
  const headY = h.bodies.head.getPosition().y
  const torsoY = h.bodies.torso.getPosition().y
  return headY < 0.4 || torsoY < 0.4
}
