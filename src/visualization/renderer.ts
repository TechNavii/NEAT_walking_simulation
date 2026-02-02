import type { HumanModel } from '../physics/human'
import type { ViewOptions } from '../simulation/types'

type Viewport = {
  scale: number
  cameraX: number
  originY: number
}

type RenderRect = {
  x: number
  y: number
  width: number
  height: number
}

type ShapeCircle = { getType?: () => 'circle'; m_p?: { x: number; y: number }; m_radius?: number }
type ShapePolygon = {
  getType?: () => 'polygon'
  m_vertices?: Array<{ x: number; y: number }>
  m_count?: number
}
type Shape = ShapeCircle | ShapePolygon

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))

export class Renderer {
  private ctx: CanvasRenderingContext2D
  private canvas: HTMLCanvasElement
  private scale = 180
  private cameraX = 0

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas 2D context not available')
    this.canvas = canvas
    this.ctx = ctx
  }

  resizeToDisplaySize(): void {
    const dpr = Math.max(1, window.devicePixelRatio || 1)
    const { clientWidth, clientHeight } = this.canvas
    const w = Math.floor(clientWidth * dpr)
    const h = Math.floor(clientHeight * dpr)
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.ctx.scale(dpr, dpr)
  }

  render(human: HumanModel, view: ViewOptions): void {
    const ctx = this.ctx
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight

    this.resizeToDisplaySize()

    const torsoX = human.bodies.torso.getPosition().x
    if (view.cameraMode === 'follow') {
      this.cameraX = torsoX
    } else {
      this.cameraX = 0
    }

    const viewport: Viewport = {
      scale: this.scale,
      cameraX: this.cameraX,
      originY: h * 0.85,
    }

    // Background
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, '#bfe8ff')
    grad.addColorStop(1, '#eaf7ff')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, w, h)

    // Ground
    ctx.fillStyle = '#4f7d3b'
    ctx.fillRect(0, viewport.originY, w, h - viewport.originY)
    ctx.fillStyle = '#6ea650'
    ctx.fillRect(0, viewport.originY - 8, w, 8)

    this.drawDistanceMarkers(viewport)
    this.drawHuman(human, viewport)
  }

  renderGrid(items: Array<{ human?: HumanModel; label: string }>, view: ViewOptions): void {
    const ctx = this.ctx
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight

    this.resizeToDisplaySize()

    const cols = 3
    const rows = Math.max(1, Math.ceil(items.length / cols))
    const cellW = w / cols
    const cellH = h / rows

    // Clear once; each panel draws its own background.
    ctx.clearRect(0, 0, w, h)

    for (let i = 0; i < items.length; i++) {
      const col = i % cols
      const row = Math.floor(i / cols)
      const rect: RenderRect = {
        x: col * cellW,
        y: row * cellH,
        width: cellW,
        height: cellH,
      }

      const item = items[i]
      if (item.human) this.renderToRect(item.human, view, rect, item.label)
      else this.renderEmptyRect(rect, item.label)

      // Panel border
      ctx.save()
      ctx.strokeStyle = 'rgba(0,0,0,0.15)'
      ctx.lineWidth = 1
      ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1)
      ctx.restore()
    }
  }

  private renderEmptyRect(rect: RenderRect, label: string): void {
    const ctx = this.ctx

    ctx.save()
    ctx.beginPath()
    ctx.rect(rect.x, rect.y, rect.width, rect.height)
    ctx.clip()

    const grad = ctx.createLinearGradient(0, rect.y, 0, rect.y + rect.height)
    grad.addColorStop(0, '#bfe8ff')
    grad.addColorStop(1, '#eaf7ff')
    ctx.fillStyle = grad
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height)

    ctx.fillStyle = 'rgba(0,0,0,0.65)'
    ctx.font = '14px system-ui'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, rect.x + rect.width / 2, rect.y + rect.height / 2)
    ctx.restore()
  }

  private renderToRect(human: HumanModel, view: ViewOptions, rect: RenderRect, label?: string): void {
    const ctx = this.ctx

    ctx.save()
    ctx.beginPath()
    ctx.rect(rect.x, rect.y, rect.width, rect.height)
    ctx.clip()

    const torsoX = human.bodies.torso.getPosition().x
    const cameraX = view.cameraMode === 'follow' ? torsoX : 0

    const viewport: Viewport = {
      scale: this.scale,
      cameraX,
      originY: rect.height * 0.85,
    }

    // Background
    const grad = ctx.createLinearGradient(0, rect.y, 0, rect.y + rect.height)
    grad.addColorStop(0, '#bfe8ff')
    grad.addColorStop(1, '#eaf7ff')
    ctx.fillStyle = grad
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height)

    // Ground
    const groundY = rect.y + viewport.originY
    ctx.fillStyle = '#4f7d3b'
    ctx.fillRect(rect.x, groundY, rect.width, rect.height - viewport.originY)
    ctx.fillStyle = '#6ea650'
    ctx.fillRect(rect.x, groundY - 8, rect.width, 8)

    this.drawDistanceMarkersRect(viewport, rect)
    this.drawHumanRect(human, viewport, rect)
    if (label) this.drawNameTag(human, viewport, rect, label)

    ctx.restore()
  }

  private drawDistanceMarkers(vp: Viewport): void {
    const ctx = this.ctx
    const w = this.canvas.clientWidth
    const groundY = vp.originY

    const minX = vp.cameraX + (-w / 2) / vp.scale
    const maxX = vp.cameraX + (w / 2) / vp.scale

    const start = Math.floor(minX)
    const end = Math.ceil(maxX)

    ctx.strokeStyle = 'rgba(255,255,255,0.7)'
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.lineWidth = 1
    ctx.font = '12px system-ui'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'

    for (let x = start; x <= end; x++) {
      const sx = this.worldToScreenX(x, vp)
      const is5 = x % 5 === 0
      const is1 = x % 1 === 0
      if (!is1) continue
      const len = is5 ? 24 : 12
      ctx.beginPath()
      ctx.moveTo(sx, groundY)
      ctx.lineTo(sx, groundY + len)
      ctx.stroke()
      if (is5) {
        ctx.fillText(`${x} m`, sx, groundY + len + 4)
      }
    }
  }

  private drawHuman(h: HumanModel, vp: Viewport): void {
    const ctx = this.ctx
    const bodies = Object.entries(h.bodies)

    for (const [name, body] of bodies) {
      const pos = body.getPosition()
      const angle = body.getAngle()

      ctx.save()
      ctx.translate(this.worldToScreenX(pos.x, vp), this.worldToScreenY(pos.y, vp))
      ctx.rotate(-angle)

      const fill =
        name === 'torso'
          ? '#4b5563'
          : name === 'head'
            ? '#6b7280'
            : name.startsWith('foot')
              ? '#374151'
              : '#9ca3af'
      ctx.fillStyle = fill
      ctx.strokeStyle = 'rgba(0,0,0,0.25)'
      ctx.lineWidth = 1

      for (let fixture = body.getFixtureList(); fixture; fixture = fixture.getNext()) {
        const shape = fixture.getShape() as unknown as Shape
        const type = shape.getType?.()
        if (type === 'circle') {
          const p = (shape as ShapeCircle).m_p ?? { x: 0, y: 0 }
          const r = (shape as ShapeCircle).m_radius ?? 0
          ctx.beginPath()
          ctx.arc(p.x * vp.scale, -p.y * vp.scale, r * vp.scale, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        } else if (type === 'polygon') {
          const verts = (shape as ShapePolygon).m_vertices ?? []
          const count: number = (shape as ShapePolygon).m_count ?? verts.length
          if (count >= 2) {
            ctx.beginPath()
            ctx.moveTo(verts[0].x * vp.scale, -verts[0].y * vp.scale)
            for (let i = 1; i < count; i++) {
              ctx.lineTo(verts[i].x * vp.scale, -verts[i].y * vp.scale)
            }
            ctx.closePath()
            ctx.fill()
            ctx.stroke()
          }
        }
      }

      ctx.restore()
    }
  }

  private worldToScreenX(x: number, vp: Viewport): number {
    const w = this.canvas.clientWidth
    return (x - vp.cameraX) * vp.scale + w / 2
  }

  private worldToScreenY(y: number, vp: Viewport): number {
    const h = this.canvas.clientHeight
    const originY = clamp(vp.originY, 0, h)
    return originY - y * vp.scale
  }

  private worldToScreenXRect(x: number, vp: Viewport, rect: RenderRect): number {
    return rect.x + (x - vp.cameraX) * vp.scale + rect.width / 2
  }

  private worldToScreenYRect(y: number, vp: Viewport, rect: RenderRect): number {
    const originY = clamp(vp.originY, 0, rect.height)
    return rect.y + originY - y * vp.scale
  }

  private drawDistanceMarkersRect(vp: Viewport, rect: RenderRect): void {
    const ctx = this.ctx

    const groundY = rect.y + vp.originY
    const minX = vp.cameraX + (-rect.width / 2) / vp.scale
    const maxX = vp.cameraX + (rect.width / 2) / vp.scale

    const start = Math.floor(minX)
    const end = Math.ceil(maxX)

    ctx.strokeStyle = 'rgba(255,255,255,0.7)'
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.lineWidth = 1
    ctx.font = '12px system-ui'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'

    for (let x = start; x <= end; x++) {
      const sx = this.worldToScreenXRect(x, vp, rect)
      const is5 = x % 5 === 0
      const len = is5 ? 18 : 10
      ctx.beginPath()
      ctx.moveTo(sx, groundY)
      ctx.lineTo(sx, groundY + len)
      ctx.stroke()
      if (is5) {
        ctx.fillText(`${x} m`, sx, groundY + len + 2)
      }
    }
  }

  private drawHumanRect(h: HumanModel, vp: Viewport, rect: RenderRect): void {
    const ctx = this.ctx
    const bodies = Object.entries(h.bodies)

    for (const [name, body] of bodies) {
      const pos = body.getPosition()
      const angle = body.getAngle()

      ctx.save()
      ctx.translate(this.worldToScreenXRect(pos.x, vp, rect), this.worldToScreenYRect(pos.y, vp, rect))
      ctx.rotate(-angle)

      const fill =
        name === 'torso'
          ? '#4b5563'
          : name === 'head'
            ? '#6b7280'
            : name.startsWith('foot')
              ? '#374151'
              : '#9ca3af'
      ctx.fillStyle = fill
      ctx.strokeStyle = 'rgba(0,0,0,0.25)'
      ctx.lineWidth = 1

      for (let fixture = body.getFixtureList(); fixture; fixture = fixture.getNext()) {
        const shape = fixture.getShape() as unknown as Shape
        const type = shape.getType?.()
        if (type === 'circle') {
          const p = (shape as ShapeCircle).m_p ?? { x: 0, y: 0 }
          const r = (shape as ShapeCircle).m_radius ?? 0
          ctx.beginPath()
          ctx.arc(p.x * vp.scale, -p.y * vp.scale, r * vp.scale, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        } else if (type === 'polygon') {
          const verts = (shape as ShapePolygon).m_vertices ?? []
          const count: number = (shape as ShapePolygon).m_count ?? verts.length
          if (count >= 2) {
            ctx.beginPath()
            ctx.moveTo(verts[0].x * vp.scale, -verts[0].y * vp.scale)
            for (let i = 1; i < count; i++) {
              ctx.lineTo(verts[i].x * vp.scale, -verts[i].y * vp.scale)
            }
            ctx.closePath()
            ctx.fill()
            ctx.stroke()
          }
        }
      }

      ctx.restore()
    }
  }

  private drawNameTag(h: HumanModel, vp: Viewport, rect: RenderRect, label: string): void {
    const ctx = this.ctx
    const head = h.bodies.head.getPosition()
    const x = this.worldToScreenXRect(head.x, vp, rect)
    const y = this.worldToScreenYRect(head.y, vp, rect) - 18

    ctx.save()
    ctx.font = '12px system-ui'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    const padX = 6
    const textW = ctx.measureText(label).width
    const boxW = textW + padX * 2
    const boxH = 16

    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'
    ctx.lineWidth = 1
    ctx.beginPath()
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x - boxW / 2, y - boxH / 2, boxW, boxH, 6)
    } else {
      ctx.rect(x - boxW / 2, y - boxH / 2, boxW, boxH)
    }
    ctx.fill()
    ctx.stroke()

    ctx.fillStyle = 'rgba(255,255,255,0.95)'
    ctx.fillText(label, x, y + 0.5)
    ctx.restore()
  }
}
