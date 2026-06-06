import type {
  CellRenderState,
  ControlsConfig,
  Counts,
  Debris,
  PopulationPoint,
  Strain,
} from './types'

type Cell = {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  biomass: number
  strain: Strain
  phase: number
  divisionProgress: number
  deathProgress: number
  dying: boolean
}

type Particle = {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  life: number
}

const GRID_SIZE = 92
const MAX_CELLS = 1200
const DISH_RADIUS = 320
const CELL_BASE_RADIUS = 4.3
const BASE_MASS = 1
const DIVISION_MASS = 2.05
const DIVISION_DURATION = 0.7
const NUTRIENT_KS = 0.6
const MAX_GROWTH_RATE = 0.95
const NUTRIENT_UPTAKE_RATE = 0.65
const MOTILITY_JITTER = 16
const DRAG = 0.91
const CHEMOTAXIS = 0.014
const NUTRIENT_DIFFUSION = 0.28
const ANTIBIOTIC_DIFFUSION = 0.22
const NUTRIENT_DECAY = 0.01
const ANTIBIOTIC_DECAY = 0.045
const ANTIBIOTIC_HILL = 1.7
const WILD_MIC = 0.4
const MUTANT_MIC = 2.1
const KILL_MAX = 2.7
const INITIAL_WILD = 140
const INITIAL_MUTANT = 10
const HISTORY_SECONDS = 140

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

class DiffusionField {
  private readonly data = new Float32Array(GRID_SIZE * GRID_SIZE)
  private readonly scratch = new Float32Array(GRID_SIZE * GRID_SIZE)
  private readonly invCell = GRID_SIZE / (DISH_RADIUS * 2)

  addGaussian(x: number, y: number, amount: number, sigma = 28): void {
    const sigmaSq = sigma * sigma
    const span = Math.max(2, Math.ceil((sigma * 2.8 * GRID_SIZE) / (DISH_RADIUS * 2)))
    const [cx, cy] = this.toGrid(x, y)

    for (let dy = -span; dy <= span; dy += 1) {
      const gy = cy + dy
      if (gy < 0 || gy >= GRID_SIZE) {
        continue
      }
      for (let dx = -span; dx <= span; dx += 1) {
        const gx = cx + dx
        if (gx < 0 || gx >= GRID_SIZE) {
          continue
        }
        const worldX = ((gx + 0.5) / GRID_SIZE) * (DISH_RADIUS * 2) - DISH_RADIUS
        const worldY = ((gy + 0.5) / GRID_SIZE) * (DISH_RADIUS * 2) - DISH_RADIUS
        if (worldX * worldX + worldY * worldY > DISH_RADIUS * DISH_RADIUS) {
          continue
        }
        const distSq = (worldX - x) * (worldX - x) + (worldY - y) * (worldY - y)
        const idx = gy * GRID_SIZE + gx
        this.data[idx] += amount * Math.exp(-distSq / (2 * sigmaSq))
      }
    }
  }

  sample(x: number, y: number): number {
    const nx = clamp((x + DISH_RADIUS) * this.invCell, 0, GRID_SIZE - 1.0001)
    const ny = clamp((y + DISH_RADIUS) * this.invCell, 0, GRID_SIZE - 1.0001)
    const x0 = Math.floor(nx)
    const y0 = Math.floor(ny)
    const x1 = Math.min(x0 + 1, GRID_SIZE - 1)
    const y1 = Math.min(y0 + 1, GRID_SIZE - 1)
    const tx = nx - x0
    const ty = ny - y0

    const i00 = y0 * GRID_SIZE + x0
    const i10 = y0 * GRID_SIZE + x1
    const i01 = y1 * GRID_SIZE + x0
    const i11 = y1 * GRID_SIZE + x1

    const a = this.data[i00] * (1 - tx) + this.data[i10] * tx
    const b = this.data[i01] * (1 - tx) + this.data[i11] * tx
    return a * (1 - ty) + b * ty
  }

  gradient(x: number, y: number): [number, number] {
    const step = 5
    const gx = this.sample(x + step, y) - this.sample(x - step, y)
    const gy = this.sample(x, y + step) - this.sample(x, y - step)
    return [gx, gy]
  }

  consume(x: number, y: number, amount: number): number {
    const [gx, gy] = this.toGrid(x, y)
    const idx = gy * GRID_SIZE + gx
    const available = this.data[idx]
    const consumed = Math.min(available, amount)
    this.data[idx] -= consumed
    return consumed
  }

  diffuse(diffusionRate: number, decayRate: number, dt: number): void {
    const d = diffusionRate * dt
    const decay = Math.max(0, 1 - decayRate * dt)

    for (let y = 0; y < GRID_SIZE; y += 1) {
      for (let x = 0; x < GRID_SIZE; x += 1) {
        const idx = y * GRID_SIZE + x
        const worldX = ((x + 0.5) / GRID_SIZE) * (DISH_RADIUS * 2) - DISH_RADIUS
        const worldY = ((y + 0.5) / GRID_SIZE) * (DISH_RADIUS * 2) - DISH_RADIUS
        if (worldX * worldX + worldY * worldY > DISH_RADIUS * DISH_RADIUS) {
          this.scratch[idx] = 0
          continue
        }

        const center = this.data[idx]
        const left = x > 0 ? this.data[idx - 1] : center
        const right = x < GRID_SIZE - 1 ? this.data[idx + 1] : center
        const up = y > 0 ? this.data[idx - GRID_SIZE] : center
        const down = y < GRID_SIZE - 1 ? this.data[idx + GRID_SIZE] : center
        const laplace = left + right + up + down - center * 4
        const next = Math.max(0, center + laplace * d)
        this.scratch[idx] = next * decay
      }
    }

    this.data.set(this.scratch)
  }

  private toGrid(x: number, y: number): [number, number] {
    const gx = Math.floor(clamp((x + DISH_RADIUS) * this.invCell, 0, GRID_SIZE - 1))
    const gy = Math.floor(clamp((y + DISH_RADIUS) * this.invCell, 0, GRID_SIZE - 1))
    return [gx, gy]
  }
}

export class SimulationEngine {
  readonly dishRadius = DISH_RADIUS
  private readonly nutrient = new DiffusionField()
  private readonly antibiotic = new DiffusionField()
  private readonly cells: Cell[] = []
  private readonly particles: Particle[] = []
  private readonly history: PopulationPoint[] = []
  private now = 0
  private cellId = 0
  private particleId = 0
  private nutrientSpawnAccumulator = 0
  private historyAccumulator = 0

  private controls: ControlsConfig = {
    nutrientSpawnRate: 1.4,
    mutationProbability: 0.018,
  }

  constructor() {
    this.seedInitialPopulation()
    for (let i = 0; i < 9; i += 1) {
      this.nutrient.addGaussian(
        (Math.random() - 0.5) * DISH_RADIUS * 0.8,
        (Math.random() - 0.5) * DISH_RADIUS * 0.8,
        4 + Math.random() * 2,
        40,
      )
    }
    this.pushHistoryPoint()
  }

  setControls(next: ControlsConfig): void {
    this.controls = next
  }

  addNutrients(x: number, y: number): void {
    this.nutrient.addGaussian(x, y, 12, 32)
  }

  injectAntibiotic(x: number, y: number): void {
    this.antibiotic.addGaussian(x, y, 8.5, 30)
  }

  step(dt: number): void {
    const frameDt = Math.min(dt, 0.033)
    this.now += frameDt

    this.spawnNutrients(frameDt)

    const newborns: Cell[] = []
    const survivors: Cell[] = []

    for (const cell of this.cells) {
      if (!cell.dying) {
        this.updateLivingCell(cell, frameDt, newborns)
      } else {
        cell.deathProgress -= frameDt / 1.2
      }

      if (cell.deathProgress > 0) {
        survivors.push(cell)
      }
    }

    this.cells.length = 0
    this.cells.push(...survivors)
    if (this.cells.length < MAX_CELLS) {
      this.cells.push(...newborns.slice(0, Math.max(0, MAX_CELLS - this.cells.length)))
    }

    this.nutrient.diffuse(NUTRIENT_DIFFUSION, NUTRIENT_DECAY, frameDt)
    this.antibiotic.diffuse(ANTIBIOTIC_DIFFUSION, ANTIBIOTIC_DECAY, frameDt)
    this.updateParticles(frameDt)

    this.historyAccumulator += frameDt
    if (this.historyAccumulator >= 0.35) {
      this.historyAccumulator = 0
      this.pushHistoryPoint()
    }
  }

  getCounts(): Counts {
    let wild = 0
    let mutant = 0
    for (const cell of this.cells) {
      if (cell.dying) {
        continue
      }
      if (cell.strain === 'wild') {
        wild += 1
      } else {
        mutant += 1
      }
    }
    return { wild, mutant }
  }

  getHistory(): PopulationPoint[] {
    return this.history
  }

  getCellRenderStates(): CellRenderState[] {
    return this.cells.map((cell) => {
      const radius = CELL_BASE_RADIUS * Math.sqrt(cell.biomass)
      const wobble = 0.25 + Math.sin(this.now * 3 + cell.phase) * 0.2
      const speed = Math.hypot(cell.vx, cell.vy)
      const stretch = clamp(speed * 0.06, 0, 0.26)
      return {
        id: cell.id,
        x: cell.x,
        y: cell.y,
        radius,
        strain: cell.strain,
        wobble,
        squashX: 1 + stretch,
        squashY: 1 - stretch * 0.7,
        divisionProgress: cell.divisionProgress,
        deathProgress: cell.deathProgress,
      }
    })
  }

  getDebrisRenderStates(): Debris[] {
    return this.particles.map((particle) => ({
      id: particle.id,
      x: particle.x,
      y: particle.y,
      radius: particle.radius,
      alpha: particle.life,
    }))
  }

  getFieldIntensity(x: number, y: number): { nutrient: number; antibiotic: number } {
    return {
      nutrient: this.nutrient.sample(x, y),
      antibiotic: this.antibiotic.sample(x, y),
    }
  }

  private seedInitialPopulation(): void {
    for (let i = 0; i < INITIAL_WILD + INITIAL_MUTANT; i += 1) {
      const angle = Math.random() * Math.PI * 2
      const radius = Math.sqrt(Math.random()) * DISH_RADIUS * 0.72
      this.cells.push({
        id: this.cellId++,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        vx: (Math.random() - 0.5) * 8,
        vy: (Math.random() - 0.5) * 8,
        biomass: BASE_MASS * (0.92 + Math.random() * 0.2),
        strain: i < INITIAL_WILD ? 'wild' : 'mutant',
        phase: Math.random() * Math.PI * 2,
        divisionProgress: 0,
        deathProgress: 1,
        dying: false,
      })
    }
  }

  private spawnNutrients(dt: number): void {
    this.nutrientSpawnAccumulator += this.controls.nutrientSpawnRate * dt

    while (this.nutrientSpawnAccumulator >= 1) {
      this.nutrientSpawnAccumulator -= 1
      const angle = Math.random() * Math.PI * 2
      const radius = Math.sqrt(Math.random()) * DISH_RADIUS * 0.92
      this.nutrient.addGaussian(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        4.8,
        24 + Math.random() * 20,
      )
    }
  }

  private updateLivingCell(cell: Cell, dt: number, newborns: Cell[]): void {
    const [gx, gy] = this.nutrient.gradient(cell.x, cell.y)
    cell.vx += (Math.random() - 0.5) * MOTILITY_JITTER * dt + gx * CHEMOTAXIS
    cell.vy += (Math.random() - 0.5) * MOTILITY_JITTER * dt + gy * CHEMOTAXIS
    cell.vx *= DRAG
    cell.vy *= DRAG

    cell.x += cell.vx * dt
    cell.y += cell.vy * dt
    this.keepCellInsideDish(cell)

    const nutrientConcentration = this.nutrient.sample(cell.x, cell.y)
    const uptake = this.nutrient.consume(cell.x, cell.y, NUTRIENT_UPTAKE_RATE * dt)
    const monod = nutrientConcentration / (NUTRIENT_KS + nutrientConcentration + 1e-6)
    const growth = MAX_GROWTH_RATE * monod * (0.3 + uptake) * dt
    cell.biomass += growth

    if (cell.biomass > DIVISION_MASS) {
      cell.divisionProgress += dt / DIVISION_DURATION
      if (cell.divisionProgress >= 1 && this.cells.length + newborns.length < MAX_CELLS) {
        cell.divisionProgress = 0
        const newStrain =
          Math.random() < this.controls.mutationProbability ? 'mutant' : cell.strain
        const theta = Math.random() * Math.PI * 2
        const offset = CELL_BASE_RADIUS * 1.9
        cell.biomass = BASE_MASS * 1.1
        newborns.push({
          id: this.cellId++,
          x: cell.x + Math.cos(theta) * offset,
          y: cell.y + Math.sin(theta) * offset,
          vx: cell.vx + (Math.random() - 0.5) * 5,
          vy: cell.vy + (Math.random() - 0.5) * 5,
          biomass: BASE_MASS,
          strain: newStrain,
          phase: Math.random() * Math.PI * 2,
          divisionProgress: 0,
          deathProgress: 1,
          dying: false,
        })
      }
    } else {
      cell.divisionProgress = Math.max(0, cell.divisionProgress - dt * 0.4)
    }

    const drugConcentration = this.antibiotic.sample(cell.x, cell.y)
    const mic = cell.strain === 'wild' ? WILD_MIC : MUTANT_MIC
    const cPow = Math.pow(drugConcentration, ANTIBIOTIC_HILL)
    const micPow = Math.pow(mic, ANTIBIOTIC_HILL)
    const killRate = KILL_MAX * (cPow / (cPow + micPow + 1e-6))
    const deathProbability = 1 - Math.exp(-killRate * dt)
    if (Math.random() < deathProbability) {
      cell.dying = true
      cell.deathProgress = 1
      this.spawnDebris(cell.x, cell.y, cell.strain === 'wild' ? 6 : 4)
    }
  }

  private keepCellInsideDish(cell: Cell): void {
    const radius = CELL_BASE_RADIUS * Math.sqrt(cell.biomass)
    const dist = Math.hypot(cell.x, cell.y)
    const maxDist = DISH_RADIUS - radius
    if (dist <= maxDist) {
      return
    }

    const nx = cell.x / dist
    const ny = cell.y / dist
    cell.x = nx * maxDist
    cell.y = ny * maxDist

    const inward = cell.vx * nx + cell.vy * ny
    if (inward > 0) {
      cell.vx -= inward * nx * 1.6
      cell.vy -= inward * ny * 1.6
    }
  }

  private spawnDebris(x: number, y: number, count: number): void {
    for (let i = 0; i < count; i += 1) {
      const theta = Math.random() * Math.PI * 2
      const speed = 12 + Math.random() * 28
      this.particles.push({
        id: this.particleId++,
        x,
        y,
        vx: Math.cos(theta) * speed,
        vy: Math.sin(theta) * speed,
        radius: 0.9 + Math.random() * 1.8,
        life: 1,
      })
    }
  }

  private updateParticles(dt: number): void {
    const alive: Particle[] = []
    for (const particle of this.particles) {
      particle.x += particle.vx * dt
      particle.y += particle.vy * dt
      particle.vx *= 0.88
      particle.vy *= 0.88
      particle.life -= dt * 0.85
      if (particle.life > 0) {
        alive.push(particle)
      }
    }
    this.particles.length = 0
    this.particles.push(...alive)
  }

  private pushHistoryPoint(): void {
    const counts = this.getCounts()
    this.history.push({ t: this.now, wild: counts.wild, mutant: counts.mutant })
    while (this.history.length > 2) {
      const oldest = this.history[0]
      if (this.now - oldest.t <= HISTORY_SECONDS) {
        break
      }
      this.history.shift()
    }
  }
}
