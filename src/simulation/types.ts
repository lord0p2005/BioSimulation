export type Strain = 'wild' | 'mutant'

export interface ControlsConfig {
  nutrientSpawnRate: number
  mutationProbability: number
}

export interface PopulationPoint {
  t: number
  wild: number
  mutant: number
}

export interface Counts {
  wild: number
  mutant: number
}

export interface CellRenderState {
  id: number
  x: number
  y: number
  radius: number
  strain: Strain
  wobble: number
  squashX: number
  squashY: number
  divisionProgress: number
  deathProgress: number
}

export interface Debris {
  id: number
  x: number
  y: number
  radius: number
  alpha: number
}
