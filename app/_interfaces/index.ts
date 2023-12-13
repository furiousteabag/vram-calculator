export type Unit = "MiB" | "GiB"

export enum Precision {
  full,
  half,
}

export enum Optimizer {
  Adam,
  SGD,
}

export interface ModelConfig {
  precision: Precision
  outPrecision: Precision
  numParams: number
  hiddenSize: number
  vocabSize: number
  numAttentionHeads: number
  numKeyValueHeads: number
  intermediateSize: number
}

export interface RunConfig {
  isTraining: boolean
  optimizer: Optimizer
  optimizerSGDMomentum: boolean
  sequenceLength: number
  batchSize: number
  numGPUs: number
  isFSDP: boolean
}

export interface ResultEstimation {
  cudaKernels: number
  parameters: number
  outputs?: number
  activations?: number
  gradients?: number
  firstMoments?: number
  secondMoments?: number
}
