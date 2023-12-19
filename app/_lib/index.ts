import { ModelConfig, Optimizer, Precision, ResultEstimation, RunConfig, Unit } from "@/app/_interfaces"

export function round(num: number, fractionDigits: number): number {
  return Number(num.toFixed(fractionDigits))
}

export function getTotalUsagePerGPU({
  resultEstimation,
  unit,
  isFirst,
}: {
  resultEstimation: ResultEstimation
  unit: Unit
  isFirst: boolean
}): number {
  const precision = unit == "MiB" ? 0 : 3
  return round(
    resultEstimation.cudaKernels +
      resultEstimation.parameters +
      (resultEstimation.outputs || 0) * Number(isFirst) +
      (resultEstimation.activations || 0) +
      (resultEstimation.gradients || 0) +
      (resultEstimation.firstMoments || 0) +
      (resultEstimation.secondMoments || 0),
    precision,
  )
}

function calculateActivations({ modelConfig, runConfig }: { modelConfig: ModelConfig; runConfig: RunConfig }): number {
  const bytesPerParam = runConfig.isTraining
    ? runConfig.trainingPrecision == Precision.mixed
      ? 2
      : 4
    : runConfig.inferencePrecision == Precision.full
      ? 4
      : 2

  const { hiddenSize, numAttentionHeads, numKeyValueHeads, intermediateSize, numLayers } = modelConfig
  const numKeyValHeads = numKeyValueHeads
  const intermSize = intermediateSize
  const headDim = hiddenSize / numAttentionHeads

  const { batchSize: bs, sequenceLength: seqLength } = runConfig

  const attentionInput = bytesPerParam * bs * seqLength * hiddenSize
  const q = bytesPerParam * bs * seqLength * headDim * numAttentionHeads
  const k = bytesPerParam * bs * seqLength * headDim * numKeyValHeads
  const softmaxOutput = bytesPerParam * bs * numAttentionHeads * Math.pow(seqLength, 2)
  const softmaxDropoutMask = 1 * bs * numAttentionHeads * Math.pow(seqLength, 2)
  const dropoutOutput = bytesPerParam * bs * numAttentionHeads * Math.pow(seqLength, 2)
  const v = bytesPerParam * bs * seqLength * headDim * numKeyValHeads
  const outProjInput = bytesPerParam * bs * seqLength * numAttentionHeads * headDim
  const attentionDropout = 1 * bs * seqLength * hiddenSize
  const attentionBlock =
    attentionInput + q + k + softmaxOutput + v + outProjInput + softmaxDropoutMask + dropoutOutput + attentionDropout

  const mlpInput = bytesPerParam * bs * seqLength * hiddenSize
  const activationInput = bytesPerParam * bs * seqLength * intermSize
  const downProjInput = bytesPerParam * bs * seqLength * intermSize
  const dropoutMask = 1 * bs * seqLength * hiddenSize
  const mlpBlock = mlpInput + activationInput + downProjInput + dropoutMask

  const layerNorms = bytesPerParam * bs * seqLength * hiddenSize * 2

  const layer = attentionBlock + mlpBlock + layerNorms

  return runConfig.isTraining ? layer * numLayers : layer
}

export function estimateResult({
  modelConfig,
  runConfig,
  unit,
}: {
  modelConfig: ModelConfig
  runConfig: RunConfig
  unit: Unit
}): ResultEstimation {
  const divisor = unit == "MiB" ? 2 ** 20 : 2 ** 30
  const precision = unit == "MiB" ? 0 : 3

  const bytesPerParam = runConfig.isTraining
    ? runConfig.trainingPrecision == Precision.mixed
      ? 6
      : 4
    : runConfig.inferencePrecision == Precision.full
      ? 4
      : 2

  // const activations =
  //   (runConfig.isTraining
  //     ? runConfig.trainingPrecision == Precision.mixed
  //       ? 1 * modelConfig.numLayers
  //       : 2 * modelConfig.numLayers
  //     : runConfig.inferencePrecision == Precision.full
  //       ? 2
  //       : 1) *
  //   runConfig.sequenceLength *
  //   runConfig.batchSize *
  //   modelConfig.hiddenSize *
  //   (34 + (5 * modelConfig.numAttentionHeads * runConfig.sequenceLength) / modelConfig.hiddenSize)
  // const outBytesPerParam = Math.max(modelConfig.outPrecision == Precision.full ? 4 : 2, bytesPerParam)

  const activations = calculateActivations({ modelConfig, runConfig })

  const gpuDivisor =
    !runConfig.isTraining && runConfig.numGPUs > 1 && runConfig.isInferenceModelParallelism ? runConfig.numGPUs : 1

  const resultEsimation: ResultEstimation = {
    cudaKernels: round((1000 * 2 ** 20) / divisor, precision),
    parameters: round((bytesPerParam * modelConfig.numParams * 10 ** 9) / gpuDivisor / divisor, precision),
    outputs: round(
      ((runConfig.isTraining ? 2 : 1) * 4 * runConfig.batchSize * runConfig.sequenceLength * modelConfig.vocabSize) /
        divisor,
      precision,
    ),
    activations: round(activations / divisor, precision),
  }

  if (runConfig.isTraining) {
    resultEsimation.gradients = round((4 * modelConfig.numParams * 10 ** 9) / divisor, precision)
    if (runConfig.optimizer == Optimizer.SGD && runConfig.optimizerSGDMomentum) {
      resultEsimation.firstMoments = round((4 * modelConfig.numParams * 10 ** 9) / divisor, precision)
    }
    if (runConfig.optimizer == Optimizer.Adam) {
      resultEsimation.firstMoments = round((4 * modelConfig.numParams * 10 ** 9) / divisor, precision)
      resultEsimation.secondMoments = round((4 * modelConfig.numParams * 10 ** 9) / divisor, precision)
    }
  }
  return resultEsimation
}
