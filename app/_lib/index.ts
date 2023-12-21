import { ModelConfig, Optimizer, Precision, ResultEstimation, RunConfig, Unit } from "@/app/_interfaces"

export function estimateResult({
  modelConfig,
  runConfig,
  unit,
}: {
  modelConfig: ModelConfig
  runConfig: RunConfig
  unit: Unit
}): ResultEstimation {
  /*
   * Auxilary variables.
   */

  const { numParams, vocabSize } = modelConfig
  const {
    optimizer,
    optimizerSGDMomentum,
    batchSize,
    sequenceLength,
    numGPUs,
    isTraining,
    trainingPrecision,
    inferencePrecision,
    isFSDP,
    isInferenceModelParallelism,
  } = runConfig

  const bytesPerParam = isTraining
    ? trainingPrecision == Precision.mixed
      ? 6
      : 4
    : inferencePrecision == Precision.full
      ? 4
      : 2
  const isParallelMode = (!isTraining && isInferenceModelParallelism) || (isTraining && isFSDP)
  const gpuDivisor = numGPUs > 1 && isParallelMode ? numGPUs : 1

  const divisor = unit == "MiB" ? 2 ** 20 : 2 ** 30
  const precision = unit == "MiB" ? 0 : 3

  /*
   * Actual calculations.
   */

  // When PyTorch uses CUDA for the first time,
  // it allocates between 300 MiB and 2 GiB of VRAM.
  const cudaKernels = 1000 * 2 ** 20

  const parameters = (bytesPerParam * numParams * 10 ** 9) / gpuDivisor
  const activations = calculateActivations({ modelConfig, runConfig })

  // On training storing probabilities after softmax
  // output which are the same size as output.
  const outputs = 4 * batchSize * sequenceLength * vocabSize * (isTraining ? 2 : 1)

  const gradients = (4 * numParams * 10 ** 9) / gpuDivisor
  const firstMoments = (4 * numParams * 10 ** 9) / gpuDivisor
  const secondMoments = (4 * numParams * 10 ** 9) / gpuDivisor

  const resultEsimation: ResultEstimation = {
    cudaKernels: round(cudaKernels / divisor, precision),
    parameters: round(parameters / divisor, precision),
    outputs: round(outputs / divisor, precision),
    activations: round(activations / divisor, precision),
    gradients: isTraining ? round(gradients / divisor, precision) : undefined,
    firstMoments:
      isTraining && ((optimizer == Optimizer.SGD && optimizerSGDMomentum) || optimizer == Optimizer.Adam)
        ? round(firstMoments / divisor, precision)
        : undefined,
    secondMoments: isTraining && optimizer == Optimizer.Adam ? round(secondMoments / divisor, precision) : undefined,
  }

  return resultEsimation
}

/**
 * Calculates number of bytes required for activations.
 * Based on https://arxiv.org/pdf/2205.05198.pdf.
 *
 * On training, returns number of bytes across all layers,
 * whereas on inference return number of bytes required to
 * store activations across single layer.
 */
function calculateActivations({ modelConfig, runConfig }: { modelConfig: ModelConfig; runConfig: RunConfig }): number {
  const { hiddenSize, numAttentionHeads, numKeyValueHeads, intermediateSize, numLayers } = modelConfig
  const { batchSize, sequenceLength, numGPUs, isTraining, trainingPrecision, inferencePrecision, isFSDP } = runConfig

  // Activations take 2 bytes in case of training with mixed
  // precision or inference with half precision.
  const bytesPerParam =
    (isTraining && trainingPrecision === Precision.mixed) || (!isTraining && inferencePrecision !== Precision.full)
      ? 2
      : 4
  const numKeyValHeads = numKeyValueHeads
  const intermSize = intermediateSize
  const headDim = hiddenSize / numAttentionHeads

  const attentionInput = bytesPerParam * batchSize * sequenceLength * hiddenSize
  const q = bytesPerParam * batchSize * sequenceLength * headDim * numAttentionHeads
  const k = bytesPerParam * batchSize * sequenceLength * headDim * numKeyValHeads
  const softmaxOutput = bytesPerParam * batchSize * numAttentionHeads * Math.pow(sequenceLength, 2)
  const softmaxDropoutMask = 1 * batchSize * numAttentionHeads * Math.pow(sequenceLength, 2)
  const dropoutOutput = bytesPerParam * batchSize * numAttentionHeads * Math.pow(sequenceLength, 2)
  const v = bytesPerParam * batchSize * sequenceLength * headDim * numKeyValHeads
  const outProjInput = bytesPerParam * batchSize * sequenceLength * numAttentionHeads * headDim
  const attentionDropout = 1 * batchSize * sequenceLength * hiddenSize
  const attentionBlock =
    attentionInput + q + k + softmaxOutput + v + outProjInput + softmaxDropoutMask + dropoutOutput + attentionDropout

  const mlpInput = bytesPerParam * batchSize * sequenceLength * hiddenSize
  const activationInput = bytesPerParam * batchSize * sequenceLength * intermSize
  const downProjInput = bytesPerParam * batchSize * sequenceLength * intermSize
  const dropoutMask = 1 * batchSize * sequenceLength * hiddenSize
  const mlpBlock = mlpInput + activationInput + downProjInput + dropoutMask

  const layerNorms = bytesPerParam * batchSize * sequenceLength * hiddenSize * 2

  const layer = attentionBlock + mlpBlock + layerNorms

  let activations = isTraining ? layer * numLayers : layer

  if (isTraining && numGPUs > 1 && isFSDP) {
    activations /= numGPUs
  }

  return activations
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

export function round(num: number, fractionDigits: number): number {
  return Number(num.toFixed(fractionDigits))
}
