import { ModelConfig, Precision, ResultEstimation, RunConfig, Unit } from "@/app/_interfaces"

export function round(num: number, fractionDigits: number): number {
  return Number(num.toFixed(fractionDigits))
}

export function getTotalUsage({ resultEstimation, unit }: { resultEstimation: ResultEstimation; unit: Unit }): number {
  const precision = unit == "MiB" ? 0 : 3
  return round(
    resultEstimation.cudaKernels +
      resultEstimation.parameters +
      (resultEstimation.outputs || 0) +
      (resultEstimation.activations || 0) +
      (resultEstimation.gradients || 0) +
      (resultEstimation.firstMoments || 0) +
      (resultEstimation.secondMoments || 0),
    precision,
  )
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
  const bytesPerParam = modelConfig.precision == Precision.full ? 4 : 2
  const outBytesPerParam = Math.max(modelConfig.outPrecision == Precision.full ? 4 : 2, bytesPerParam)
  const divisor = unit == "MiB" ? 2 ** 20 : 2 ** 30
  const precision = unit == "MiB" ? 0 : 3
  const resultEsimation: ResultEstimation = {
    cudaKernels: round((361 * 2 ** 20) / divisor, precision),
    parameters: round((bytesPerParam * modelConfig.numParams * 10 ** 9) / divisor, precision),
    outputs: round(
      ((runConfig.isTraining ? 2 : 1) *
        outBytesPerParam *
        runConfig.batchSize *
        runConfig.sequenceLength *
        modelConfig.vocabSize) /
        divisor,
      precision,
    ),
  }
  if (runConfig.isTraining) {
    resultEsimation.gradients = round((bytesPerParam * modelConfig.numParams * 10 ** 9) / divisor, precision)
  }
  return resultEsimation
}
