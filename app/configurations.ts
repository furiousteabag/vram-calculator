import { ModelConfig, Optimizer, Precision, RunConfig } from "@/app/_interfaces"

export const defaultRunConfig: RunConfig = {
  inferencePrecision: Precision.half,
  trainingPrecision: Precision.mixed,
  isTraining: true,
  optimizer: Optimizer.SGD,
  optimizerSGDMomentum: true,
  sequenceLength: 512,
  batchSize: 4,
  numGPUs: 1,
  isFSDP: true,
  isInferenceModelParallelism: true,
}

export const modelConfigPresets: {
  label: string
  modelConfig: ModelConfig
}[] = [
  {
    label: "NousResearch/Llama-2-70b-hf",
    modelConfig: {
      numParams: 70,
      numLayers: 80,
      vocabSize: 32000,
      hiddenSize: 8192,
      intermediateSize: 28672,
      numAttentionHeads: 64,
      numKeyValueHeads: 8,
    },
  },
  {
    label: "NousResearch/Llama-2-13b-hf",
    modelConfig: {
      numParams: 13.058,
      numLayers: 40,
      vocabSize: 32000,
      hiddenSize: 5120,
      intermediateSize: 13824,
      numAttentionHeads: 40,
      numKeyValueHeads: 40,
    },
  },
  {
    label: "NousResearch/Llama-2-7b-hf",
    modelConfig: {
      numParams: 6.772,
      hiddenSize: 4096,
      vocabSize: 32000,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 11008,
      numLayers: 32,
    },
  },
  {
    label: "mistralai/Mistral-7B-v0.1",
    modelConfig: {
      numParams: 7.51,
      hiddenSize: 4096,
      vocabSize: 32000,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      intermediateSize: 14336,
      numLayers: 32,
    },
  },
  {
    label: "microsoft/phi-2",
    modelConfig: {
      numParams: 2.78,
      hiddenSize: 2560,
      vocabSize: 51200,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 4 * 2560,
      numLayers: 32,
    },
  },
  {
    label: "microsoft/phi-1_5",
    modelConfig: {
      numParams: 1.418,
      hiddenSize: 2048,
      vocabSize: 51200,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 4 * 2048,
      numLayers: 24,
    },
  },
  {
    label: "gpt2-xl",
    modelConfig: {
      numParams: 1.608,
      hiddenSize: 1600,
      vocabSize: 50257,
      numAttentionHeads: 25,
      numKeyValueHeads: 25,
      intermediateSize: 4 * 1600,
      numLayers: 48,
    },
  },
  {
    label: "gpt2-large",
    modelConfig: {
      numParams: 0.812,
      hiddenSize: 1280,
      vocabSize: 50257,
      numAttentionHeads: 20,
      numKeyValueHeads: 20,
      intermediateSize: 4 * 1280,
      numLayers: 36,
    },
  },
  {
    label: "gpt2-medium",
    modelConfig: {
      numParams: 0.38,
      hiddenSize: 1024,
      vocabSize: 50257,
      numAttentionHeads: 16,
      numKeyValueHeads: 16,
      intermediateSize: 4 * 1024,
      numLayers: 24,
    },
  },
  {
    label: "gpt2",
    modelConfig: {
      numParams: 0.137,
      hiddenSize: 768,
      vocabSize: 50257,
      numAttentionHeads: 12,
      numKeyValueHeads: 12,
      intermediateSize: 4 * 768,
      numLayers: 12,
    },
  },
]
