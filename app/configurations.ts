import { ModelConfig, Optimizer, Precision, RunConfig } from "@/app/_interfaces"

export const defaultRunConfig: RunConfig = {
  isTraining: false,
  optimizer: Optimizer.Adam,
  optimizerSGDMomentum: false,
  sequenceLength: 1024,
  batchSize: 8,
  numGPUs: 1,
  isFSDP: false,
}

export const modelConfigPresets: {
  label: string
  modelConfig: ModelConfig
}[] = [
  {
    label: "NousResearch/Llama-2-7b-hf",
    modelConfig: {
      precision: Precision.half,
      outPrecision: Precision.full,
      numParams: 6.738,
      hiddenSize: 4096,
      vocabSize: 32000,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 11008,
    },
  },
  {
    label: "mistralai/Mistral-7B-v0.1",
    modelConfig: {
      precision: Precision.half,
      outPrecision: Precision.full,
      numParams: 7.242,
      hiddenSize: 4096,
      vocabSize: 32000,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      intermediateSize: 14336,
    },
  },
  {
    label: "microsoft/phi-2",
    modelConfig: {
      precision: Precision.half,
      outPrecision: Precision.full,
      numParams: 2.78,
      hiddenSize: 2560,
      vocabSize: 51200,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 4 * 2560,
    },
  },
  {
    label: "microsoft/phi-1_5",
    modelConfig: {
      precision: Precision.half,
      outPrecision: Precision.full,
      numParams: 1.418,
      hiddenSize: 2048,
      vocabSize: 51200,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      intermediateSize: 4 * 2048,
    },
  },
  {
    label: "gpt2-xl",
    modelConfig: {
      precision: Precision.full,
      outPrecision: Precision.half,
      numParams: 1.558,
      hiddenSize: 1600,
      vocabSize: 50257,
      numAttentionHeads: 25,
      numKeyValueHeads: 25,
      intermediateSize: 4 * 1600,
    },
  },
  {
    label: "gpt2-large",
    modelConfig: {
      precision: Precision.full,
      outPrecision: Precision.half,
      numParams: 0.774,
      hiddenSize: 1280,
      vocabSize: 50257,
      numAttentionHeads: 20,
      numKeyValueHeads: 20,
      intermediateSize: 4 * 1280,
    },
  },
  {
    label: "gpt2-medium",
    modelConfig: {
      precision: Precision.full,
      outPrecision: Precision.half,
      numParams: 0.355,
      hiddenSize: 1024,
      vocabSize: 50257,
      numAttentionHeads: 16,
      numKeyValueHeads: 16,
      intermediateSize: 4 * 1024,
    },
  },
  {
    label: "gpt2",
    modelConfig: {
      precision: Precision.full,
      outPrecision: Precision.half,
      numParams: 0.124,
      hiddenSize: 768,
      vocabSize: 50257,
      numAttentionHeads: 12,
      numKeyValueHeads: 12,
      intermediateSize: 4 * 768,
    },
  },
]
