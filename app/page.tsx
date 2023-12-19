"use client"

import StackedBarChart from "@/app/_components/StackedBarChart"
import { Optimizer, Precision, Unit } from "@/app/_interfaces"
import { estimateResult, getTotalUsagePerGPU } from "@/app/_lib"
import { defaultRunConfig, modelConfigPresets } from "@/app/configurations"
import { HelpOutline } from "@mui/icons-material"
import { IconButton, Tooltip } from "@mui/material"
import Autocomplete from "@mui/material/Autocomplete"
import Box from "@mui/material/Box"
import Checkbox from "@mui/material/Checkbox"
import Chip from "@mui/material/Chip"
import Container from "@mui/material/Container"
import FormControlLabel from "@mui/material/FormControlLabel"
import Grid from "@mui/material/Grid"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import ListItemText from "@mui/material/ListItemText"
import Stack from "@mui/material/Stack"
import Switch from "@mui/material/Switch"
import TextField from "@mui/material/TextField"
import Typography from "@mui/material/Typography"
import React, { useState } from "react"

export default function App() {
  const [modelConfigPreset, setModelConfigPreset] = useState(modelConfigPresets[5])
  const [modelConfig, setModelConfig] = useState(modelConfigPresets[5].modelConfig)
  const [runConfig, setRunConfig] = useState(defaultRunConfig)
  const [resultUnit, setResultUnit] = useState<Unit>("MiB")

  const resultEstimation = estimateResult({ modelConfig, runConfig, unit: resultUnit })

  return (
    <Container maxWidth="md">
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12}>
          <Typography variant="h4" align="center" sx={{ fontWeight: "bold" }}>
            VRAM Estimator
          </Typography>
          <Typography variant="subtitle1" align="center">
            Estimate GPU VRAM usage of transformer-based models
          </Typography>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Stack spacing={2} justifyItems="center">
            <Typography variant="h5" align="center" sx={{ fontWeight: "bold" }}>
              Running Parameters
            </Typography>

            <Stack spacing={1} direction="row" justifyContent="center">
              <Chip
                label="Inference"
                color="primary"
                variant={runConfig.isTraining ? "outlined" : "filled"}
                onClick={() => setRunConfig({ ...runConfig, isTraining: false })}
              />
              <Chip
                label="Training"
                color="primary"
                variant={runConfig.isTraining ? "filled" : "outlined"}
                onClick={() => setRunConfig({ ...runConfig, isTraining: true })}
              />
            </Stack>

            {!runConfig.isTraining && (
              <Stack spacing={1} direction="row" alignItems="center" justifyContent="left">
                <Typography variant="body1">Precision: </Typography>
                <Chip
                  label="fp16/bf16"
                  color="primary"
                  variant={runConfig.inferencePrecision == Precision.half ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, inferencePrecision: Precision.half })}
                />
                <Chip
                  label="fp32"
                  color="primary"
                  variant={runConfig.inferencePrecision == Precision.full ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, inferencePrecision: Precision.full })}
                />
              </Stack>
            )}

            {runConfig.isTraining && (
              <Stack spacing={1} direction="row" alignItems="center" justifyContent="left">
                <Typography variant="body1">Precision: </Typography>
                <Chip
                  label="mixed"
                  color="primary"
                  variant={runConfig.trainingPrecision == Precision.mixed ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, trainingPrecision: Precision.mixed })}
                />
                <Chip
                  label="full (fp32)"
                  color="primary"
                  variant={runConfig.trainingPrecision == Precision.full ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, trainingPrecision: Precision.full })}
                />
                <Tooltip
                  title="Mixed Precision training mode maintains additional model parameters in half precision, speeding up forward pass and reducing activations size"
                  enterTouchDelay={10}
                  leaveTouchDelay={5000}
                >
                  <IconButton>
                    <HelpOutline />
                  </IconButton>
                </Tooltip>
              </Stack>
            )}

            {runConfig.isTraining && (
              <Stack spacing={1} direction="row" alignItems="center" justifyContent="left">
                <Typography variant="body1">Optimizer: </Typography>
                <Chip
                  label="Adam"
                  color="primary"
                  variant={runConfig.optimizer === Optimizer.Adam ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, optimizer: Optimizer.Adam })}
                />
                <Chip
                  label="SGD"
                  color="primary"
                  variant={runConfig.optimizer === Optimizer.SGD ? "filled" : "outlined"}
                  onClick={() => setRunConfig({ ...runConfig, optimizer: Optimizer.SGD })}
                />

                {runConfig.isTraining && runConfig.optimizer == Optimizer.SGD && (
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={runConfig.optimizerSGDMomentum}
                        onChange={(e) => setRunConfig({ ...runConfig, optimizerSGDMomentum: e.target.checked })}
                      />
                    }
                    label="momentum"
                  />
                )}
              </Stack>
            )}

            <Stack spacing={1} direction="row" alignItems="top" justifyContent="center">
              <TextField
                label="Sequence Length"
                value={runConfig.sequenceLength > 0 ? runConfig.sequenceLength : ""}
                error={runConfig.sequenceLength === 0}
                onChange={(e) =>
                  Number(e.target.value) >= 0
                    ? setRunConfig({ ...runConfig, sequenceLength: Number(e.target.value) })
                    : runConfig.sequenceLength
                }
                helperText={runConfig.sequenceLength === 0 ? "Can't be empty!" : ""}
              />

              <TextField
                label={runConfig.numGPUs > 1 ? "Per GPU Batch Size" : "Batch Size"}
                value={runConfig.batchSize > 0 ? runConfig.batchSize : ""}
                error={runConfig.batchSize === 0}
                onChange={(e) =>
                  Number(e.target.value) >= 0
                    ? setRunConfig({ ...runConfig, batchSize: Number(e.target.value) })
                    : runConfig.batchSize
                }
                helperText={runConfig.batchSize === 0 ? "Can't be empty!" : ""}
              />
            </Stack>

            <TextField
              label="Number of GPUs"
              value={runConfig.numGPUs > 0 ? runConfig.numGPUs : ""}
              error={runConfig.numGPUs === 0}
              onChange={(e) =>
                Number(e.target.value) >= 0
                  ? setRunConfig({ ...runConfig, numGPUs: Number(e.target.value) })
                  : runConfig.numGPUs
              }
              helperText={runConfig.numGPUs === 0 ? "Can't be empty!" : ""}
            />

            {runConfig.isTraining && runConfig.numGPUs > 1 && (
              <FormControlLabel
                control={
                  <Switch
                    checked={runConfig.isFSDP}
                    onClick={() => setRunConfig({ ...runConfig, isFSDP: !runConfig.isFSDP })}
                  />
                }
                label="FSDP parallelization"
              />
            )}

            {!runConfig.isTraining && runConfig.numGPUs > 1 && (
              <Stack spacing={1} direction="row" alignItems="center" justifyContent="left">
                <FormControlLabel
                  control={
                    <Switch
                      checked={runConfig.isInferenceModelParallelism}
                      onClick={() =>
                        setRunConfig({
                          ...runConfig,
                          isInferenceModelParallelism: !runConfig.isInferenceModelParallelism,
                        })
                      }
                    />
                  }
                  label="Model Parallelism"
                />
                <Tooltip
                  title="Split model layers across available GPUs. This is default behaviour when using device_map='auto' on model loading with transformers and accelerate"
                  enterTouchDelay={10}
                  leaveTouchDelay={5000}
                >
                  <IconButton>
                    <HelpOutline />
                  </IconButton>
                </Tooltip>
              </Stack>
            )}

            <Typography variant="h5" align="center" sx={{ fontWeight: "bold" }}>
              Model Parameters
            </Typography>

            <Typography variant="body1" align="center">
              Model Parameters could be taken from <code>config.json</code> on HuggingFace or directly from model via{" "}
              <code>model.config</code>
            </Typography>

            <Autocomplete
              options={modelConfigPresets}
              renderInput={(params) => <TextField {...params} label="Parameters Preset" />}
              value={modelConfigPreset.modelConfig == modelConfig ? modelConfigPreset : null}
              onChange={(_event, newValue) => {
                if (newValue) {
                  setModelConfigPreset(newValue)
                  setModelConfig(newValue.modelConfig)
                }
              }}
            />

            <TextField
              label="Number of Parameters (billions)"
              value={modelConfig.numParams > 0 ? modelConfig.numParams : ""}
              error={modelConfig.numParams === 0}
              onChange={(e) =>
                Number(e.target.value) >= 0
                  ? setModelConfig({ ...modelConfig, numParams: Number(e.target.value) })
                  : modelConfig.numParams
              }
              helperText={modelConfig.numParams === 0 ? "Can't be empty!" : ""}
            />

            <Stack spacing={1} direction="row" alignItems="top" justifyContent="center">
              <TextField
                label="Number of Layers"
                value={modelConfig.numLayers > 0 ? modelConfig.numLayers : ""}
                error={modelConfig.numLayers === 0}
                onChange={(e) =>
                  Number(e.target.value) >= 0
                    ? setModelConfig({ ...modelConfig, numLayers: Number(e.target.value) })
                    : modelConfig.numLayers
                }
                helperText={modelConfig.numLayers === 0 ? "Can't be empty!" : ""}
              />

              <TextField
                label="Vocab Size"
                value={modelConfig.vocabSize > 0 ? modelConfig.vocabSize : ""}
                error={modelConfig.vocabSize === 0}
                onChange={(e) =>
                  Number(e.target.value) >= 0
                    ? setModelConfig({ ...modelConfig, vocabSize: Number(e.target.value) })
                    : modelConfig.vocabSize
                }
                helperText={modelConfig.vocabSize === 0 ? "Can't be empty!" : ""}
              />
            </Stack>

            <Stack spacing={1} direction="row" alignItems="top" justifyContent="center">
              <Box flex={1}>
                <TextField
                  label="Hidden Size"
                  value={modelConfig.hiddenSize > 0 ? modelConfig.hiddenSize : ""}
                  error={modelConfig.hiddenSize === 0}
                  onChange={(e) =>
                    Number(e.target.value) >= 0
                      ? setModelConfig({ ...modelConfig, hiddenSize: Number(e.target.value) })
                      : modelConfig.hiddenSize
                  }
                  helperText={modelConfig.hiddenSize === 0 ? "Can't be empty!" : ""}
                />
              </Box>

              <Box flex={1}>
                <TextField
                  label="Number of Attention Heads"
                  value={modelConfig.numAttentionHeads > 0 ? modelConfig.numAttentionHeads : ""}
                  error={modelConfig.numAttentionHeads === 0}
                  onChange={(e) =>
                    Number(e.target.value) >= 0
                      ? setModelConfig({ ...modelConfig, numAttentionHeads: Number(e.target.value) })
                      : modelConfig.numAttentionHeads
                  }
                  helperText={modelConfig.numAttentionHeads === 0 ? "Can't be empty!" : ""}
                />
              </Box>
            </Stack>

            <Stack spacing={1} direction="row" alignItems="top" justifyContent="center">
              <Box flex={1}>
                <TextField
                  label="Intermediate Size"
                  value={modelConfig.intermediateSize > 0 ? modelConfig.intermediateSize : ""}
                  error={modelConfig.intermediateSize === 0}
                  onChange={(e) =>
                    Number(e.target.value) >= 0
                      ? setModelConfig({ ...modelConfig, intermediateSize: Number(e.target.value) })
                      : modelConfig.intermediateSize
                  }
                  helperText={
                    modelConfig.intermediateSize === 0
                      ? "Can't be empty!"
                      : "Expanding dimensionality within MLP block. Usually it is 4 × hidden size."
                  }
                />
              </Box>

              <Box flex={1}>
                <TextField
                  label="Number of Key Value Heads"
                  value={modelConfig.numKeyValueHeads > 0 ? modelConfig.numKeyValueHeads : ""}
                  error={modelConfig.numKeyValueHeads === 0}
                  onChange={(e) =>
                    Number(e.target.value) >= 0
                      ? setModelConfig({ ...modelConfig, numKeyValueHeads: Number(e.target.value) })
                      : modelConfig.numKeyValueHeads
                  }
                  helperText={
                    modelConfig.numKeyValueHeads === 0
                      ? "Can't be empty!"
                      : "Might be different from number of attention heads when using Grouped Query Attention"
                  }
                />
              </Box>
            </Stack>
          </Stack>
        </Grid>
        <Grid item alignItems="center" xs={12} sm={6}>
          <Stack spacing={2} justifyItems="center">
            <Typography variant="h5" align="center" sx={{ fontWeight: "bold" }}>
              Estimation Result
            </Typography>

            <Stack spacing={1} direction="row" justifyContent="center">
              <Chip
                label="MiB"
                color="primary"
                variant={resultUnit == "MiB" ? "filled" : "outlined"}
                onClick={() => setResultUnit("MiB")}
              />
              <Chip
                label="GiB"
                color="primary"
                variant={resultUnit == "GiB" ? "filled" : "outlined"}
                onClick={() => setResultUnit("GiB")}
              />
            </Stack>

            <StackedBarChart resultEstimation={resultEstimation} numGPUs={runConfig.numGPUs} />

            <List dense={true}>
              <ListItem>
                <ListItemText
                  primary={
                    <Typography variant="body1">
                      Total VRAM usage is{" "}
                      <b>
                        {getTotalUsagePerGPU({
                          resultEstimation,
                          unit: resultUnit,
                          isFirst: true,
                        })}{" "}
                      </b>{" "}
                      {resultUnit} {runConfig.numGPUs > 1 ? "on 0-th GPU" : ""}
                    </Typography>
                  }
                  secondary={
                    runConfig.numGPUs > 1 && (
                      <Typography variant="body1">
                        Total VRAM usage is{" "}
                        <b>
                          {getTotalUsagePerGPU({
                            resultEstimation,
                            unit: resultUnit,
                            isFirst: false,
                          })}
                        </b>{" "}
                        {resultUnit} on rest GPUs
                      </Typography>
                    )
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      <span style={{ color: "#8884d8", fontWeight: "bold" }}>CUDA Kernels</span> use
                      <span style={{ fontWeight: "bold" }}> {resultEstimation.cudaKernels} </span>
                      {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                    </span>
                  }
                  secondary={`When PyTorch uses CUDA for the first time, it allocates between 300 MiB and 2 GiB of VRAM`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      <span style={{ color: "#76ba1b", fontWeight: "bold" }}>Parameters</span> use
                      <span style={{ fontWeight: "bold" }}> {resultEstimation.parameters} </span>
                      {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                    </span>
                  }
                  secondary={`Number of Parameters (${
                    modelConfig.numParams
                  } billion) × number of bytes per parameter (${
                    runConfig.isTraining
                      ? runConfig.trainingPrecision == Precision.mixed
                        ? "6; parameters are stored in both full precision and half precision"
                        : "4"
                      : runConfig.inferencePrecision == Precision.full
                        ? 4
                        : 2
                  }) ${
                    !runConfig.isTraining && runConfig.numGPUs > 1 && runConfig.isInferenceModelParallelism
                      ? `÷ Number of GPUs (${runConfig.numGPUs})`
                      : ""
                  }`}
                />
              </ListItem>

              {resultEstimation.activations && (
                <ListItem>
                  <ListItemText
                    primary={
                      <span>
                        <span style={{ color: "#a4de02", fontWeight: "bold" }}>Activations</span> use
                        <span style={{ fontWeight: "bold" }}> {resultEstimation.activations} </span>
                        {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                      </span>
                    }
                    secondary={
                      (runConfig.isTraining
                        ? `Sum of sizes of all intermediate tensors during forward pass across all ${modelConfig.numLayers} layers.`
                        : `Size of a biggest tensor within forward pass. It is estimated as the sum of all intermediate tensors within computation of a single layer.`) +
                      ` Activations size have quadratic dependence on Sequence Length.`
                    }
                  />
                </ListItem>
              )}

              {resultEstimation.gradients && (
                <ListItem>
                  <ListItemText
                    primary={
                      <span>
                        <span style={{ color: "#ba000d", fontWeight: "bold" }}>Gradients</span> use
                        <span style={{ fontWeight: "bold" }}> {resultEstimation.gradients} </span>
                        {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                      </span>
                    }
                    secondary={`Gradient is stored for each parameter in full precision, so it is Number of Parameters (${modelConfig.numParams} billion) × number of bytes per parameter (4)`}
                  />
                </ListItem>
              )}

              {resultEstimation.firstMoments && (
                <ListItem>
                  <ListItemText
                    primary={
                      <span>
                        <span style={{ color: "#f44336", fontWeight: "bold" }}>First Moments</span> use
                        <span style={{ fontWeight: "bold" }}> {resultEstimation.firstMoments} </span>
                        {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                      </span>
                    }
                    secondary={`Optimizer stores moving average of gradients for each parameter in full precision, so it is Number of Parameters (${modelConfig.numParams} billion) × number of bytes per parameter (4)`}
                  />
                </ListItem>
              )}

              {resultEstimation.secondMoments && (
                <ListItem>
                  <ListItemText
                    primary={
                      <span>
                        <span style={{ color: "#ff7961", fontWeight: "bold" }}>Second Moments</span> use
                        <span style={{ fontWeight: "bold" }}> {resultEstimation.secondMoments} </span>
                        {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "per GPU" : ""}
                      </span>
                    }
                    secondary={`Optimizer stores moving average of squared gradients for each parameter in full precision, so it is Number of Parameters (${modelConfig.numParams} billion) × number of bytes per parameter (4)`}
                  />
                </ListItem>
              )}

              {resultEstimation.outputs && (
                <ListItem>
                  <ListItemText
                    primary={
                      <span>
                        <span style={{ color: "#ffc658", fontWeight: "bold" }}>Output tensor</span> uses
                        <span style={{ fontWeight: "bold" }}> {resultEstimation.outputs} </span>
                        {resultUnit} of VRAM {runConfig.numGPUs > 1 ? "(same GPU as inputs)" : ""}
                      </span>
                    }
                    secondary={`Batch Size (${runConfig.batchSize}) × Sequence Length (${
                      runConfig.sequenceLength
                    }) × Vocabulary Size (${modelConfig.vocabSize}) × number of bytes per parameter (4) ${
                      runConfig.isTraining
                        ? "× 2 (storing probabilities after softmax output which are the same size as output)"
                        : runConfig.inferencePrecision != Precision.full
                          ? "(even we infer model in half precision, outputs are still almost always casted to fp32 within the model itself)"
                          : ""
                    }`}
                  />
                </ListItem>
              )}
            </List>
          </Stack>
        </Grid>
      </Grid>
    </Container>
  )
}
