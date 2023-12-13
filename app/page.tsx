"use client"

import StackedBarChart from "@/app/_components/StackedBarChart"
import { Optimizer, Precision, Unit } from "@/app/_interfaces"
import { estimateResult, getTotalUsage } from "@/app/_lib"
import { defaultRunConfig, modelConfigPresets } from "@/app/configurations"
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
  const [modelConfigPreset, setModelConfigPreset] = useState(modelConfigPresets[0])
  const [modelConfig, setModelConfig] = useState(modelConfigPresets[0].modelConfig)
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
            Estimate GPU VRAM usage based on params
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

            {runConfig.isTraining && (
              <Stack spacing={1} direction="row" justifyContent="center">
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
              </Stack>
            )}

            {runConfig.isTraining && runConfig.optimizer == Optimizer.SGD && (
              <Box display="flex" justifyContent="center">
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={runConfig.optimizerSGDMomentum}
                      onChange={(e) => setRunConfig({ ...runConfig, optimizerSGDMomentum: e.target.checked })}
                    />
                  }
                  label="momentum"
                />
              </Box>
            )}

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

            {runConfig.isTraining && (
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

            <Typography variant="h5" align="center" sx={{ fontWeight: "bold" }}>
              Model Parameters
            </Typography>

            <Autocomplete
              options={modelConfigPresets}
              renderInput={(params) => <TextField {...params} label="Parameters Preset" />}
              value={modelConfigPreset.modelConfig == modelConfig ? modelConfigPreset : null}
              onChange={(event, newValue) => {
                if (newValue) {
                  setModelConfigPreset(newValue)
                  setModelConfig(newValue.modelConfig)
                }
              }}
            />

            <Stack spacing={1} direction="row" alignItems="center" justifyContent="center">
              <Chip
                label="fp16/bf16"
                color="primary"
                variant={modelConfig.precision == Precision.half ? "filled" : "outlined"}
                onClick={() => setModelConfig({ ...modelConfig, precision: Precision.half })}
              />
              <Chip
                label="fp32"
                color="primary"
                variant={modelConfig.precision == Precision.full ? "filled" : "outlined"}
                onClick={() => setModelConfig({ ...modelConfig, precision: Precision.full })}
              />
            </Stack>

            {modelConfig.precision != Precision.full && (
              <Stack spacing={1} alignItems="center" justifyContent="center">
                <Stack spacing={1} direction="row" alignItems="center" justifyContent="center">
                  <Typography variant="body1">Output Tensor dtype:</Typography>
                  <Chip
                    label="fp16/bf16"
                    color="primary"
                    variant={modelConfig.outPrecision == Precision.half ? "filled" : "outlined"}
                    onClick={() => setModelConfig({ ...modelConfig, outPrecision: Precision.half })}
                  />
                  <Chip
                    label="fp32"
                    color="primary"
                    variant={modelConfig.outPrecision == Precision.full ? "filled" : "outlined"}
                    onClick={() => setModelConfig({ ...modelConfig, outPrecision: Precision.full })}
                  />
                </Stack>

                <Typography variant="body2" color="textSecondary">
                  Usually, output type is the same as weights type, but Llama and Mistral models explicitly convert
                  output to float32 with `.float()`
                </Typography>
              </Stack>
            )}

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
                  : "Expanding dimensionality within MLP block. Usully it is 4 * hidden size."
              }
            />
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
                  primary={`Total VRAM usage is ${getTotalUsage({
                    resultEstimation,
                    unit: resultUnit,
                  })} ${resultUnit} ${runConfig.numGPUs > 1 ? "per GPU" : ""}`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={`CUDA kernels use ${resultEstimation.cudaKernels} ${resultUnit} of VRAM ${
                    runConfig.numGPUs > 1 ? "per GPU" : ""
                  }`}
                  secondary={`Fixed value`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={`Model parameters use ${resultEstimation.parameters} ${resultUnit} of VRAM ${
                    runConfig.numGPUs > 1 ? "per GPU" : ""
                  }`}
                  secondary={`Number of parameters (${
                    modelConfig.numParams
                  } billion) * number of bytes per parameter (${modelConfig.precision == Precision.full ? 4 : 2})`}
                />
              </ListItem>

              {resultEstimation.gradients && (
                <ListItem>
                  <ListItemText
                    primary={`Gradients use ${resultEstimation.gradients} ${resultUnit} of VRAM ${
                      runConfig.numGPUs > 1 ? "per GPU" : ""
                    }`}
                    secondary={`Gradient is stored for each parameter with same precision (${
                      modelConfig.precision == Precision.full ? 4 : 2
                    } bytes) as parameter itself`}
                  />
                </ListItem>
              )}
              {resultEstimation.outputs && (
                <ListItem>
                  <ListItemText
                    primary={`Output tensor uses ${resultEstimation.outputs} ${resultUnit} of VRAM ${
                      runConfig.numGPUs > 1 ? "per GPU" : ""
                    }`}
                    secondary={`Batch size (${runConfig.batchSize}) * sequence length (${
                      runConfig.sequenceLength
                    }) * vocabulary size (${modelConfig.vocabSize}) * number of bytes per parameter (${Math.max(
                      modelConfig.outPrecision == Precision.full ? 4 : 2,
                      modelConfig.precision == Precision.full ? 4 : 2,
                    )}) ${
                      runConfig.isTraining
                        ? "* 2 (we have to store probabilities after softmax output which are the same size as output)"
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
