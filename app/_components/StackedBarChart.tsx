import { ResultEstimation } from "@/app/_interfaces"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

export default function StackedBarChart({
  resultEstimation,
  numGPUs,
}: {
  resultEstimation: ResultEstimation
  numGPUs: number
}) {
  let data = []
  for (let i = numGPUs - 1; i >= 0; i--) {
    data.push({
      name: i,
      "CUDA Kernels": resultEstimation.cudaKernels,
      Parameters: resultEstimation.parameters,
      Outputs: resultEstimation.outputs ?? 0,
      Activations: resultEstimation.activations ?? 0,
      Gradients: resultEstimation.gradients ?? 0,
      "First Moments": resultEstimation.firstMoments ?? 0,
      "Second Moments": resultEstimation.secondMoments ?? 0,
    })
  }

  return (
    <ResponsiveContainer width="100%" height={150 + 100 * Math.log2(numGPUs)}>
      <BarChart layout="vertical" data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="name" type="category" label={{ value: "GPU Index", angle: -90 }} />
        <Tooltip />
        <Legend />
        {resultEstimation.cudaKernels != null && <Bar dataKey="CUDA Kernels" stackId="a" fill="#8884d8" />}
        {resultEstimation.parameters != null && <Bar dataKey="Parameters" stackId="a" fill="#82ca9d" />}
        {resultEstimation.activations != null && <Bar dataKey="Activations" stackId="a" fill="#ff8042" />}
        {resultEstimation.gradients != null && <Bar dataKey="Gradients" stackId="a" fill="#ba000d" />}
        {resultEstimation.firstMoments != null && <Bar dataKey="First Moments" stackId="a" fill="#f44336" />}
        {resultEstimation.secondMoments != null && <Bar dataKey="Second Moments" stackId="a" fill="#ff7961" />}
        {resultEstimation.outputs != null && <Bar dataKey="Outputs" stackId="a" fill="#ffc658" />}
      </BarChart>
    </ResponsiveContainer>
  )
}