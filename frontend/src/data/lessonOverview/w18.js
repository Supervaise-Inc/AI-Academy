// Week 18 · Scaling & Deploying AI (backend/content/week-18.md)
export default {
  week: 18,
  tagline: 'Split the work across GPUs and use fewer bits so a big model trains, fits and serves.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'One GPU has two limits: memory (about 16 bytes per parameter) and time.',
    'Data parallelism cures time, tensor cures memory, pipeline splits the layers.',
    'Fewer bits and a KV cache speed serving; decoding is memory-bound.',
  ],
  outcomes: [
    'Compute the memory of a model at FP32, FP16, INT8 and INT4 and the GPUs it needs.',
    'Explain data, tensor and pipeline parallelism and compute the pipeline bubble.',
    'Explain mixed precision and quantization, and reason about batching and the KV cache.',
  ],
  terms: [
    { t: 'All-reduce', d: 'Averages the gradients of all GPUs; it moves 2(N - 1)/N of the data.' },
    { t: 'Pipeline bubble', d: 'Idle GPU time in a pipeline: (p - 1) / (m + p - 1).' },
    { t: 'Mixed precision', d: 'Compute in 16 bits, keep a 32-bit master copy, scale the loss.' },
    { t: 'Quantization', d: 'Round weights to a few integer levels with a scale; error is at most scale / 2.' },
    { t: 'Batching', d: 'Serve several requests at once so they share each read of the weights.' },
    { t: 'KV cache', d: 'Stored keys and values of earlier tokens; it can be larger than the model.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Parts 1-2', min: 65 },
    { label: 'Parts 3-4', min: 60 },
    { label: 'Parts 5-6', min: 60 },
    { label: 'Part 7', min: 20 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'micro-batches m',
    yLabel: 'idle time %',
    series: [
      { label: 'p = 4 stages', points: [[1, 75], [4, 42.9], [8, 27.3], [16, 15.8], [32, 8.6]] },
      { label: 'p = 8 stages', points: [[1, 87.5], [4, 63.6], [8, 46.7], [16, 30.4], [32, 17.9]] },
    ],
    caption: 'Pipeline bubble (p - 1)/(m + p - 1): more micro-batches shrink it.',
  },
};
