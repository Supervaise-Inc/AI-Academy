// Week 18 · Scaling & Deploying AI (backend/content/week-18.md)
export default {
  week: 18,
  tagline: 'Big models need splitting across GPUs and fewer bits per number.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'Weight memory is parameters times bytes per parameter.',
    'Data, tensor and pipeline parallelism split the work in different ways.',
    'Mixed precision speeds training; quantization shrinks a model for inference.',
  ],
  outcomes: [
    'Explain why one GPU is not enough for large models.',
    'Compare data, tensor and pipeline parallelism and say when each is used.',
    'Estimate model memory at FP32, FP16, INT8 and INT4.',
  ],
  terms: [
    { t: 'Precision', d: 'The number format used to store a value, such as FP32, FP16 or INT8.' },
    { t: 'Data parallelism', d: 'Every GPU holds a full model copy and handles a different slice of the batch.' },
    { t: 'Tensor parallelism', d: 'A big weight matrix in one layer is cut into pieces, one per GPU.' },
    { t: 'Pipeline parallelism', d: 'Groups of layers run as stages on different GPUs, fed with micro-batches.' },
    { t: 'Quantization', d: 'Weights stored as low-bit integers plus a scale: smaller and often faster.' },
    { t: 'KV cache', d: 'Stored attention keys and values, so new tokens skip recomputing the prompt.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Memory sums', min: 20 },
    { label: 'Precision lab', min: 120 },
    { label: 'Parallel check', min: 30 },
    { label: 'Reflect', min: 10 },
  ],
  graphic: {
    type: 'bars',
    unit: ' GB',
    items: [
      { label: 'FP32', value: 28 },
      { label: 'FP16', value: 14 },
      { label: 'INT8', value: 7 },
      { label: 'INT4', value: 3.5 },
    ],
    caption: 'Weight memory of a 7B model: fewer bits per number, less memory.',
  },
};
