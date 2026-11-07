// Week 11 · Multimodal AI (backend/content/week-11.md)
export default {
  week: 11,
  tagline: 'Cut a picture into patches and treat them as tokens; CLIP shares one space with text.',
  time: 'About 4 hours: a 60-minute lesson, then a 3-hour lab',
  ideas: [
    'A ViT cuts an image into patches: 224 ÷ 16 = 14, so 196 tokens plus a CLS token.',
    'Self-attention cannot see order, so patches need position embeddings.',
    'CLIP puts images and text in one space; cosine similarity gives labels.',
  ],
  outcomes: [
    'Compute the patch tokens, patch vector length and embedding parameters of a ViT.',
    'Explain why a bag of patches needs position embeddings.',
    'Compute a cosine-similarity matrix, zero-shot probabilities and a contrastive loss.',
  ],
  terms: [
    { t: 'Multimodal model', d: 'A model that takes in, makes or links more than one kind of data.' },
    { t: 'Patch token', d: 'One flattened image square, projected to a vector: (side ÷ patch)² of them.' },
    { t: 'CLS token', d: 'A special learned vector that ends up summarising the whole image.' },
    { t: 'Position embedding', d: 'A vector added to each patch so the model knows where it came from.' },
    { t: 'Contrastive learning', d: 'Pull matching pairs together and push non-matching pairs apart.' },
    { t: 'Zero-shot classification', d: 'Pick the class whose prompt is closest to the image, with no retraining.' },
  ],
  path: [
    { label: 'Lesson', min: 60 },
    { label: 'Setup, 1-3', min: 80 },
    { label: 'Tiny ViT, 4-5', min: 55 },
    { label: 'CLIP, 6', min: 30 },
    { label: 'Design, 7', min: 15 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Photo', sub: '224 × 224 × 3' },
      { label: 'Patches', sub: '14 × 14 = 196' },
      { label: 'Project', sub: '768 numbers each' },
      { label: 'Add CLS', sub: '197 tokens' },
      { label: 'Attention', sub: '38,809 scores' },
    ],
    caption: '16 × 16 patches: 197² = 38,809 scores, not 2.5 billion for pixel tokens.',
  },
};
