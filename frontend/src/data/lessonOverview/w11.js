// Week 11 · Multimodal AI (backend/content/week-11.md)
export default {
  week: 11,
  tagline: 'Cut a picture into patches and treat them like words; CLIP links pictures to text.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'A Vision Transformer treats each image patch as a token.',
    'A ViT has no built-in image assumptions, so it needs more data.',
    'CLIP puts pictures and captions in one space for zero-shot labels.',
  ],
  outcomes: [
    'Explain what a multimodal model is and name three real applications.',
    'Calculate the number of patches and tokens for a given image and patch size.',
    'Explain how CLIP aligns images and text and how zero-shot classification works.',
  ],
  terms: [
    { t: 'Multimodal model', d: 'A model that works with more than one kind of data, like images and text.' },
    { t: 'Vision Transformer', d: 'A Transformer that reads an image as a sequence of patch tokens.' },
    { t: 'CLS token', d: 'A special learned vector that ends up summarising the whole image.' },
    { t: 'Inductive bias', d: 'An assumption built into a design, like a CNN’s nearby pixels belong together.' },
    { t: 'Contrastive learning', d: 'Pull matching pairs together and push non-matching pairs apart.' },
    { t: 'Zero-shot classification', d: 'Pick the class whose text is closest to the image, with no retraining.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Patches', min: 40 },
    { label: 'Tiny ViT', min: 60 },
    { label: 'CLIP', min: 80 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Image', sub: '3 × 224 × 224' },
      { label: 'Patches', sub: '196 × 768' },
      { label: 'Projection', sub: '768 to D' },
      { label: 'CLS + position', sub: '197 tokens' },
      { label: 'Encoder', sub: 'then class scores' },
    ],
    caption: 'A ViT with 16 × 16 patches: only the front end differs from a text Transformer.',
  },
};
