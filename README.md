# AI-Multimessenger-Example

A case study on using an AI assistant for gravitational-wave research:
reproducing the GW170817 detection and parameter estimation with
[HyperWave](https://github.com/asasli/HyperWave), then adding the
AT2017gfo light curve and a joint GW–EM analysis.

## Contents

- `Using AI for Real Research A GW170817 Case Study.pptx` — training slides:
  the conversation step by step, what the AI delivered, and an expert review
  of what it got right and wrong.
- `Outputs-chatGPT-Astra-10082026/` — the scripts and notebook the AI produced
  (pilot, PE driver, tutorial, joint GW–EM), unmodified.

## Caveat

The AI-generated code runs, but the full-PE sampler settings are far too
small to converge and the hyperbolic likelihood is never actually run.
See the review section of the slides before using any of it for science.
