# VLM Counterfactual Consistency

This project studies whether vision-language models truly reason about visual content or rely on superficial correlations in Visual Question Answering (VQA).

## Goal
We measure and improve causal consistency in VQA through counterfactual interventions such as:
- Negation
- Attribute swaps
- Entailment
- Spatial perturbations

## Models
- LLaVA-1.5
- InstructBLIP

## Datasets
- GQA
- VQA v2

## Main Components
- Counterfactual question generation
- VLM benchmarking
- Consistency Score evaluation
- Consistency-aware training with augmentation and pairwise loss

## Planned Repository Layout
- `src/counterfactual/`: generate related counterfactual question families
- `src/models/`: run inference with VLMs
- `src/evaluation/`: compute consistency and accuracy metrics
- `src/training/`: fine-tuning and loss implementations
- `data/`: raw and processed datasets
- `results/`: metrics, predictions, and analysis outputs