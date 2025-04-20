# Predicting COVID-19 hospitalization from clinical narratives

## Introduction

This notebook demonstrates an end-to-end pipeline to predict whether a COVID-19 infection will lead to hospitalization. We use publicly available clinical notes data from Synthea Covid19 100K dataset.

## Motivation

What happens when you hand an LLM model a synthetic clinical narrative and ask it: *“Would this patient be hospitalized due to COVID-19?”*
I tested using zero-shot and few-shot prompting on a realistic-looking dataset and compared the responses against a logistic regression model.
I assumed gpt-4o would easily outperform the baseline. Spoiler: it didn’t.
Included assets: Jupyter notebook and prompt results as text files. 

## Analysis steps

1. Create narratives combining comorbidities, encounters, and medications.
2. Generate embeddings with sentence-transformers and trained a logistic regression (LogReg) model
3. Use Chain-of-Thought (CoT) to interpret predictions step by step. 
4. Run GPT-4o and o3-mini in zero-shot and few-shot modes on a test batch of 100 patients.
5. Compare performance (GPT vs. LogReg) using standard metrics

## Findings

* LLMs are not plug-and-play classifiers; they need careful prompt tuning.
* Few-shot prompting helps, but is sensitive to the balance and clarity of examples. 
* Model selection in combination with the right few-shot prompting strategy comes close to LogReg performance.
* Traditional models perform extremely well for binary prediction when features are well engineered.
* LLM evaluation needs more than accuracy: sensitivity to rare cases and repeatability.
