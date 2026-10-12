# Generalisation from Fourier Sparsity in Quantum Machine Learning

## Abstract

Variational quantum machine learning (QML) models can be represented by truncated Fourier series. While general results about how the size of the in principle accessible Fourier spectrum influence generalisation behaviour are known, the impact of structural features of the Fourier spectrum, in particular sparsity, have not been explored. The goal of this project is to explore via numerical simulation experiments whether Fourier sparsity or other properties of the Fourier spectrum can "explain" generalisation behaviour of QML models in the sense that they lead to quantities that correlate well with the true generalisation error. As such, the project will mainly consist in identifying relevant QML architectures based on the recent literature, training those models while enforcing regularisation to preferentially obtain solutions with sparse (or otherwise structured) Fourier spectra, and then studying the correlation between Fourier spectra quantities and observed generalisation error. Additionally, the numerical QML part of the project may be complemented by considerations of the role of Fourier spectra in classical ML or by rigorous proofs of generalisation bounds from Fourier sparsity.

## Problem Statement

Results about how sparsity affects generalisation behaviour are known* but have not yet been rigorously extended beyond theory. While theoretical results are known, we lack numerical explorations of how Fourier sparsity influences generalisation behaviour. In particular, given the current state of the literature, it is possible that the Fourier sparsity-based generalisation bounds proven so far are overly pessimistic and/or that other properties of the Fourier spectrum may hold more explanatory power for generalisation in QML models.

The aim of this project is to create and train various QML models so that experimental data can be obtained, which would be used to analyse how sparsity correlates to the generalisation gap.

* Reference paper discussing this
https://arxiv.org/abs/1611.03530 Understanding deep learning requires rethinking generalization
https://dl.acm.org/doi/10.1145/3446776 Understanding deep learning (still) requires rethinking generalization
https://www.nature.com/articles/s41467-024-45882-z Understanding quantum machine learning also requires rethinking generalization

Wording needs improving and this section needs extending


## Objectives

Split into must have, should have, could have

Complete preliminary research into:
- Quantum Computing
- QML
- Fourier representation of QML models
- Fourier spectra of QML models
- Generalisation gap analysis

Create a pipeline for the creation, training and evaluation of QML models using PennyLane

Use this pipeline to conduct experiments on relevant QML problems
Provide a pool of architectures and problems that could be used

Evaluate the generalisation gap

* https://arxiv.org/abs/2603.22964 A PAC-Bayesian approach to generalization for quantum models

## Methods & Methodology

Python: PennyLane, NumPy and an appropriate ML library

DCS Batch Compute system

Brief explanation of QML and Fourier representation, and Fourier sparsity

A QML pipeline that can be extended for use on relevant problems will be created

Define generalisation gap and why it is relevant, draw from classical literature

## Timetable

Gantt chart/table

Add buffer zones

Term 1:
- Research quantum computing
- Research QML
- Research Fourier representation of QML models
- Produce a basic QML pipeline using PennyLane in parallel with the above two points

Term 1-2
- Research Fourier spectra of QML models, including training and evaluation of generalisation gap

Term 2
- Use the pipeline to create models for "relevant" problems
- Evaluation & interpretation

## Resources & Risks

Quantum computing lecture notes
Other sources used will be referenced in the final report

PennyLane codebook - introductory topics in quantum computing with a parallel coding course

Version control using GitHub

PC, laptop and DCS Batch Compute - "While small scale training will take place on my own hardware during the implementation phase, I intend to use the DCS Batch Compute system to train the models that will be used for analysis...

Risks:
Small scale experiments may not be sufficiently expressive to support interpretation -> DCS Batch Compute to mitigate this
Delay in initial research as the topic is unfamiliar -> buffers in timetable

## Legal, Social, Ethical and Professional Issues & Considerations

No intention to collect personal data, but if public data is used its integrity should be confirmed

Goal to publish experimental data on GitHub

Statistical analysis will be conducted using conventional methods - give examples
