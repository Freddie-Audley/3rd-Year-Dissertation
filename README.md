# Project Specification

### Generalisation from Fourier Sparsity in Quantum Machine Learning

## Abstract

Variational quantum machine learning (QML) models can be represented by truncated Fourier series. While general results about how the size of the in principle accessible Fourier spectrum influence generalisation behaviour are known, the impact of structural features of the Fourier spectrum, in particular sparsity, have not been explored. The goal of this project is to explore via numerical simulation experiments whether Fourier sparsity or other properties of the Fourier spectrum can "explain" generalisation behaviour of QML models in the sense that they lead to quantities that correlate well with the true generalisation error. As such, the project will mainly consist in identifying relevant QML architectures based on the recent literature, training those models while enforcing regularisation to preferentially obtain solutions with sparse (or otherwise structured) Fourier spectra, and then studying the correlation between Fourier spectra quantities and observed generalisation error. Additionally, the numerical QML part of the project may be complemented by considerations of the role of Fourier spectra in classical ML or by rigorous proofs of generalisation bounds from Fourier sparsity.

## Problem Statement

Results about how sparsity affects generalisation behaviour are known* but have not yet been rigorously extended beyond theory. The aim of this project is to create and train various QML models so that experimental data can be obtained, which would be used to analyse how sparsity correlates to the generalisation gap.

* Reference paper discussing this

Wording needs improving and this section needs extending


## Objectives

Complete preliminary research into:
- Quantum Computing
- QML
- Fourier representation of QML models
- Fourier spectra of QML models

Create a pipeline for the creation, training and evaluation of QML models using PennyLane

Use this pipeline to conduct experiments on relevant QML problems

Evaluate and interpret the data gathered from these experiments

## Methods & Methodology

Python: PennyLane, NumPy and an appropriate ML library
- I presume Python is the best and perhaps only relevant choice?

Brief explanation of QML and Fourier representation, and Fourier sparsity

A QML pipeline that can be extended for use on relevant problems will be created

Define generalisation gap and why it is relevant

## Timetable

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

Hardware? I have a PC and laptop, and there are DCS machines. Is this a relevant sub-section?

Risks:
  Not really sure what would go here

## Legal, Social, Ethical and Professional Issues & Considerations

No intention to collect personal data, but if public data is used its integrity should be confirmed

Experimental data obtained can/will be published (?)
- Is GitHub an appropriate platform for this?

Statistical analysis will be conducted using conventional methods
