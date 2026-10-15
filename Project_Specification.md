# **Generalisation from Fourier Sparsity in Quantum Machine Learning** 

# **Project Specification** 

CS344 Discrete Mathematics Project 

Supervisor: Dr Matthias C. Caro 

### **Department of Computer Science** 

University of Warwick 

## **1 Problem Statement** 

Quantum machine learning (QML) investigates the use of quantum computational models for machine learning tasks. One common approach uses parameterised quantum circuits, in which classical input data are encoded into a quantum state, trainable quantum gates transform that state, and measurements produce the model’s output. The output of such a model can be expressed as 

$$ f_{\theta}(\mathbf{x}) = \langle 0|^{\otimes n} U^{\dagger}(\mathbf{x},\mathbf{\theta}) O U(\mathbf{x},\mathbf{\theta}) |0 \rangle^{\otimes n} $$

where $\mathbf{x}$ is the input data, $\mathbf{\theta}$ is the vector of trainable parameters, $U(\mathbf{x},\mathbf{\theta})$ represents the quantum circuit, and $O$ is the observable being measured. 

The circuit parameters are optimised using a classical training procedure to minimise a loss function, resulting in a hybrid quantum-classical learning model. 

An important property of parameterised quantum circuits is that their outputs can be expressed as finite Fourier series [1]. This representation allows the spectral properties of QML models, including Fourier sparsity, to be investigated in relation to their generalisation performance. 

While results about how sparsity affects generalisation behaviour are known [2], there has been limited numerical exploration of this relationship. In particular, given the current state of the literature, it is possible that the Fourier sparsity-based generalisation bounds proven so far are overly pessimistic, or that other properties of the Fourier spectrum may hold more explanatory power for generalisation in QML models. Existing generalisation bounds may not fully explain the empirical behaviour of QML models. For example, classical deep learning models can generalise despite having the ability to memorise random labels [3, 4], and similar phenomena have been investigated in quantum architectures [5]. These observations motivate investigating whether Fourier sparsity, or other properties of the Fourier spectrum, can better explain the generalisation behaviour of QML models. This project will use numerical simulations to investigate these relationships empirically. 

## **2 Objectives** 

### **Research** 

- Review the literature on QML, Fourier representations of parameterised quantum circuits, and generalisation in quantum models. 

### **Implementation** 

#### **Must** 

- Develop a reproducible pipeline to construct, train and evaluate parameterised quantum circuits on a supervised learning task 

- Support gradient-based optimisation and record the training configuration, model parameters and evaluation results 

- Implement a method to estimate or calculate the Fourier spectrum of the selected models 

- Train multiple models that exhibit variation in their Fourier sparsity 

#### **Should** 

- Allow the choice of optimiser and relevant hyperparameters 

- Extend the experiments to a second learning task or circuit architecture 

#### **Could** 

- Extend the pipeline to more complex datasets 

- Investigate alternative quantum circuit architectures 

### **Analysis** 

#### **Must** 

- Evaluate the training loss, test loss and generalisation gap for each model 

- Investigate the relationship between Fourier sparsity and the observed generalisation gap 

#### **Should** 

- Repeat experiments using different initialisations to assess the variability of results 

#### **Could** 

- Investigate generalisation under a distribution shift 

- Produce rigorous proofs of generalisation bounds from Fourier sparsity 

- Investigate whether other properties of the Fourier spectrum provide additional explanatory power for observed generalisation behaviour 

## **3 Methods & Methodology** 

### **3.1 Background** 

Quantum machine learning investigates the use of quantum computational models for machine learning tasks, including whether quantum resources can provide advantages over classical approaches for particular problems. In QML, the expectation value of a quantum neural network can be expressed as a finite Fourier series [1]: 

$$ f_{\theta}(x)=\sum_{\omega \in \Omega}c_{\omega}(\theta)e^{i \omega x} $$

A model is considered ”Fourier sparse” if the number of non-zero Fourier coefficients is small relative to the total number of accessible frequencies: 

$$ |\{\omega \in \Omega : | c_{\omega}(\theta) | \neq 0\}| \ll |\Omega| $$

For numerical experiments, a threshold $\varepsilon > 0$ will be used to distinguish significant coefficients from those treated as numerically zero. Thus, the resulting sparsity measure is 

$$ |\{\omega \in \Omega : | c_{\omega}(\theta) | > \varepsilon \}| \ll |\Omega| $$

Existing theoretical work has established generalisation bounds for parameterised quantum circuits in terms of properties including sparsity [2]. Intuitively, a model with a sparse Fourier spectrum has fewer active frequency components contributing to its output, restricting the complexity of the functions it can represent. Such restrictions may influence the model’s ability to fit training data and generalise to unseen data. However, existing generalisation bounds may not fully capture the empirical behaviour of QML models, motivating numerical investigation. 

A useful metric when evaluating the performance of a machine learning model is the generalisation gap [6]. This is empirically defined as 

$$ \Delta \mathcal{L} = \mathcal{L}\_{test} - \mathcal{L}\_{train} $$

where $\mathcal{L}\_{train}$ is the average value of a chosen loss function over the training dataset, and $\mathcal{L}\_{test}$ is the average value of the same loss function over a separate test dataset that is not used during training. A positive generalisation gap indicates that the model incurs a greater loss on the test data than on the training data, which may indicate overfitting. A gap close to zero indicates similar average losses on the two datasets, though does not necessarily imply good performance on either. 

Primary analysis will focus on in-distribution generalisation, with training and test examples sampled independently from the same underlying data distribution. Out-of-distribution evaluation may be considered as an extension. 

### **3.2 Implementation Details** 

QML models will be implemented in Python using PennyLane, alongside an appropriate classical machine learning (ML) library. Several candidate architectures will be considered, including hardware-efficient parameterised quantum circuits, data-reuploading circuits, and quantum convolutional neural networks (QCNNs). These architectures offer different circuit structures and data-encoding strategies that may influence the Fourier spectra of the resulting models. The initial implementation will prioritise relatively simple architectures to establish a functional training and evaluation pipeline, with more complex architectures considered if time and computational resources permit. 

Candidate learning tasks will include synthetic classification problems, synthetic regression tasks, and conventional datasets such as Iris [7] and MNIST [8]. Synthetic problems offer greater control over the data distribution and target function, making them particularly suitable for investigating the relationship between Fourier spectral properties and generalisation. Conventional datasets may be considered to assess whether the observed relationships extend to more realistic learning tasks. 

The final selection of architectures and learning tasks will be guided by their suitability for Fourier spectral analysis, the feasibility of training the models, and the ability to obtain meaningful variation in their spectral properties. The initial experiments will establish the pipeline using a relatively simple task before progressing to more complex configurations where appropriate. 

The DCS Batch Compute<sup>1</sup> system will be used to facilitate model training, allowing experiments to run for extended periods without relying on local hardware. 

## **5 Resources & Risks** 

### **5.1 Resources** 

Preliminary research has included studying the fundamentals of QC using sections 1 and 2 of the Quantum Information and Computation lecture notes by Richard Jozsa [9], alongside completing the first three modules of the PennyLane Codebook. Further materials consulted throughout the project will be referenced in subsequent reports. 

GitHub is being used for version control, providing backups in the event of data loss and a record of project development through its commit history. 

The available hardware includes a PC and a laptop, although neither is suited to QML training beyond the implementation of the pipeline. The DCS Batch Compute system<sup>1</sup> will be used to train the larger scale models from which data will be collected. 

### **5.2 Risks** 

Small scale experiments may not be sufficiently expressive to support interpretation. Access to the DCS Batch Compute<sup>1</sup> system can help mitigate this limitation by providing additional computational resources for larger experiments, although the feasible scale will depend on the complexity of the circuits and the simulation method used. 

Given the specialised nature of the project, additional time has been allocated for preliminary research. The timetable also incorporates buffers to accommodate external academic commitments and other unforeseen circumstances. 

> 1 `https://warwick.ac.uk/fac/sci/dcs/admissions/postgraduatetaughtdegrees/facilities_research/` ”High-performance computing” 

## **6 Legal, Social, Ethical and Professional Issues & Considerations** 

There is no intention to collect or process primary personal data during this project. Any public datasets used will be handled in accordance with any license conditions they may have. To ensure scientific rigour, the integrity of such datasets will be validated prior to use. 

Statistical analysis will be completed using established methods [6] to ensure that the conclusions drawn from this research are reproducible and consistent with the underlying data. One of the goals of this project is to publish the final results, source code and methodology to a public GitHub repository. This commitment reinforces the need to follow accepted professional standards. If this is done, an open-source license will be obtained, such as MIT or Apache 2.0, to ensure clear terms of reuse for the wider community. 

## **References** 

- [1] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. The effect of data encoding on the expressive power of variational quantum machine learning models. _arXiv preprint arXiv:2008.08605_ , 2021. URL `https://arxiv.org/abs/2008.08605` . 

- [2] Matthias C. Caro, Elies Gil-Fuster, Johannes Jakob Meyer, Jens Eisert, and Ryan Sweke. Encoding-dependent generalization bounds for parametrized quantum circuits. _Quantum_ , 5:582, 2021. doi: 10.22331/q-2021-11-17-582. URL `https://quantumjournal.org/papers/q-2021-11-17-582/` . 

- [3] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. _arXiv preprint arXiv:1611.03530_ , 2016. URL `https://arxiv.org/abs/1611.03530` . 

- [4] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. _Communications of the ACM_ , 64(3): 107–115, 2021. doi: 10.1145/3446776. URL `https://doi.org/ 10.1145/3446776` . 

- [5] Elies Gil-Fuster, Jens Eisert, and Carlos Bravo-Prieto. Understanding quantum machine learning also requires rethinking generalization. _Nature Communications_ , 15(1):2277, 2024. URL `https://www.nature.com/articles/s41467-024-45882-z` . 

- [6] Pablo Rodriguez-Grasa, Matthias C. Caro, Jens Eisert, Elies GilFuster, Franz J. Schreiber, and Carlos Bravo-Prieto. A PACbayesian approach to generalization for quantum models. _arXiv preprint arXiv:2603.22964_ , 2026. URL `https://arxiv.org/ abs/2603.22964` . 

- [7] R. A. FISHER. The use of multiple measurements in taxonomic problems. _Annals of Eugenics_ , 7(2):179–188, 1936. doi: https://doi.org/10.1111/j.1469-1809.1936.tb02137.x. URL `https://onlinelibrary.wiley.com/doi/abs/10.1111/ j.1469-1809.1936.tb02137.x` . 

- [8] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278–2324, 1998. doi: 10.1109/ 5.726791. URL `http://yann.lecun.com/exdb/publis/pdf/ lecun-98.pdf` . 

- [9] Richard Jozsa. Quantum information and computation lecture notes, 2019. URL `https://www.qi.damtp.cam.ac.uk/files/ PartIIIQC/Part%202%20QIC%20lecturenotes.pdf` . Accessed: August - September, 2026. 
