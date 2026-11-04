# Graphormer — Public Repository Transparency Documentation

## Overview

Graphormer is an open-source deep learning package for training and evaluating models for molecular modeling. This document covers the public [Microsoft Graphormer repository](https://github.com/microsoft/Graphormer), including its graph-based molecular property prediction code and the Distributional Graphormer (DiG) research examples. It does not describe the advanced pretrained versions available exclusively through Azure Quantum Elements.

The repository provides code, training/evaluation examples, and references to pretrained models. Consult the repository and linked papers for task-specific architecture and data descriptions.

### What can Graphormer do?

Graphormer uses Transformer-based graph representations with centrality, spatial, and edge encodings for graph-level prediction. The public toolkit supports molecular property prediction and 3D catalyst modeling, with interfaces for PyG, DGL, OGB, and OCP datasets and a Fairseq backbone. Available pretrained model references include PCQM4M and PCQM4Mv2.

Architecture and benchmark details are described in [Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets](https://arxiv.org/abs/2203.04810). The [DiG documentation](https://github.com/microsoft/Graphormer/tree/main/distributional_graphormer) describes equilibrium-distribution modeling and conformation sampling examples for proteins, protein–ligand systems, catalyst adsorption, and property-guided structure sampling. These are distinct task-specific models, not capabilities guaranteed by every Graphormer checkpoint.

### Intended uses

Graphormer is intended for molecular and materials science research: reproducing published results; comparing graph-learning methods; training, fine-tuning, and evaluating models on appropriately licensed data; and investigating molecular property prediction and catalyst-related tasks. DiG examples additionally support research on molecular conformations and equilibrium distributions.

The code is shared with researchers, students, molecular/materials scientists, and developers to facilitate reproducibility, independent scrutiny, and further research. Research into drug or materials discovery is an intended area of exploration, not a claim that predictions establish clinical utility, chemical safety, or production readiness.

Users should have expertise in the relevant scientific domain and machine learning, or work under qualified supervision. Treat outputs as research hypotheses or screening aids and independently validate them computationally or experimentally before acting on them.

### Out-of-scope uses

Graphormer is not a validated clinical, diagnostic, treatment-selection, toxicity-certification, or laboratory-safety system. Predicted properties, relaxed energies, and sampled structures do not establish safety, efficacy, stable binding, or experimental feasibility.

We do not recommend commercial or real-world reliance without further application-specific testing, validation, and development. The public research code and pretrained checkpoints are not a substitute for validated scientific workflows.

Do not assume that a checkpoint generalizes to unseen chemical families, elements, proteins, targets, or physical conditions, or that performance on one benchmark transfers to another task. Developers must assess accuracy, safety, data coverage, and downstream fairness concerns for each intended application.

Do not use unvalidated outputs as the sole basis for consequential or irreversible decisions, including experimental actions that could cause injury or decisions affecting an individual’s legal, financial, or life opportunities. Qualified human review and appropriate institutional safety processes remain necessary.

Harmful chemical or biological applications and attempts to bypass laboratory safety, ethics, or legal requirements are outside the intended research uses. These use recommendations do not change the MIT code license.

## How to get started

Start with the [public repository README](https://github.com/microsoft/Graphormer) and its linked documentation and examples. For the main Fairseq-based Graphormer code, the README recommends a containerized legacy environment. After obtaining the repository, initialize its submodules with `git submodule update --init --recursive`, then build the image with `docker build --tag graphormer:legacy .`.

Verify CPU imports and graph preprocessing with `docker run --rm graphormer:legacy python docker/verify_environment.py`. GPU execution requires the NVIDIA Container Toolkit; use the README’s GPU verification instructions. Keep datasets and checkpoints separate from the source checkout, select a task-matched example and checkpoint, and follow that task’s preprocessing and evaluation instructions. DiG examples have separate instructions in their respective subdirectories; do not assume the main container covers them.

## Evaluation

Published evaluations examine molecular property prediction on PCQM4M/PCQM4Mv2 and 3D catalyst modeling on OC20. DiG documents separate demonstrations involving protein conformation sampling, protein–ligand binding, catalyst adsorption, and property-guided sampling. Evaluation claims are specific to the corresponding model, task, data split, and experimental configuration.

See [Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets](https://arxiv.org/abs/2203.04810) for the molecular and catalyst experiments, and the [DiG documentation](https://github.com/microsoft/Graphormer/tree/main/distributional_graphormer) and its linked paper for distributional modeling. The results below summarize published research; they are not a new evaluation of the current repository.

### Evaluation methods

The molecular property experiments report mean absolute error (MAE; lower is better) on PCQM4M and PCQM4Mv2. These are quantum-chemistry benchmarks. The technical report evaluates 2D graph inputs without 3D geometric information for these property-prediction experiments and separately describes a 3D extension for OC20 catalyst–adsorbate modeling.

The technical report compares Graphormer architectural variants, including Pre-LN and Post-LN normalization, and discusses comparisons with message-passing graph neural networks. Comparisons must use matching dataset versions, splits, target definitions, and model/training configurations; results from different benchmarks are not interchangeable.

Evaluated variants include Graphormer Base and Large and a task-specific 3D extension. See the linked technical report for their architecture, training settings, and benchmark protocols. DiG uses separate task-specific configurations; the PCQM results below should not be attributed to DiG or to every checkpoint in the repository.

Results can change with checkpoint selection, model configuration, data coverage, preprocessing, random seed, and software environment. Public benchmark scores alone do not establish robustness on a downstream user’s chemistry or experimental conditions.

Review status: The sources reviewed for this document do not establish completed task-specific harms testing, a comprehensive security assessment, or RAI/DSB approval. No such testing, approval, or mitigation effectiveness is claimed here; the project team should confirm the appropriate evidence before publication.

### Evaluation results

As one concrete published comparison, Table 1 of the [March 2022 technical report (v1)](https://arxiv.org/pdf/2203.04810v1) reports validation MAE of 0.1193 for PostLNBase versus 0.1229 for PreLNBase on PCQM4M, and 0.0864 versus 0.0889, respectively, on PCQM4Mv2. These values describe that paper version’s experiments, not independently reproduced current-checkpoint scores. The report also describes improvements on 3D catalyst modeling; consult the paper’s task-specific results rather than treating this example as a universal performance claim.

## Limitations

Graphormer is research software. Benchmark accuracy does not establish readiness for commercial, clinical, or other consequential use. Independent application-specific testing and scientific validation are needed before real-world reliance.

Inputs are task-specific molecular graphs or structures, not natural-language prompts. Representation choices, supported atom/edge features, geometry conventions, and preprocessing constrain applicability. These graph models should not be treated as general-purpose language models.

Predicted properties and sampled conformations can be inaccurate, physically implausible, or misleading. A plausible structure or favorable predicted value is not evidence of experimental feasibility, safety, efficacy, or stable binding. Use qualified human oversight and independent computational or experimental checks.

Models can inherit errors, omissions, and chemical or structural coverage biases from their training data and labels. Performance can degrade on new molecular families, targets, elements, or conditions. Data and model suitability must be reassessed for each downstream task.

The main Graphormer code depends on a pinned Fairseq revision and a legacy environment. The README pins Python 3.9, PyTorch 1.9.1, CUDA 11.1 user-space libraries, and pip 23.3.2; it describes a reproducibility environment, not compatibility with current releases. Multi-node training is not covered by that setup.

Public code, datasets, and checkpoints are separate artifacts. Their download availability, provenance, permitted uses, and limitations should be verified individually. DiG data/checkpoint download links may be time-limited; do not assume that older instructions or links remain usable. No new checkpoint download or execution test was performed for this document.

A legacy container and pinned dependencies do not establish security. Use trusted artifact sources, review dependencies, and assess the risks of loading model and dataset files. No runtime safety filter, comprehensive security assurance, or mechanism to prevent harmful downstream repurposing is claimed. The public code can be copied and modified outside maintainers’ control.

## Best practices

For reliable research, match the checkpoint, dataset version, target, and preprocessing to the documented experiment. Record the source revision, environment, training settings, random seeds, and evaluation protocol. Confirm baseline reproduction before extending the method.

Recommended scientific and operational safeguards:

- Validate atom and bond features, atom ordering, coordinate conventions, units, missing values, and task-specific inputs. Test a small known example before running larger experiments.
- Evaluate on held-out data representative of the intended chemistry and use suitable out-of-distribution tests. Compare with appropriate baselines, inspect error patterns, and assess uncertainty rather than relying only on an aggregate benchmark score.
- Review selected predictions and sampled structures with qualified domain experts. Independently validate consequential conclusions using appropriate computation or experiments and follow institutional safety and ethics procedures.
- Obtain code, data, and checkpoints from trusted sources; verify provenance and integrity where possible. Use isolated, least-privilege environments and review dependency vulnerabilities. Do not treat containerization as a guarantee of safety.
- Do not publish credentials, confidential molecular structures, personal information, or sensitive data in examples, logs, checkpoints, or issue attachments. Share a minimal non-sensitive reproducer when reporting a problem.

Users are responsible for obtaining datasets legally and ethically, securing required rights and consent where applicable, and checking redistribution terms for each dataset and checkpoint. The code license should not be assumed to cover third-party data, model weights, or dependencies.

Assess privacy and confidentiality risks for custom datasets, particularly human-derived or donor-linked data and proprietary molecular structures. Minimize stored or logged sensitive information and review any external storage or services used in the workflow.

Users remain responsible for compliance with applicable data-protection requirements, institutional policies, and laboratory safety and ethics processes. Reassess the risks when adding new data, checkpoints, tasks, or downstream applications.

Describe AI-derived outputs as predictions or samples, not experimentally established findings. Report dataset and checkpoint provenance, applicable limitations, evaluation conditions, and negative results so downstream users can assess the evidence.

## License

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact

This research was conducted by members of [Microsoft Research](https://www.microsoft.com/en-us/research/). Bugs and feature requests can be reported through the repository’s GitHub Issues, as described in [SUPPORT.md](https://github.com/microsoft/Graphormer/blob/main/SUPPORT.md). Do not include confidential or sensitive information in public reports; use the repository’s security-reporting guidance for suspected vulnerabilities.
