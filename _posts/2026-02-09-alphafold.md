---
title: "AlphaFold — A Deep Learning Breakthrough in Protein Structure Prediction"
layout: single
date: 2026-02-12
author_profile: true
math: true
categories: [AlphaFold]
tags: [Protein Folding, Evoformer, Geometry, MSA, CASP]
---

# AlphaFold 2 — A Deep Learning Breakthrough

This post follows the structure of my seminar presentation and walks step-by-step from biological foundations to the architecture and results of AlphaFold2 and AlphaFold3.

---

# 1. Biomolecular Structure — Proteins

Proteins are large biomolecules composed of one or more amino acid chains.  
They perform nearly every task in cellular life.

**Key principle:**

> The 3D shape of a protein determines its function.

Each amino acid consists of:

- Alpha carbon \(C_\alpha\)
- Amino group (NH₂)
- Carboxyl group (COOH)
- Side chain \(R\)

There are **20 standard amino acids**, each defined by its side chain.

![Amino Acid Structure](/assets/images/amino_acid.png)

---

# 2. The Protein Folding Problem

Proteins have four structural levels:

1. **Primary** — amino acid sequence  
2. **Secondary** — α-helices and β-sheets  
3. **Tertiary** — full 3D fold  
4. **Quaternary** — multi-chain complexes  

![Protein Structure Levels](/assets/images/protein_levels.png)

Folding transforms a linear sequence into a functional 3D structure.

### Levinthal’s Paradox

If each residue had only a few conformations, the number of possible structures would be astronomical:

\[
\sim 10^{143}
\]

Yet proteins fold within milliseconds.

This implies:

> Folding is highly constrained and guided — not random search.

---

# 3. Approximation Methods

Two approaches exist:

## Experimental Methods

- X-ray crystallography  
- NMR spectroscopy  
- Cryo-EM  

These measure final structures but are expensive and slow.

## Computational Methods

Given a sequence:

\[
x = (x_1, \dots, x_n)
\]

predict coordinates:

\[
f_\theta(x) \in \mathbb{R}^{n \times 3}
\]

This is the protein structure prediction problem.

---

# 4. Quality Assessment

Proteins are typically represented as:

- Point clouds (3D atomic coordinates)
- Density volumes

A naive loss would be RMSD:

\[
L(p_1, p_2) = \sum_{i=1}^{n_C} \| p_1^{(i)} - p_2^{(i)} \|_2
\]

But proteins have no canonical orientation.

Therefore, CASP uses **GDT (Global Distance Test)**:

- Computes percentage of residues within threshold λ
- More robust to outliers

CASP (since 1994) benchmarks structure prediction every two years.

---

# 5. AlphaFold2 — Full Pipeline

![AlphaFold Pipeline](/assets/images/alphafold_pipeline.png)

The pipeline consists of:

1. Sequence input
2. MSA search
3. Template search (PDB)
4. Evoformer
5. Structure Module
6. Recycling

Importantly:

> AlphaFold does NOT simulate folding physics.  
> It learns structural constraints from data.

---

# 6. Input Feature Extraction

## Multiple Sequence Alignment (MSA)

The model searches genetic databases (e.g., BFD) to construct an MSA.

Why?

Evolution preserves structure.

If residue \(i\) mutates and residue \(j\) co-mutates:

\[
\text{Cov}(i,j) > 0 \Rightarrow \text{spatial proximity}
\]

This is **coevolution**.

MSA tensor:

\[
M \in \mathbb{R}^{N_{seq} \times N_{res} \times d}
\]

---

## Pair Representation

Residue-pair features:

\[
P \in \mathbb{R}^{N_{res} \times N_{res} \times d}
\]

Computed via outer-product operations over MSA embeddings.

These encode potential geometric relationships.

---

# 7. Evoformer

Core innovation.

Evoformer jointly processes:

- MSA representation
- Pair representation

It consists of 48 blocks.

## MSA Transformer

- Row attention → residue interactions  
- Column attention → evolutionary coupling  

## Pair Transformer

Triangle attention:

\[
P_{ij} \leftarrow \text{Attention}(P_{ik}, P_{kj})
\]

This enforces geometric consistency:

If \(i\) close to \(k\)  
and \(k\) close to \(j\),  
then \(i\) must be constrained relative to \(j\).

This approximates triangle inequality constraints.

---

# 8. Structure Module

After Evoformer refinement:

Residues are treated as rigid frames:

\[
(R_i, t_i) \in SO(3) \times \mathbb{R}^3
\]

Each residue is represented as a triangle defined by backbone atoms (N, Cα, C).

## Invariant Point Attention (IPA)

Attention operates directly in 3D space.

Crucially:

\[
f(Rx + t) = R f(x) + t
\]

This guarantees rotation and translation equivariance.

IPA iteratively updates:

\[
(R_i, t_i)
\]

Torsion angles determine final atomic coordinates.

---

# 9. Training and Loss

Main structural loss:

## Frame Aligned Point Error (FAPE)

Instead of global RMSD:

\[
\text{FAPE} = \sum_i \| F_i^{-1}(x_j) - F_i^{*-1}(x_j^*) \|
\]

Advantages:

- Local frame alignment
- Preserves chirality
- Stabilizes training

Additional losses:

- Distogram loss (distance matrix prediction)
- Auxiliary torsion angle losses

Training uses gradient descent:

\[
w_{t+1} = w_t - \eta \nabla L(w_t)
\]

---

# 10. Datasets

AlphaFold2 uses:

## Sequence Database
- BFD (2.2 billion sequences)

## Structure Database
- Protein Data Bank (PDB)

Templates guide prediction but are not required.

---

# 11. Results — CASP14

| CASP | Year | Top GDT_TS |
|------|------|------------|
| CASP13 | 2018 | 65 |
| CASP14 | 2020 | **92.4** |

Median RMSD ≈ 1 Å.

Performance tripled over previous methods.

Generalization:

- 3,144 unseen chains
- Median RMSD = 1.46 Å

AlphaFold2 essentially solved single-chain structure prediction.

---

# 12. Discussion

### Strengths

- Near experimental accuracy
- Strong geometric inductive bias
- End-to-end differentiable
- Recycling refinement

### Limitations

- Static structures (no dynamics)
- Struggles with complexes
- Depends on rich MSA data

These limitations motivated AlphaFold3.

---

# Conclusion

AlphaFold2 demonstrates that:

- Evolution encodes geometric information
- Attention mechanisms can infer structure
- Geometric deep learning can solve long-standing scientific problems

It represents a milestone in AI for science.
