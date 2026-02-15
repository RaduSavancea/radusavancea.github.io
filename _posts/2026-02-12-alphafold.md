---
title: "AlphaFold — A Deep Learning Breakthrough in Protein Structure Prediction"
math: true
layout: single
date: 2026-02-12
author_profile: true
classes: wide
categories: [AlphaFold]
tags: [Protein Folding, Evoformer, Geometry, MSA, CASP]
---

## Introduction

This blog post explores the fundamental concepts of protein folding and the key architectural components behind AlphaFold 2 and AlphaFold 3.

AlphaFold represents a pivotal achievement in biological AI, particularly in **3D protein structure prediction**. The work led to the 2024 Nobel Prize in Chemistry being awarded to **Demis Hassabis**, leader of the AlphaFold project, and **John Jumper**, for groundbreaking contributions to computational biology and protein folding prediction.

We begin with the biological foundations before moving towards the deep learning architectures that enabled this breakthrough. While this post primarily focuses on protein structures, it is important to note that AlphaFold 3 extends beyond isolated proteins and is capable of modeling full biomolecular complexes, including interactions with DNA, RNA, ligands, and ions.

---

## Contents

1. [Proteins](#proteins)  
2. [Protein Folding Problem](#protein-folding-problem)  
3. [Previous Work and Quality Assessments](#previous-work-and-quality-assessments)  
4. AlphaFold 2  
5. AlphaFold 3  
6. Experiments and Results  
7. Concluding Thoughts  

---

# Proteins

Proteins are essential biomolecules and macromolecules present in every cell of all living organisms. They are among the most abundant organic molecules in biological systems and perform a vast range of functions necessary for life.

Proteins are responsible for nearly every task of cellular life. They act as enzymes, structural components, transport molecules, signaling agents, immune defenders, and molecular motors. Because structure determines function, understanding protein structure is central to understanding biology itself.

The **structure of a protein is directly linked to its function**, making structural biology fundamental for studying physiological processes such as human health, disease, enzyme catalysis, immune response, and drug interactions. Structural insight enables rational drug design and targeted therapeutic development.

Although the amino acid sequence encodes the primary structural information of a protein, the final three-dimensional conformation is not determined by sequence alone. The cellular environment also plays a crucial role. Factors such as pH, temperature, ionic concentration, molecular crowding, and the presence of molecular chaperones can significantly influence folding pathways and stability.

---

## Amino Acids — The Building Blocks

Proteins are composed of one or more **chains of amino acids**. Each amino acid is an organic molecule built around a central carbon atom, known as the C-α atom, which is bonded to four distinct groups: an amino group (NH₂), a carboxyl group (COOH), a hydrogen atom, and a variable side chain, often referred to as the R-group.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/amino_acid.png' | relative_url }}" alt="Amino Acid Structure" width="500">
  <figcaption>
    <strong>Figure 1:</strong> General structure of an amino acid showing the central C-α atom, amino group, carboxyl group, hydrogen atom, and variable side chain.
    <br>
    <em>Source:</em> Adapted from GeeksforGeeks, <em>Amino Acids: Definition, Structure, Properties, Classification</em> [1].
  </figcaption>
</figure>

The side chain determines the chemical and physical properties of the amino acid. In nature, there are **20 standard amino acids**, each characterized by a unique side chain and represented by a one-letter code. Once incorporated into a protein chain, an amino acid is referred to as a **residue**.

When amino acids connect, they form peptide bonds through a dehydration synthesis reaction. The carboxyl group of one amino acid reacts with the amino group of another, releasing a water molecule and forming a stable covalent linkage.

A protein is therefore fundamentally a linear sequence of residues, and this sequence encodes the information required for the protein to adopt its final three-dimensional structure.

---

# Protein Folding Problem

Proteins do not function as linear chains. Instead, they fold into highly specific three-dimensional structures that determine their biological activity.

Protein structure is commonly described at four hierarchical levels. The primary structure corresponds to the amino acid sequence. The secondary structure refers to local motifs such as α-helices and β-sheets formed by hydrogen bonding along the backbone. The tertiary structure describes the complete three-dimensional conformation of a single polypeptide chain. The quaternary structure refers to the arrangement of multiple interacting chains (subunits) into a functional complex.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/protein_levels.png' | relative_url }}" alt="Protein Structure Levels" width="650">
  <figcaption>
    <strong>Figure 2:</strong> Hierarchical organization of protein structure — from primary sequence to quaternary assembly.
    <br>
    <em>Source:</em> Biology4Alevel, <em>Protein Structure</em> [2].
  </figcaption>
</figure>

Protein folding can be understood as an energy-driven physical process in which a polypeptide chain transitions from a random coil to a thermodynamically stable, low-energy three-dimensional conformation. This transformation is driven by hydrogen bonding, hydrophobic interactions, electrostatic forces, and van der Waals interactions.

Correct folding is essential for biological function. Misfolded proteins can become inactive or toxic and are associated with numerous diseases, including Alzheimer’s disease, cancer, neurodegenerative disorders, and certain allergies.

Understanding protein folding is therefore central to medicine, drug discovery, and biotechnology. Structural insight enables the rational design of pharmaceuticals and helps interpret the functional impact of genetic mutations and variations.

---

## Why Is Protein Folding Important?

Protein misfolding is linked to numerous diseases. For example, the accumulation of misfolded amyloid proteins plays a central role in Alzheimer’s disease, while structural abnormalities in regulatory proteins are associated with various forms of cancer. When proteins fail to adopt their correct three-dimensional conformation, they can lose functionality or become toxic to cells.

Understanding protein folding is therefore central to several major scientific and industrial fields. In **medicine**, structural insight enables the identification of disease mechanisms at the molecular level. In **drug discovery**, knowledge of a protein’s three-dimensional structure allows researchers to design molecules that precisely bind to active or regulatory sites. In **biotechnology**, engineered proteins with specific structural properties can be developed for industrial enzymes, synthetic biology applications, or targeted therapies.

Improved insight into folding mechanisms directly supports the development of new pharmaceuticals. By understanding how proteins fold and interact, researchers can predict binding interfaces, stabilize unstable proteins, and design inhibitors that block harmful interactions.

Moreover, protein structures help interpret genetic variation and mutations. Many mutations do not simply change a sequence, they alter structural stability or binding behavior. Structure prediction therefore helps explain why certain mutations are pathogenic while others are benign.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Corona.JPG' | relative_url }}" alt="Predicted protein structure example" width="500">
  <figcaption><strong>Figure 3:</strong> Example of a predicted protein structure (Q8W3K0 — plant disease resistance protein) from the AlphaFold Protein Structure Database.</figcaption>
</figure>

The ability to computationally predict accurate protein structures at scale fundamentally changes how we approach biology, medicine, and molecular engineering.


---

## Levinthal’s Paradox

The complexity of protein folding becomes evident when considering the astronomical number of possible conformations a protein chain could theoretically adopt.

In 1969, Cyrus Levinthal estimated that the number of potential conformations for a typical protein could be on the order of

$$
10^{300}
$$

If a protein were to explore all these conformations randomly, reaching its native structure would take longer than the age of the universe. Yet, in reality, many proteins fold within milliseconds to seconds.

This apparent contradiction is known as **Levinthal’s Paradox**. It demonstrates that protein folding is not a random search through conformational space, but a highly constrained and guided physical process. Understanding this efficiency has been one of the central challenges of molecular biology and it is precisely this challenge that makes AlphaFold transformative.

---

# Previous Work and Quality Assessments

The effort to determine protein structures dates back to the 1950s and marks the beginning of modern structural biology. In 1955, Frederick Sanger determined the amino acid sequence of insulin, revealing the primary structure of a protein and earning the Nobel Prize in Chemistry in 1958.

Two major approaches have been developed to determine protein structures: experimental methods and computational methods.

Experimental techniques such as **X-ray crystallography** expose crystallized proteins to X-rays and reconstruct electron density maps from diffraction patterns. A typical X-ray pipeline involves protein crystallization — a particularly challenging step for membrane proteins due to their low solubility — followed by X-ray diffraction and reconstruction of electron density maps. One of the central technical challenges in this process is the *phase problem*, which requires specialized methods to resolve. Even after obtaining an electron density map, reconstructing the final 3D atomic model involves a degree of interpretation; especially at lower resolutions, two experts may produce slightly different structural models from the same data. While highly accurate, this method captures only the final folded structure, requires substantial laboratory effort, and is expensive — often costing on the order of \$100,000–\$1,000,000 per structure [3].


**Nuclear magnetic resonance (NMR)** spectroscopy measures protein structures in solution and can capture dynamics on very small time scales, which is important because folding and conformational transitions can occur on the order of 50–3000 s⁻¹. However, NMR is generally limited to smaller proteins and can be experimentally demanding.

In contrast, reading protein sequences is comparatively inexpensive. Amino acid sequences can be obtained directly via mass spectrometry (roughly \$100 per sample) or indirectly through genome or exome sequencing (approximately \$100–\$1000 per sequence). This creates a significant asymmetry in biology: sequence data is abundant and cheap, whereas experimentally determined structures are scarce and costly.

Computational approaches aim to predict protein structure directly from the amino acid sequence. These methods incorporate evolutionary information, structural priors, physicochemical constraints, and increasingly, machine learning models trained on large structural databases. The growing gap between inexpensive sequence acquisition and expensive structural determination is precisely the bottleneck that modern computational approaches, such as AlphaFold, attempt to overcome.

---

## Quality Assessment

Protein structures are typically represented either as a point cloud of atomic coordinates or as a three-dimensional density volume. A common evaluation metric is the **Root Mean Square Error (RMSE)** over predicted and ground-truth C-α atom positions:

$$
L(p_1, p_2) = \sum_{i=1}^{n_C} \| p_1^{(i)} - p_2^{(i)} \|^2
$$

However, because protein structures have no canonical orientation, RMSE is only meaningful after spatial alignment.

To enable fair comparison between prediction methods, the community established the **Critical Assessment of Structure Prediction (CASP)** in 1994. CASP is held every two years and evaluates models on newly solved experimental structures. Instead of relying solely on RMSE, it commonly uses the **Global Distance Test (GDT)** score, which is more robust to outliers and reports performance on a scale from 0 to 100.

Additional metrics such as **lDDT (local Distance Difference Test)** and **DockQ** are used to assess local structural accuracy and the quality of predicted molecular interfaces.

In 2020 and 2022, AlphaFold dramatically outperformed previous approaches in CASP, nearly doubling the GDT accuracy of earlier systems. This marked a turning point in computational biology and set the stage for the deep learning revolution in protein structure prediction.

---

*The next sections will explore the architecture of AlphaFold 2, including MSA feature extraction, the Evoformer, and the structure module, followed by the advances introduced in AlphaFold 3.*


# AlphaFold 2

AlphaFold 2 fundamentally redefined protein structure prediction by combining evolutionary biology, geometric deep learning, and transformer-based architectures into a single end-to-end differentiable system.

At its core lies a crucial biological insight: **evolution contains structural information**.

Many mutations are tolerated without disrupting overall fold, allowing sequence divergence while preserving structural constraints. As a result, protein sequences can diverge substantially across distant species (often by more than 70%), while their three-dimensional structures remain remarkably conserved [3]. In other words:

> Protein structure is far more conserved than protein sequence.

This observation is central. If structure changes slowly while sequences mutate, then comparing homologous sequences across species allows us to infer structural constraints indirectly.

Rather than predicting structure directly from physics-based simulation, AlphaFold 2 learns a representation of residue–residue relationships extracted from evolutionary variation and then converts this relational structure into three-dimensional geometry through an equivariant structure module.

Conceptually, the system operates in two major stages:

1. **Relational reasoning in representation space (Evoformer)**
2. **Geometric realization in 3D space (Structure Module)**

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Fold2_Archi.PNG' | relative_url }}" width="750">
  <figcaption><strong>Figure 1:</strong> Complete AlphaFold 2 pipeline (Jumper et al., 2021).</figcaption>
</figure>

---

## Input Feature Extraction

The input to AlphaFold 2 is a single amino acid sequence of length $N$. From this sequence, the model constructs two primary feature tensors:

- A **Multiple Sequence Alignment (MSA)** representation  
- A **Pair representation**

These two representations form the foundation of the entire architecture.

---

### Multiple Sequence Alignment (MSA)

Starting from the target sequence, AlphaFold performs a large-scale database search to retrieve homologous sequences from diverse species. These sequences are arranged in a two-dimensional table called a **Multiple Sequence Alignment (MSA)**, where:

- Each row corresponds to a homologous protein sequence  
- Each column corresponds to a specific residue position  

By aligning homologues in this way, corresponding residues across species are placed in the same column.

This alignment encodes powerful evolutionary signals:

- **Conservation** of a column often implies structural or functional importance (e.g., catalytic residues, ligand binding sites).
- **Co-evolution** between two columns suggests structural coupling — if residue $i$ mutates and residue $j$ consistently mutates in response, the two residues are likely interacting in 3D space.

Intuitively, if two residues participate in a bonding mechanism, mutating one without compensating changes would destabilize the protein. Therefore, correlated mutations preserve structural integrity [4].

The alignment is embedded into a tensor:

$$
\mathrm{MSA} \in \mathbb{R}^{N_{\text{seq}} \times N \times c_m}
$$

where:

- $N_{\text{seq}}$ is the number of aligned sequences  
- $N$ is the number of residues  
- $c_m$ is the embedding dimension  

This tensor does not explicitly encode distances. Instead, it encodes **evolutionary constraints**, from which geometric structure can be inferred.

Evolution therefore acts as indirect supervision for 3D structure prediction.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/MSA.PNG' | relative_url }}" width="650">
  <figcaption><strong>Figure 2:</strong> Multiple Sequence Alignment capturing evolutionary conservation and co-evolution signals.</figcaption>
</figure>

---

### Pair Representation

While the MSA captures evolutionary variation, protein structure itself is fundamentally about **relationships between residues**.

To explicitly reason about residue–residue interactions, AlphaFold maintains a second tensor:

$$
\mathrm{Pair} \in \mathbb{R}^{N \times N \times c_z}
$$

Each element $\mathrm{Pair}[i,j]$ is a learned feature vector encoding the model’s current belief about how residues $i$ and $j$ relate geometrically.

This tensor can be interpreted as a **complete graph over residues**, where:

- Nodes correspond to residues  
- Edges store relational features  

Unlike classical contact maps, this representation does not explicitly store distances. Instead, it stores a high-dimensional latent encoding capable of supporting geometric reasoning.

Together, the MSA tensor and the Pair tensor provide two complementary perspectives:

- The MSA captures evolutionary constraints across species.
- The Pair representation captures geometric relationships within a single protein.

The Evoformer iteratively refines both representations, allowing evolutionary signals to shape geometric reasoning before any 3D coordinates are predicted.


---

## The Evoformer: Iterative Representation Refinement

The Evoformer is the core reasoning engine of AlphaFold 2. It consists of a deep stack of blocks that iteratively refine both the MSA and pair representations.

Within each block:

- Row and column attention refine MSA features  
- Triangle updates refine pairwise relationships  
- Information flows bidirectionally between the two representations  

Stacking many Evoformer blocks allows the network to progressively refine its internal hypothesis of the protein fold. The relational representation becomes increasingly geometric and globally consistent with depth.

Conceptually, the Evoformer behaves like a learned constraint solver operating over a dense residue graph enriched with evolutionary evidence.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/EvoFormer.PNG' | relative_url }}" width="700">
  <figcaption><strong>Figure 3:</strong> Evoformer architecture showing MSA attention and pair updates.</figcaption>
</figure>

---

### Axial Attention in the MSA

AlphaFold applies **axial attention**, meaning attention is performed separately along each dimension of the MSA tensor.

#### Row Attention (Across Residues)

For a fixed sequence index \( s \):

$$
\mathrm{MSA}(s, :, :) \in \mathbb{R}^{N \times c_m}
$$

Queries, keys, and values are computed as:

$$
Q_{s,i} = W_Q x_{s,i}, \quad
K_{s,j} = W_K x_{s,j}, \quad
V_{s,j} = W_V x_{s,j}
$$

Attention weights:

$$
\alpha_{s,i,j} =
\mathrm{softmax}_j
\left(
\frac{Q_{s,i} \cdot K_{s,j}}{\sqrt{d}} + b_{i,j}
\right)
$$

Here, \( b_{i,j} \) is derived from the pair representation, allowing relational geometry to bias MSA attention.

The updated embedding is:

$$
x'_{s,i} = \sum_{j=1}^{N} \alpha_{s,i,j} V_{s,j}
$$

Row attention captures **long-range intra-sequence dependencies**, modeling how distant residues influence one another.

---

### Triangle Updates: Enforcing Geometric Consistency

Pairwise relationships alone cannot guarantee global consistency in 3D space. If residue \( i \) is close to \( k \), and \( k \) is close to \( j \), then \( i \) cannot be arbitrarily far from \( j \).

AlphaFold enforces this consistency through:

- **Triangle attention**
- **Triangle multiplicative updates**

For a fixed pair \( (i,j) \), triangle attention aggregates information over all third residues \( k \), allowing geometric constraints to propagate transitively through the residue graph.

Triangle multiplicative updates perform learned multiplicative interactions across residue triplets, resembling message passing in graph neural networks.

Importantly, no explicit triangle inequality is imposed in the loss. Instead, geometric consistency emerges because inconsistent relational structures increase downstream structural error.

---

## Structure Module: From Representation to Geometry

After the Evoformer produces a consistent relational representation, AlphaFold translates this abstract structure into explicit 3D coordinates.

The Structure Module is designed to be **SE(3)-equivariant**, meaning:

- Rotating or translating the entire protein does not change internal decisions  
- Predictions transform consistently under rigid-body motion  

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Structure_Module.PNG' | relative_url }}" width="650">
  <figcaption><strong>Figure 4:</strong> Structure module translating relational embeddings into 3D frames.</figcaption>
</figure>


---

### Rigid-Body Representation of Residues

Each residue \( i \) is assigned a rigid transformation:

$$
T_i = (R_i, t_i)
$$

where:

- \( R_i \in SO(3) \) is a rotation matrix  
- \( t_i \in \mathbb{R}^3 \) is a translation vector  

Initially:

$$
R_i = I, \quad t_i = 0
$$

The network must learn to position and orient every residue in 3D space.

---

### Invariant Point Attention (IPA)

Invariant Point Attention augments standard attention with geometric awareness.

Each residue predicts learned local points:

$$
p_i^{(m)} \in \mathbb{R}^3
$$

These are transformed into global coordinates:

$$
\hat{p}_i^{(m)} = R_i p_i^{(m)} + t_i
$$

The attention score between residues includes both feature similarity and geometric proximity:

$$
\mathrm{score}(i,j) =
Q_i \cdot K_j
-
\sum_m
\|\hat{p}_i^{(m)} - \hat{p}_j^{(m)}\|^2
$$

Because Euclidean distance is invariant under rotation and translation, attention decisions depend only on relative geometry, ensuring SE(3) equivariance.

---

### Frame Updates

The network predicts incremental rigid-body transformations:

$$
\Delta R_i, \quad \Delta t_i
$$

Frames are updated as:

$$
R_i \leftarrow \Delta R_i R_i
$$

$$
t_i \leftarrow \Delta R_i t_i + \Delta t_i
$$

This iterative refinement (typically ~8 steps) resembles a learned geometric optimization procedure:

- Early iterations establish coarse layout  
- Later iterations refine spatial precision  

---

### Backbone and Side-Chain Construction

Once backbone frames are refined, atomic coordinates are placed deterministically.

For atom \( a \) in residue \( i \):

$$
x_{i,a} =
R_i \hat{x}_{i,a}(\chi_i)
+
t_i
$$

where:

- \( \hat{x}_{i,a}(\chi_i) \) are canonical local coordinates  
- \( \chi_i \) are predicted torsion angles  

Torsion angles are predicted via sine and cosine representations:

$$
(\sin \chi_i, \cos \chi_i)
$$

This avoids discontinuities due to angular periodicity.

Bond lengths and bond angles remain fixed, ensuring chemically valid geometry.

---

## Conceptual Summary

AlphaFold 2 separates protein folding into two conceptual phases:

1. **Learning relational constraints in representation space (Evoformer)**
2. **Converting relational structure into 3D geometry (Structure Module)**
3. **Refining geometry through SE(3)-equivariant updates**

This separation between high-dimensional relational reasoning and geometric instantiation is one of the central architectural innovations that enabled AlphaFold 2 to achieve near-experimental accuracy in CASP.

It marked the moment when deep learning transformed structural biology from a decades-long grand challenge into a tractable computational problem.


# AlphaFold 3 — From Folding to Generative Molecular Modeling

AlphaFold 3 represents a fundamental architectural shift compared to AlphaFold 2.  
While AlphaFold 2 solved protein folding through deterministic equivariant refinement of residue frames, AlphaFold 3 generalizes structure prediction into a unified, generative model of full biomolecular systems.

Instead of predicting a single rigidly refined protein structure, AlphaFold 3 models:

- Proteins  
- DNA and RNA  
- Small-molecule ligands  
- Metal ions  
- Covalent modifications  

within one shared architecture. Folding and binding are no longer separate tasks — structure and interaction are predicted jointly.

This transition required a conceptual change: from geometric constraint solving to probabilistic generative modeling.

---

# A Unified Molecular Representation

AlphaFold 2 operated primarily at the residue level. AlphaFold 3 instead introduces a unified token space where every molecular component is represented consistently.

If a system contains \( N \) total molecular units (residues, atoms, bases, ions), the model constructs:

$$
\text{Single} \in \mathbb{R}^{N \times c}
$$

$$
\text{Pair} \in \mathbb{R}^{N \times N \times c_p}
$$

The single representation stores per-token features, while the pair representation stores relational information between all token pairs.

Unlike AlphaFold 2, no assumption is made that tokens correspond only to amino acid residues. This unified representation enables modeling heterogeneous complexes directly.

---

# The Pairformer

The Evoformer trunk of AlphaFold 2 is replaced by the **Pairformer** in AlphaFold 3.  

The Pairformer jointly updates token embeddings and pair embeddings through attention and message passing.

Attention between tokens \( i \) and \( j \) is computed as:

$$
\mathrm{score}(i,j)
=
Q_i \cdot K_j
+
b_{ij}
$$

where

$$
Q_i = W_Q s_i,
\qquad
K_j = W_K s_j
$$

and the pair bias is derived from the pair representation:

$$
b_{ij} = W_b z_{ij}
$$

This formulation integrates relational reasoning directly into attention.

Pair embeddings are updated alongside token embeddings using learned transformations such as:

$$
z_{ij}
\leftarrow
\mathrm{MLP}(z_{ij}, s_i, s_j)
$$

Unlike AlphaFold 2, there are no explicit triangle attention modules.  
Relational consistency is instead learned implicitly through the diffusion objective.

---

# Diffusion-Based Structure Generation

The most significant architectural innovation of AlphaFold 3 is the introduction of diffusion modeling for structure generation.

Rather than refining rigid-body frames, the model directly generates atomic coordinates.

---

## Forward Diffusion Process (Training)

During training, clean coordinates

$$
x_0 \in \mathbb{R}^{3N}
$$

are progressively corrupted with Gaussian noise:

$$
x_t
=
\sqrt{\alpha_t} \, x_0
+
\sqrt{1 - \alpha_t} \, \epsilon
$$

where

$$
\epsilon \sim \mathcal{N}(0, I)
$$

and \( t \) denotes the diffusion timestep.

As \( t \) increases, the structure becomes increasingly noisy.

---

## Noise Prediction Objective

The model learns to predict the added noise:

$$
\hat{\epsilon}_\theta(x_t, t, \text{context})
$$

The training objective minimizes:

$$
\mathcal{L}
=
\mathbb{E}_{x_0, \epsilon, t}
\left[
\|
\epsilon
-
\hat{\epsilon}_\theta(x_t, t)
\|^2
\right]
$$

The context includes:

- Token embeddings from the Pairformer  
- Pair embeddings  
- Molecular identity features  
- Optional template information  

---

## Reverse Diffusion (Inference)

At inference time, the process is reversed.

We begin with pure Gaussian noise:

$$
x_T \sim \mathcal{N}(0, I)
$$

Coordinates are iteratively updated via learned denoising steps:

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}
\hat{\epsilon}_\theta(x_t, t)
\right)
+
\sigma_t z
$$

where

$$
z \sim \mathcal{N}(0, I)
$$

After sufficient iterations, the final structure \( x_0 \) is obtained.

---

# Conformer Generation and Chemical Priors

Ligands and flexible molecules introduce additional challenges due to internal torsional freedom.

AlphaFold 3 therefore generates chemically valid conformers prior to diffusion. These conformers respect:

- Fixed bond lengths  
- Bond angles  
- Allowed torsion states  

They provide plausible internal geometries, but remain flexible during diffusion.

This differs from AlphaFold 2, which predicted torsion angles explicitly and relied on rigid residue frames.

---

# Geometric Consistency and Equivariance

AlphaFold 2 enforced SE(3)-equivariance explicitly through rigid-body frames and invariant point attention.

AlphaFold 3 instead learns geometric consistency statistically.

If coordinates are globally rotated by \( R \), the model learns to satisfy approximately:

$$
\hat{\epsilon}_\theta(R x_t)
=
R \, \hat{\epsilon}_\theta(x_t)
$$

Equivariance is not hard-coded through rigid-body updates but emerges from training on 3D structures.

Geometric inconsistencies increase diffusion loss, pushing the model toward physically valid configurations.

---

# Sampling and Structural Ensembles

Because diffusion is stochastic, different random initializations:

$$
x_T \sim \mathcal{N}(0, I)
$$

produce different valid structures.

This enables:

- Sampling alternative conformations  
- Modeling flexible binding modes  
- Representing uncertainty  
- Generating structural ensembles  

Unlike AlphaFold 2, which produces largely deterministic outputs, AlphaFold 3 naturally models structural variability.

---

# Conceptual Comparison with AlphaFold 2

AlphaFold 2 can be summarized as:

- Deterministic  
- Residue-centric  
- Explicit geometric reasoning  
- Rigid-body equivariant refinement  

AlphaFold 3 is:

- Generative  
- Atom-centric and multi-molecular  
- Diffusion-based  
- Unified interaction modeling  

If AlphaFold 2 acts as a geometric constraint solver, AlphaFold 3 behaves as a learned molecular simulator guided by relational context.


---

## References

[1] GeeksforGeeks, *Amino Acids: Definition, Structure, Properties, Classification*.  
https://www.geeksforgeeks.org/amino-acids-definition-structure-properties-classification/

[2] Biology4Alevel, *Protein Structure*.  
http://biology4alevel.blogspot.com/2014/

[3] Boris Burkov, *Why AlphaFold 2 Matters*.  
https://borisburkov.net/2021-12-25-1/

[4] Simon Kohl, *Highly Accurate Protein Structure Prediction with AlphaFold*, Heidelberg.ai, May 05 2022.  
https://heidelberg.ai/2022/05/05/alpha-fold.html :contentReference[oaicite:1]{index=1}