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

AlphaFold represents a pivotal achievement in biological AI, particularly in **3D protein structure prediction**. The work led to the 2024 Nobel Prize in Chemistry being awarded to Demis Hassabis, leader of the AlphaFold project, and John Jumper, for groundbreaking contributions to computational biology and protein folding prediction.

We begin with the biological foundations before moving towards the deep learning architectures that enabled this breakthrough. While this post primarily focuses on protein structures, it is important to note that AlphaFold 3 extends beyond isolated proteins and is capable of modeling full biomolecular complexes, including interactions with DNA, RNA, ligands, and ions.

---

## Contents

1. [Proteins](#proteins)  
2. [Protein Folding Problem](#protein-folding-problem)  
3. [Previous Work and Quality Assessments](#previous-work-and-quality-assessments)  
4. [AlphaFold 2](#alphafold-2)  
5. [AlphaFold 3](#alphafold-3)  
6. [Experiments and Results](#experiments-and-results)  
7. [Concluding Thoughts and Limitations](#concluding-thoughts)    

---

# Proteins

Proteins are essential biomolecules and macromolecules present in every cell of all living organisms. They are among the most abundant organic molecules in biological systems and perform a vast range of functions necessary for life.

Proteins are responsible for nearly every task of cellular life. They act as enzymes, structural components, transport molecules, signaling agents, immune defenders, and molecular motors. Because structure determines function, understanding protein structure is central to understanding biology itself.

The **structure of a protein is directly linked to its function**, making structural biology fundamental for studying physiological processes such as human health, disease, enzyme catalysis, immune response, and drug interactions. Structural insight enables rational drug design and targeted therapeutic development.

Although the amino acid sequence encodes the primary structural information of a protein, the final three-dimensional conformation is not determined by sequence alone. The cellular environment also plays a crucial role. Factors such as pH, temperature, ionic concentration, molecular crowding, and the presence of molecular chaperones can significantly influence folding pathways and stability.

---

## Amino Acids — The Building Blocks

Proteins are composed of one or more chains of amino acids. Each amino acid is an organic molecule built around a central carbon atom, known as the C-α atom, which is bonded to four distinct groups: an amino group (NH₂), a carboxyl group (COOH), a hydrogen atom, and a variable side chain, often referred to as the R-group.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/amino_acid.png' | relative_url }}" alt="Amino Acid Structure" width="450">
  <figcaption>
    <strong>Figure 1:</strong> General structure of an amino acid showing the central C-α atom, amino group, carboxyl group, hydrogen atom, and variable side chain.
    <br>
    <em>Source:</em> Adapted from GeeksforGeeks, <em>Amino Acids: Definition, Structure, Properties, Classification</em> <sup><a href="#ref1">[1]</a></sup>.
  </figcaption>
</figure>

The side chain determines the chemical and physical properties of the amino acid. In nature, there are 20 standard amino acids, each characterized by a unique side chain and represented by a one-letter code. Once incorporated into a protein chain, an amino acid is referred to as a **residue**.

When amino acids connect, they form peptide bonds through a dehydration synthesis reaction. The carboxyl group of one amino acid reacts with the amino group of another, releasing a water molecule and forming a stable covalent linkage.

A protein is therefore fundamentally a linear sequence of residues, and this sequence encodes the information required for the protein to adopt its final three-dimensional structure.

---

# Protein Folding Problem

Proteins do not function as linear chains. Instead, they fold into highly specific three-dimensional structures that determine their biological activity.

Protein structure is commonly described at four hierarchical levels. The primary structure corresponds to the amino acid sequence. The secondary structure refers to local motifs such as α-helices and β-sheets formed by hydrogen bonding along the backbone. The tertiary structure describes the complete three-dimensional conformation of a single polypeptide chain. The quaternary structure refers to the arrangement of multiple interacting chains (subunits) into a functional complex.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/protein_levels.png' | relative_url }}" alt="Protein Structure Levels" width="600">
  <figcaption>
    <strong>Figure 2:</strong> Hierarchical organization of protein structure — from primary sequence to quaternary assembly.
    <br>
    <em>Source:</em> Biology4Alevel, <em>Protein Structure</em> <sup><a href="#ref2">[2]</a></sup>.
  </figcaption>
</figure>

Protein folding can be understood as an energy-driven physical process in which a polypeptide chain transitions from a random coil to a thermodynamically stable, low-energy three-dimensional conformation. This transformation is driven by hydrogen bonding, hydrophobic interactions, electrostatic forces, and van der Waals interactions.

Correct folding is essential for biological function. Misfolded proteins can become inactive or toxic and are associated with numerous diseases, including Alzheimer’s disease, cancer, neurodegenerative disorders, and certain allergies.

Understanding protein folding is therefore central to medicine, drug discovery, and biotechnology. Structural insight enables the rational design of pharmaceuticals and helps interpret the functional impact of genetic mutations and variations.

---

## Why Is Protein Folding Important?

Protein misfolding is linked to numerous diseases. For example, the accumulation of misfolded amyloid proteins plays a central role in Alzheimer’s disease, while structural abnormalities in regulatory proteins are associated with various forms of cancer. When proteins fail to adopt their correct three-dimensional conformation, they can lose functionality or become toxic to cells.

Understanding protein folding is therefore central to several major scientific and industrial fields. In medicine, structural insight enables the identification of disease mechanisms at the molecular level. In drug discovery, knowledge of a protein’s three-dimensional structure allows researchers to design molecules that precisely bind to active or regulatory sites. In biotechnology, engineered proteins with specific structural properties can be developed for industrial enzymes, synthetic biology applications, or targeted therapies.

Improved insight into folding mechanisms directly supports the development of new pharmaceuticals. By understanding how proteins fold and interact, researchers can predict binding interfaces, stabilize unstable proteins, and design inhibitors that block harmful interactions.

Moreover, protein structures help interpret genetic variation and mutations. Many mutations do not simply change a sequence, they alter structural stability or binding behavior. Structure prediction therefore helps explain why certain mutations are pathogenic while others are benign.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Corona.JPG' | relative_url }}" alt="Predicted protein structure example" width="450">
  <figcaption><strong>Figure 3:</strong>  Human coronavirus OC43 spike protein (4,665 residues), heavily glycosylated and bound by neutralizing antibodies.</figcaption>
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

Experimental techniques such as **X-ray crystallography** expose crystallized proteins to X-rays and reconstruct electron density maps from diffraction patterns. A typical X-ray pipeline involves protein crystallization, a particularly challenging step for membrane proteins due to their low solubility, followed by X-ray diffraction and reconstruction of electron density maps. One of the central technical challenges in this process is the *phase problem*, which requires specialized methods to resolve. Even after obtaining an electron density map, reconstructing the final 3D atomic model involves a degree of interpretation; especially at lower resolutions, two experts may produce slightly different structural models from the same data. While highly accurate, this method captures only the final folded structure, requires substantial laboratory effort, and is expensive, often costing over \$100,000 per structure <sup><a href="#ref3">[3]</a></sup>.


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

Many mutations are tolerated without disrupting overall fold, allowing sequence divergence while preserving structural constraints. As a result, protein sequences can diverge substantially across distant species (often by more than 70%), while their three-dimensional structures remain remarkably conserved <sup><a href="#ref3">[3]</a></sup>. In other words:

> Protein structure is far more conserved than protein sequence.

This observation is central. If structure changes slowly while sequences mutate, then comparing homologous sequences across species allows us to infer structural constraints indirectly.

Rather than predicting structure directly from physics-based simulations, AlphaFold 2 learns a representation of residue–residue relationships extracted from evolutionary variation and then converts this relational structure into a three-dimensional object through an equivariant structure module.

Conceptually, the system operates in two major stages:

1. **Relational reasoning within the representation space (Evoformer)**
2. **Geometric reconstruction in 3D space (Structure Module)**

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Fold2_Archi.PNG' | relative_url }}" width="750">
  <figcaption><strong>Figure 4:</strong> Complete AlphaFold 2 pipeline (Jumper et al., 2021) <sup><a href="#ref5">[5]</a></sup>.</figcaption>
</figure>

---

## Input Feature Extraction

The input to AlphaFold 2 is a single amino acid sequence of length ***N***. From this sequence, the model constructs two primary feature tensors:

- A **Multiple Sequence Alignment (MSA)** representation  
- A **Pair representation**

These two representations form the foundation of the entire architecture.

---

### Multiple Sequence Alignment (MSA)

Starting from the target sequence, AlphaFold performs a large-scale database lookup to retrieve homologous sequences from diverse species. These sequences are arranged in a matrix called a **Multiple Sequence Alignment (MSA)**, where:

- Each row corresponds to a homologous protein sequence  
- Each column corresponds to a specific residue position  

By aligning homologues in this way, corresponding residues across species are placed in the same column.

This alignment encodes powerful evolutionary signals:

- **Conservation** of a column often implies structural or functional importance (e.g., catalytic residues, ligand binding sites).
- **Co-evolution** between two columns suggests structural coupling; if residue *i* mutates and residue *j* consistently mutates in response, the two residues are likely interacting in 3D space.

Intuitively, if two residues participate in a bonding mechanism, mutating one without compensating changes would destabilize the protein. Therefore, correlated mutations preserve structural integrity <sup><a href="#ref4">[4]</a></sup>.

The matrix alignments are embedded into a 3D tensor:

$$
\begin{aligned}
\mathrm{MSA} &\in \mathbb{R}^{N_{\text{seq}} \times N \times c_m} \\
N_{\text{seq}} &:\ \text{number of aligned sequences} \\
N &:\ \text{number of residues in the protein} \\
c_m &:\ \text{embedding dimension}
\end{aligned}
$$

This tensor does not explicitly encode distances. Instead, it encodes **evolutionary constraints**, from which geometric structure can be inferred.

Evolution therefore acts as indirect supervision for the 3D structure prediction.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Coevolution.JPG' | relative_url }}" width="550">
  <figcaption><strong>Figure 5:</strong> Multiple Sequence Alignment capturing evolutionary conservation and co-evolution signals.</figcaption>
</figure>

---

### Pair Representation

While the MSA captures evolutionary variation, protein structure itself is fundamentally about **relationships between residues**. To reason explicitly about residue–residue interactions, AlphaFold maintains a second 3D tensor:

$$
\mathrm{Pair} \in \mathbb{R}^{N \times N \times c_z}
$$

Each slice of this tensor is a learned feature vector encoding the model’s current belief about how residues *i* and *j* relate geometrically.

This representation can be interpreted as a complete graph over residues: each residue corresponds to a node, and every pair of residues is connected by an edge that stores relational features.

Together, the MSA tensor and the Pair tensor provide complementary perspectives. The MSA captures evolutionary constraints across species, whereas the Pair representation models geometric relationships within a single protein. 


---

## The Evoformer: Iterative Representation Refinement

The Evoformer is the core reasoning engine of AlphaFold 2. It consists of a deep stack of blocks (48 in total) that iteratively refine both the MSA and pair representations.

Within each block, two complementary processes take place:

- Evolutionary information is learned through axial attention (row-wise and column-wise) in the MSA stack  
- Geometric relationships are refined through triangle updates on the pair stack, that respect the triangle inequality  

Crucially, information flows bidirectionally between these two representations. 

As the network trunk is progressively refined, evolutionary patterns become embedded within the representations, providing the foundation for reconstructing the protein’s three-dimensional structure.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Evo.JPG' | relative_url }}" width="700">
  <figcaption><strong>Figure 6:</strong> Evoformer architecture showing MSA attention and pair updates <sup><a href="#ref5">[5]</a></sup>.</figcaption>
</figure>

---

## Axial Attention in the MSA Stack

Applying full attention over the entire tensor would be computationally infeasible. Instead, AlphaFold factorizes attention along individual tensor modes after unfolding them, technique known as **axial attention**.

This means that attention is applied along one axis at a time, with the tensor unfolded along the specific mode of interest:

- Along residues (row-wise attention; mode-2 tensor unfolding)  
- Along sequences (column-wise attention; mode-1 tensor unfolding)  


---

### Row-wise Attention — Structural Reasoning Within a Sequence

In row-wise attention, a single sequence is considered while attention is applied across its residue positions. One row of the MSA, with shape

$$
N \times c_m
$$

is selected, and self-attention is applied along the residue axis.

This operation captures long-range interactions between residues within a single protein chain.

Importantly, row-wise attention is biased by the pair representation. The pair tensor provides relational information about which residue pairs are likely to interact. This information is incorporated into the attention logits as an additive bias term. 

---

### Column-wise Attention — Evolutionary Reasoning Across Sequences

In column-wise attention, a single residue position is fixed while attention is applied across homologous sequences.

A column of the MSA, with shape

$$
N_{\text{seq}} \times c_m
$$

is selected, and self-attention is applied along the sequence axis.

This mechanism captures evolutionary variation at a specific residue position. Signals such as conservation, mutation patterns and co-evolution are extracted.

Column-wise attention therefore models how a structural position behaves across evolutionary time. Together, row and column attention allow the Evoformer to extract meaningful information from the evolution process.

---

## Triangle Updates: Enforcing Geometric Consistency

While axial attention extracts evolutionary patterns, it does not by itself guarantee geometric consistency. This is why the triangle updates attempt to enforce geometric consistency within the model.

If residue *i* is predicted to be close to residue *k*, and residue *k* is close to residue *j*, then residue *i* cannot be arbitrarily far from residue *j*. In Euclidean space, distances must satisfy the triangle inequality:

$$
d(i,j) \le d(i,k) + d(k,j)
$$

AlphaFold does not hard-code this constraint into the loss function. Instead, it introduces triangle updates within the pair representation, allowing geometric consistency to emerge through the architecture itself.

---

### Triangle Attention

The pair tensor, as we previously mentioned, can be interpreted as a complete graph over residues, where each edge stores a learned embedding describing the relationship between two residues.

To update the relationship between residues *i* and *j*, the model considers all possible third residues *k*. In other words, it reasons over triangles (*i*, *j*, *k*).

The attention weights over the intermediate residue *k* are computed as:

$$
\alpha_{ijk}
=
\mathrm{softmax}_k
\left(
\frac{Q_{ij} \cdot K_{ik}}{\sqrt{d}}
+
b_{jk}
\right)
$$

The updated pair embedding becomes:

$$
z'_{ij}
=
\sum_k
\alpha_{ijk} V_{ik}
$$

Residue *k* therefore acts as a mediator between *i* and *j*. If both (*i*, *k*) and (*j*, *k*) suggest compatibility, this influences the updated belief about (*i*, *j*).

A symmetric variant ensures that information flows consistently around both orientations of the triangle.

---

### Why This Matters

Triangle attention allows local pairwise predictions to become globally coherent and respect the geometry of the Euclidean
space. Instead of treating each residue pair independently, the model assures their compatibility with other surrounding residues.

No explicit geometric rule is imposed. Instead, geometry is enforced through the way in which the attention mechanism is built, encouraging the pair representation to become geometrically consistent.

The combination of axial attention, which extracts evolutionary patterns, and triangle attention, which enforces geometric structure, makes the learned feature representation rich and lays the foundation for the structure module.

---

## Structure Module: From Representation to Geometry

After the Evoformer produces a consistent relational representation, AlphaFold translates this abstract structure into explicit 3D coordinates.

The Structure Module performs this transition. It treats the protein not as a continuous chain at first, but as a collection of independent rigid units, also denoted in the paper as a **residue gas**.

Each amino acid is modeled as a rigid triangle defined by its backbone atoms (N, Cα, C). Importantly, these residues are initially *disconnected* and placed at the origin (idea of cutting the peptide chain). The polymer chain constraint is temporarily ignored. The network must therefore learn how to assemble this unordered collection of rigid residues into a coherent three-dimensional structure.

The Structure Module is designed to be **SE(3)-equivariant**, meaning:

- Rotating or translating the entire protein does not change internal decisions  
- Predictions transform consistently under rigid-body motion  

<figure style="text-align: center;">
  <img src="{{ '/assets/images/Structure_Module.PNG' | relative_url }}" width="650">
  <figcaption><strong>Figure 7:</strong> Structure module translating relational embeddings into 3D frames (Jumper et al., 2021) <sup><a href="#ref5">[5]</a></sup>.</figcaption>
</figure>


---

### Rigid-Body Representation of Residues

Each residue *i* is assigned a rigid transformation:

$$
T_i = (R_i, t_i)
$$

where:

$$
R_i \in SO(3) \quad \text{is a rotation matrix}, \qquad
t_i \in \mathbb{R}^3 \quad \text{is a translation vector}.
$$

Initially:

$$
R_i = I, \quad t_i = 0
$$

All residues therefore begin as overlapping triangles at the origin, also known as the "gas" configuration, or "Black hole initialization". The task of the structure module is to progressively rotate and translate these rigid units into their correct spatial arrangement.

---

### Invariant Point Attention (IPA)

The central mechanism enabling geometric reasoning is the Invariant Point Attention (IPA).

IPA augments standard feature-based attention with geometric distance invariance. Each residue predicts a set of learned local reference points:

$$
p_i^{(m)} \in \mathbb{R}^3
$$

These points are defined in the local coordinate frame of the residue. They are mapped into global coordinates via the current rigid transformation:

$$
\hat{p}_i^{(m)} = R_i p_i^{(m)} + t_i
$$

Attention between residues depends not only on feature similarity but also on geometric proximity <sup><a href="#ref6">[6]</a></sup>:

$$
\mathrm{score}(i,j)
=
Q_i \cdot K_j
-
\sum_m
\| \hat{p}_i^{(m)} - \hat{p}_j^{(m)} \|^2
$$

The crucial term is the squared Euclidean distance between transformed points.

Because Euclidean distance is invariant under global rotation and translation, this scoring function depends only on relative geometry, not absolute orientation in space. This ensures overall SE(3) equivariance.

If the entire protein were rotated or translated by a rigid motion:

- Each frame would transform accordingly  
- Distances between points would remain unchanged  
- Attention weights would remain identical  

Intuitively, residues attend more strongly to nearby residues in 3D space. The closer two residues are predicted to be, the stronger their geometric interaction.

---

### Frame Updates

After the attention mechanism, the network predicts incremental rigid-body updates:

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

These updates are applied iteratively (typically around 8 refinement steps). Early iterations establish coarse global layout, while later iterations refine local geometry.

This process resembles a learned geometric optimization procedure operating directly in the SE(3) group.


---

### Reintroducing the Chain: Backbone and Side-Chain Construction

Once the backbone frames are refined, atomic coordinates are placed deterministically.

For atom *a* in residue *i*:

$$
\begin{aligned}
x_{i,a}
&=
R_i \hat{x}_{i,a}(\chi_i)
+
t_i \\[8pt]
\hat{x}_{i,a}(\chi_i)
&:\ \text{canonical local coordinates of atom } a \text{ in residue } i \\[4pt]
\chi_i
&:\ \text{predicted torsion angles of residue } i
\end{aligned}
$$ 

Torsion angles are predicted via sine and cosine representations:

$$
(\sin \chi_i, \cos \chi_i)
$$

This avoids discontinuities due to angular periodicity.

Bond lengths and bond angles remain fixed, ensuring chemically valid geometry.

---

## Conceptual Summary

AlphaFold 2 separates protein folding into three conceptual phases:

1. **Learning relational evolutionary patterns and residue interactions in representation space (Evoformer)**
2. **Converting relational structure into three-dimensional geometry (Structure Module)**
3. **Refining geometry through SE(3)-equivariant updates**

This separation between high-dimensional relational reasoning and geometric knowledge is one of the central architectural innovations that enabled AlphaFold 2 to achieve near-experimental accuracy in CASP.


# AlphaFold 3 

AlphaFold 3 represents a substantial architectural step forward for the AlphaFold framework, moving toward more general structure prediction. While many core principles from AlphaFold 2 remain unchanged: relational reasoning, pairwise representations, deep iterative refinement; the mechanism used to generate molecular structure changes fundamentally.

Instead of predicting a single rigidly refined protein structure, AlphaFold 3 models:

- Proteins  
- DNA and RNA  
- Small-molecule ligands  
- Metal ions  
- Covalent modifications  

within one shared architecture. Folding and binding are no longer separate problems. Structure and interaction are predicted simultaneously within a unified architecture.

This transition required a conceptual change: from geometric constraint solving to probabilistic generative modeling.

---


## Architectural Differences

Despite the generative shift, the high-level structure resembles AlphaFold 2:

1. A large relational trunk builds abstract representations  
2. A structure module converts those representations into 3D coordinates  

However, both components have evolved.

- The Evoformer trunk is replaced by the Pairformer
- The SE(3)-equivariant Structure Module is replaced by a Diffusion Module

The architectural pattern remains similar: relational reasoning first, coordinate generation second. What changes is how geometry is produced.


## A Unified Molecular Representation

AlphaFold 2 operated primarily at the residue level. AlphaFold 3 instead introduces a unified token space where every molecular component is represented consistently.

If a system contains ***N*** total molecular units (residues, atoms, bases, ions), the model constructs:

$$
\text{Single} \in \mathbb{R}^{N \times c}
$$

$$
\text{Pair} \in \mathbb{R}^{N \times N \times c_p}
$$

The single representation stores per-token features, while the pair representation stores relational information between all token pairs.

Unlike AlphaFold 2, no assumption is made that tokens correspond only to amino acid residues. This unified representation enables modeling heterogeneous complexes directly.


---

## The Pairformer


The Evoformer in AlphaFold 2 alternated between MSA-based axial attention and triangle updates over residue pairs. While extremely powerful, this design was tightly coupled to explicit multiple sequence alignment (MSA) processing.

The Pairformer in AlphaFold 3 simplifies and generalizes this architecture.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/AF3.PNG' | relative_url }}" width="750">
  <figcaption><strong>Figure 8:</strong> AlphaFold 3 architecture (Abramson et al., 2024) <sup><a href="#ref7">[7]</a></sup>.</figcaption>
</figure>

Key differences include:

- The explicit stacked MSA representation of AlphaFold 2 is removed, with evolutionary information distilled into a unified single representation  
- A single representation replaces the alternating MSA stack  
- No outer-product mean updates  
- More streamlined communication between single and pair representations  

Attention between tokens *i* and *j* is computed as:

$$
\mathrm{score}(i,j)
=
Q_i \cdot K_j
+
b_{ij}
$$

where the bias term is derived from the pair representation and injected directly into the attention mechanism. This allows relational information to modulate attention weights without explicitly alternating between separate MSA and pair stacks.

Pair embeddings are updated in parallel through learned transformations and multiplicative interactions, preserving the idea that protein structure prediction is fundamentally about residue–residue relationships.

Conceptually, the Pairformer acts as a lighter and more general relational trunk. Whereas the Evoformer was tightly specialized for protein MSAs, the Pairformer is designed to operate across heterogeneous molecular systems, including proteins, nucleic acids, ligands, and ions.


---

## Diffusion-Based Structure Generation

The most significant innovation in AlphaFold 3 is the replacement of rigid-body refinement with diffusion modeling.

AlphaFold 2 predicted backbone frames and torsion angles deterministically.  
AlphaFold 3 instead generates full atomic coordinates through iterative denoising.

During training, clean coordinates are progressively corrupted with Gaussian noise:

$$
x_t
=
\sqrt{\alpha_t} \, x_0
+
\sqrt{1 - \alpha_t} \, \epsilon
$$

The network learns to predict the added noise, conditioned on the relational representations built by the Pairformer.

At inference time, the process is reversed. Starting from pure Gaussian noise:

$$
x_T \sim \mathcal{N}(0, I)
$$

the model iteratively denoises coordinates until a coherent molecular structure emerges.

Most of the computation happens in the conditioning trunk (the Pairformer). The diffusion module then samples structures consistent with the learned relational representation.

---

## Conformer Generation and Chemical Priors

For ligands and flexible molecules, chemically valid conformers are generated prior to diffusion.

These conformers respect:

- Fixed bond lengths  
- Bond angles  
- Allowed torsion states  

Diffusion then adjusts global placement and interaction geometry.

This differs from AlphaFold 2, where torsion angles were predicted explicitly within rigid residue frames. AlphaFold 3 moves closer to full atomic generative modeling.


---

## Summary of the Shift

AlphaFold 2 produced nearly deterministic predictions for a given input sequence and it can be seen as a geometric solver in SE(3).

AlphaFold 3 introduces stochastic sampling and is a generative model over atomic coordinates. Different initial noise realizations can yield different, yet physically plausible, conformations.

This probabilistic formulation enables:

- Sampling alternative binding poses  
- Modeling conformational variability  
- Representing structural uncertainty  
- Generating structural ensembles  

The treatment of equivariance also evolves. In AlphaFold 2, SE(3)-equivariance was enforced explicitly through specialized architectural components such as IPA. In AlphaFold 3, rigid-body equivariance is encouraged statistically through a combination of equivariant conditioning and random rotation–translation data augmentation applied during training. Rather than hard-coding symmetry into attention updates, the model learns to respect geometric symmetries within the diffusion framework.

# Experiments and Results

## Historical Context: CASP Performance Progression

Before AlphaFold 3, AlphaFold 2 fundamentally changed the field at CASP14 (2020).  
The table below shows the progression of leading methods across CASP editions.



| CASP Edition | Year | Top GDT_TS | Leading Method                         |
|--------------|------|------------|----------------------------------------|
| CASP5        | 2002 | 40         | Rosetta                                |
| CASP7        | 2006 | 50         | Rosetta (fragment assembly)            |
| CASP10       | 2012 | 55         | Zhang-Server (QUARK)                   |
| CASP12       | 2016 | 55         | RaptorX, Baker-Robetta                 |
| CASP13       | 2018 | 65         | AlphaFold 1 (DeepMind)                 |
| CASP14       | 2020 | **92.4**   | **AlphaFold 2 (DeepMind)**             |


At CASP14, AlphaFold 2 achieved a median GDT-TS score of 92.4, drastically surpassing previous methods.  
For many targets, predictions reached near-experimental accuracy, with Cα RMSD values close to 1 Å.

AlphaFold 3 extends this success beyond protein folding into full biomolecular interaction modeling.

## Evaluation Datasets in AlphaFold 3

Unlike AlphaFold 2, which was primarily evaluated on folding benchmarks such as CASP, AlphaFold 3 was tested on a broader range of interaction-focused datasets. The goal was not only to assess folding accuracy, but to evaluate how well the model predicts full molecular assemblies.

AlphaFold 3 was evaluated on 8,856 PDB complexes released between May 1, 2022, and January 12, 2023. High-homology chains (>40% sequence identity to the training set) were removed, and chains and interfaces were scored separately. A low-homology subset was used for direct protein–protein comparison against AlphaFold-Multimer v2.3.

For antibody–antigen systems, a dedicated subset of 71 complexes (166 interfaces across 65 interface clusters) was analyzed. These systems are particularly challenging due to flexible loops and highly specific binding geometries.

Protein–ligand performance was evaluated using the PoseBusters benchmark (v1 and v2), which includes chemical validity checks, steric clash detection, and pocket-aligned ligand RMSD with a success threshold of < 2 Å. Crystallization artifacts and glycans were removed to ensure fair comparison.

Additionally, AlphaFold 3 was evaluated on several CASP15 RNA targets (R1116, R1117, R1126, R1128, R1136, R1138, R1189, R1190) and compared against RF2NA and other RNA prediction systems <sup><a href="#ref7">[7]</a></sup>.

## Protein–Protein Interactions

On the recent PDB low-homology subset, AlphaFold 3 consistently outperforms AlphaFold-Multimer v2.3. Improvements are observed across general protein–protein interfaces and are particularly pronounced for antibody–antigen complexes.

<figure>
  <img src="{{ '/assets/images/ProteinProtein.JPG' | relative_url }}"
       style="width: 60%; display: block; margin: 0 auto;">
  <figcaption style="text-align: center;">
    <strong>Figure 8:</strong> AlphaFold 3 improvements for protein–protein interactions (Abramson et al., 2024) <sup><a href="#ref7">[7]</a></sup>.
  </figcaption>
</figure>

---

# Protein–Ligand Prediction Improvements

Another striking advance mentioned in the results section of the AlphaFold 3 paper is its unified modeling of protein–ligand systems. Unlike traditional docking pipelines, AF3 does not require a predefined binding pocket and does not separate folding from docking. Instead, ligand poses are generated through the same diffusion process used for protein structure refinement.

<figure style="text-align: center;">
  <img src="{{ '/assets/images/ProteinLing.JPG' | relative_url }}" width="750">
  <figcaption>
    <strong>Figure 9:</strong> AlphaFold 3 improvements for protein–ligand prediction evaluated on PoseBusters and recent PDB datasets (Abramson et al., 2024) <sup><a href="#ref7">[7]</a></sup>.
  </figcaption>
</figure>

# Broader Implications

Taken together, these results illustrate a conceptual shift.

AlphaFold 2 established that deep learning could solve single-chain protein folding with near-experimental accuracy. AlphaFold 3 extends this paradigm toward unified interaction modeling across proteins, ligands, nucleic acids, and other biomolecular components.

The empirical gains in protein–protein and protein–ligand benchmarks suggest as well that diffusion-based generative modeling provides improved robustness in structurally diverse systems.



# Concluding Thoughts and Limitations


Although AlphaFold represents a major breakthrough in structure prediction, important limitations remain. These become particularly visible in the transition from AlphaFold 2 to the more generative AlphaFold 3 framework.

One issue observed in the AlphaFold 3 paper is the appearance of structured predictions in regions that are experimentally known to be intrinsically disordered. These regions do not adopt a single stable conformation in reality, yet the model may generate ordered structures for them. This behavior is often referred to as "hallucination". In most cases, the predicted confidence for these regions is low, indicating uncertainty. However, the model may still output geometrically coherent shapes that do not reflect true biological disorder. This illustrates a broader challenge in generative modeling: the system must not only generate plausible structures, but also correctly represent uncertainty and flexibility.

Another limitation concerns conformational diversity. Proteins are dynamic systems that exist as ensembles of conformations rather than single fixed structures. Many biological processes depend on structural transitions, such as conformational switching, induced-fit binding, and interactions. Current AlphaFold models primarily predict one dominant conformation or a limited set of sampled structures. They do not yet capture the full thermodynamic landscape of accessible states.

A related challenge arises when moving from structure prediction to molecular design. A protein or complex generated computationally may appear stable and chemically reasonable in static three-dimensional space. However, its behavior inside a living organism might be different and depends on many additional factors. These aspects are not directly modeled by current structure prediction systems.

Biological systems operate in dynamic, crowded, and context-dependent environments. Accurately modeling how proteins behave under physiological conditions requires integrating structural prediction with thermodynamics, kinetics, and systems-level biology. While AlphaFold has significantly advanced static structure prediction, understanding and modeling dynamic biomolecular behavior remains an open area of research.

Future progress may involve combining generative structure models with molecular dynamics simulations, experimental constraints, or time-dependent modeling approaches. The ability to predict not only structure, but also structural flexibility and functional behavior, represents the next major challenge in biomolecular AI.



---



## References

<div id="ref1">

[1] GeeksforGeeks. (n.d.). <em>Amino Acids: Definition, Structure, Properties, Classification</em>. Retrieved from  
https://www.geeksforgeeks.org/amino-acids-definition-structure-properties-classification/

</div>

<div id="ref2">

[2] Biology4Alevel. (2014). <em>Protein Structure</em>. Retrieved from  
http://biology4alevel.blogspot.com/2014/

</div>

<div id="ref3">

[3] Burkov, B. (2021). <em>Why AlphaFold 2 Matters</em>. Retrieved from  
https://borisburkov.net/2021-12-25-1/

</div>

<div id="ref4">

[4] Kohl, S. (2022). <em>Highly Accurate Protein Structure Prediction with AlphaFold</em>. Heidelberg.ai. Retrieved from  
https://heidelberg.ai/2022/05/05/alpha-fold.html

</div>

<div id="ref5">

[5] Jumper, J., Evans, R., Pritzel, A., et al. (2021). <em>Highly accurate protein structure prediction with AlphaFold</em>. Nature, 596, 583–589. Retrieved from  
https://doi.org/10.1038/s41586-021-03819-2

</div>

<div id="ref6">

[6] AI4Pharm. (2022). <em>AlphaFold2 — Understanding the Architecture and Invariant Point Attention (IPA) Module</em>. Retrieved from  
https://www.ai4pharm.info/alphafold2

</div>

<div id="ref7">

[7] Abramson, J., Adler, J., Dunger, J., et al. (2024). <em>Accurate structure prediction of biomolecular interactions with AlphaFold 3</em>. Nature. Retrieved from  
https://doi.org/10.1038/s41586-024-07487-w

</div>