# Trajectory tokenization

The trajectory of a single PMMA chain is obtained from a `LAMMPS` simulation and visualized using `ovito`, as shown in the figure below.
<img src="data\PMMA_single_chain.png" alt="The snapshot of a PMMA chain molecular" width="300" />

The script `data\generate_trajectory.py` constructs a continuous spatio-temporal tensor from the LAMMPS trajectory by mathematically mapping atomic coordinate data into a state sequence of positions and orientational quaternions. 

Initially, the algorithm parses the data file's topology to identify $N$ backbone atoms and their corresponding side-chain atoms connected via A-B bonds. For any given timeframe $t$ and specific bonded pair, let the spatial position of the backbone atom be denoted as $\mathbf{p} = (x, y, z)$ and the corresponding side-chain atom as $\mathbf{s} = (x_ {s}, y_ {s}, z_ {s})$. The relative orientation of the side chain is computed via the displacement vector $\mathbf{v} = \mathbf{s} - \mathbf{p}$, which is then normalized to extract the pure directional components $\mathbf{\hat{d}}$:

$$\mathbf{\hat{d}} = \frac{\mathbf{v}}{\Vert{}\mathbf{v}\Vert{}} = (d_x, d_y, d_z)$$

To represent this orientation without singularity-prone Euler angles, the direction vector is mapped to a quaternion $\mathbf{q} = (q_ {w}, q_ {x}, q_ {y}, q_ {z})$ describing the rotation from the standard Cartesian reference vector $\mathbf{k} = (0, 0, 1)$ to $\mathbf{\hat{d}}$. The unnormalized quaternion components are derived algebraically from the scaled rotation axis $\mathbf{k} \times \mathbf{\hat{d}}$ and the angle cosine derived from $\mathbf{k} \cdot \mathbf{\hat{d}}$, yielding the deduction $q_ {w} = 1.0 + d_ {z}$, $q_ {x} = -d_ {y}$, $q_ {y} = d_ {x}$, and $q_ {z} = 0.0$. This vector is subsequently normalized by its Euclidean norm to generate a unit quaternion:

$$\mathbf{q}_{norm} = \frac{(1.0 + d_z, -d_y, d_x, 0.0)}{\sqrt{(1.0 + d_z)^2 + d_y^2 + d_x^2}}$$

In the edge case where the vector is perfectly anti-parallel to the reference ($d_ {z} \lt -0.999999$), the script avoids division by zero by strictly assigning the orthogonal quaternion $\mathbf{q}_ {norm} = (0.0, 1.0, 0.0, 0.0)$. Finally, the algorithm concatenates the normalized quaternion and the backbone position into a 7-dimensional state vector:

$$\mathbf{x} = [q_w, q_x, q_y, q_z, p_x, p_y, p_z]$$

for each atom pair, iterating over all $T$ frames to compile the final dataset as a mathematical tensor of dimensions $(T, N, 7)$.

We have also tested the conversion, with the result shown in `data\test.ipynb`

The resulting tensor is used as input for the subsequent machine learning model, enabling the analysis of polymer dynamics and conformational changes over time.

# Training the model

The following algorithms are extracted from the original project: https://github.com/bjing2016/mdgen.

**Algorithm 1:** Velocity Network (`LatentMDGenModel`)
***
* **Require:** Target latent inputs $x$, initial frame roto-translations $g_ {1}$, torsions $\tau_ {1}$, amino acid identities $A$, flow timestep $t$, conditioning masks.
1. **Embed inputs:** $x \leftarrow \text{Linear}(x) + \text{PositionalEmbeddings} + \text{TimeEmbeddings}$
2. **Add initial state conditioning:** $x \leftarrow x + \text{Linear}(x_ {cond}) + \text{Embedding}(x_ {cond-mask})$
3. **Embed timestep:** $t_ {emb} \leftarrow \text{TimestepEmbedder}(t \times \text{time-multiplier})$
4. **Initialize prepended IPA features:** $x_ {ipa} \leftarrow \text{Embedding}(A)$
5. **for** $l = 1$ **to** $\text{num-ipa-layers}$ **do**
   1. $x_ {ipa} \leftarrow \text{IPALayer}(x_ {ipa}, t_ {emb}, mask_ {t=1}, g_ {1})$
6. **Broadcast IPA output across time:** $x \leftarrow x + x_ {ipa}[:, \text{None}]$
7. **for** $l = 1$ **to** $\text{num-transformer-layers}$ **do**
   1. $x \leftarrow \text{LatentMDGenLayer}(x, t_ {emb}, mask, g_ {1})$
8. **Project to flow velocity:** $v \leftarrow \text{FinalLayer}(x, t_ {emb})$
9. **return** $v$

**Algorithm 2:** Main Transformer Block (`LatentMDGenLayer`)
***
* **Require:** Input tensor $x$, timestep embedding $t_ {emb}$, mask $mask$, initial frame geometry $g_ {1}$.
1. **Extract modulation parameters:** 
   * $(\gamma_ {s}, \beta_ {s}, g_ {s}, \gamma_ {t}, \beta_ {t}, g_ {t}, \gamma_ {m}, \beta_ {m}, g_ {m}) \leftarrow \text{chunk}(\text{AdaLN}(t_ {emb}), 9)$
2. **if** $\text{Interleaved IPA is configured}$ **then**
   1. $x \leftarrow x + \text{InvariantPointAttention}(\text{LayerNorm}(x), g_ {1}, mask)$
3. **Spatial Attention (Residues):**
   1. $x_ {norm} \leftarrow \text{Modulate}(\text{LayerNorm}(x), \gamma_ {s}, \beta_ {s})$
   2. $x \leftarrow x + g_ {s} \odot \text{AttentionWithRoPE}_ {spatial}(x_ {norm}, mask)$
4. **Temporal Attention (Frames):**
   1. $x_ {norm} \leftarrow \text{Modulate}(\text{LayerNorm}(x), \gamma_ {t}, \beta_ {t})$
   2. $x \leftarrow x + g_ {t} \odot \text{AttentionWithRoPE}_ {temporal}(x_ {norm}, mask)$
5. **Feed-Forward Network (MLP):**
   1. $x_ {norm} \leftarrow \text{Modulate}(\text{LayerNorm}(x), \gamma_ {m}, \beta_ {m})$
   2. $x \leftarrow x + g_ {m} \odot \text{MLP}(x_ {norm})$
6. **return** $x$

**Algorithm 3:** Invariant Point Attention Block (`IPALayer`)
***
* **Require:** Input tensor $x$, timestep embedding $t_ {emb}$, mask $mask$, initial frame geometry $g_ {1}$.
1. **Extract modulation parameters:** 
   * $(\gamma_ {s}, \beta_ {s}, g_ {s}, \gamma_ {m}, \beta_ {m}, g_ {m}) \leftarrow \text{chunk}(\text{AdaLN}(t_ {emb}), 6)$
2. **Geometric Processing:**
   1. $x \leftarrow x + \text{InvariantPointAttention}(\text{LayerNorm}(x), g_ {1}, mask)$
3. **Spatial Attention:**
   1. $x_ {norm} \leftarrow \text{Modulate}(\text{LayerNorm}(x), \gamma_ {s}, \beta_ {s})$
   2. $x \leftarrow x + g_ {s} \odot \text{AttentionWithRoPE}_ {spatial}(x_ {norm}, mask)$
4. **Feed-Forward Network (MLP):**
   1. $x_ {norm} \leftarrow \text{Modulate}(\text{LayerNorm}(x), \gamma_ {m}, \beta_ {m})$
   2. $x \leftarrow x + g_ {m} \odot \text{MLP}(x_ {norm})$
5. **return** $x$

# Result analysis

The training result is shown in `result\1.2_50frames_PMMA_1w\ellipsoid_symmetric\log.out`. The comparison of the inference result and the original trajectory is shown in `result\1.2_50frames_PMMA_1w\inference\test_dataset.ipynb`. We can find that the generated trajectory is continuous and smooth, which is expected for a molecular dynamics simulation. In addition, the generated trajectory is similar to the original trajectory, which indicates that the model can learn the dynamics of the PMMA chain.