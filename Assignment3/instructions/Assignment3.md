# DD2424 — Assignment 3

In this assignment you will train and test a three‑layer network with multiple outputs to classify images from the CIFAR‑10 dataset. The first layer will be a **convolution** applied with a stride equal to the width of the filter. In computer vision this corresponds to a **patchify** layer (Figure 1) and is the first layer used in Vision Transformers and architectures such as **MLP‑Mixer** and **ConvMixer**. You will train the network using mini‑batch gradient descent on a cost function that computes the **cross‑entropy loss** of the classifier on the labeled training data and an **L2 regularization** term on the weight matrices.

> **Figure 1 (conceptual):** To patchify an input image, split it into a regular grid of non‑overlapping sub‑regions. For ViTs, the pixels in each patch are flattened, linearly projected (affine transform), and the resulting vectors are fed to a Transformer. Patchify can be implemented as a convolution with stride equal to the filter width.

The overall structure of your code should mimic the previous assignments. Parameters differ slightly, and you must change the functions that (1) evaluate the network (forward pass) and (2) compute gradients (backward pass). Implementing the first convolutional layer efficiently is key to achieve reasonable CPU training times. The reward should be improved performance over Assignment 2 (but don’t get your expectations too high!).

---

## Background 1: Network with an initial patchify layer

Each input image (X) is a 3D array of size (32 \times 32 \times 3).

The network applies:

$$
H_i = \max(0,; X * F_i),\quad i = 1,\ldots,n_f \tag{1}
$$

$$
\mathbf{h} = \begin{pmatrix}
\operatorname{vec}(H_1)\
\operatorname{vec}(H_2)\
\vdots\
\operatorname{vec}(H_{n_f})
\end{pmatrix} \tag{2}
$$

$$
\mathbf{x}_1 = \max\bigl(0,; W_1, \mathbf{h} + \mathbf{b}_1\bigr) \tag{3}
$$

$$
\mathbf{s} = W_2, \mathbf{x}_1 + \mathbf{b}_2 \tag{4}
$$

$$
\mathbf{p} = \operatorname{SoftMax}(\mathbf{s}) \tag{5}
$$

Where:

* The input image (X) has size (32\times 32\times 3).
* Each filter (F_i) has size (f \times f \times 3) and is applied with stride (f) and no padding. Possible values: (f \in {2,4,8,16}). The set of filters is (\mathcal{F} = {F_1,\ldots,F_{n_f}}).
* Each (H_i) has size ((32/f) \times (32/f) \times 1).
* The (H_i) are flattened and concatenated, so (\mathbf{h}) has size (n_f n_p \times 1) where (n_p = (32/f)^2) is the number of sub‑patches.
* Two fully‑connected layers follow: (W_1) has size (d \times d_0) with (d_0 = n_f n_p), and (W_2) has size (K \times d). Thus (\mathbf{s}) has size (K \times 1).
* Softmax:
  $$\operatorname{SoftMax}(\mathbf{s}) = \frac{\exp(\mathbf{s})}{\mathbf{1}^\top \exp(\mathbf{s})} \tag{6}$$
* Predicted class:
  $$k^* = \arg\max_{1\le k \le K} {p_1,\ldots,p_K} \tag{7}$$

**Vectorization convention.** If
$$ H = \begin{pmatrix} H_{11} & H_{12} \ H_{21} & H_{22} \end{pmatrix}, \quad \text{then}\quad \operatorname{vec}(H) = \begin{pmatrix} H_{11} \ H_{12} \ H_{21} \ H_{22} \end{pmatrix}. $$

> For clarity and simpler implementation, biases were omitted in the initial derivation (but you will add them later).

---

## Background 2: Writing the convolution as a matrix multiplication

To make back‑prop transparent and efficient on CPU (and to exploit fast matrix ops), express convolutions as matrix multiplications.

Consider an input matrix (X \in \mathbb{R}^{4\times 4}) and a filter (F \in \mathbb{R}^{2\times 2}). Convolving with stride 2 and no padding yields a (2\times 2) output. We can write:

$$ H = X * F ; \Rightarrow ; \mathbf{h} = M_X, \operatorname{vec}(F), \tag{9} $$

where (M_X \in \mathbb{R}^{4\times 4}) is built by stacking all non‑overlapping (2\times 2) sub‑blocks of (X):

$$
M_X = \begin{pmatrix}
X_{11} & X_{12} & X_{21} & X_{22}\
X_{13} & X_{14} & X_{23} & X_{24}\
X_{31} & X_{32} & X_{41} & X_{42}\
X_{33} & X_{34} & X_{43} & X_{44}
\end{pmatrix}. \tag{10}
$$

### Multiple channels

For a 3D input with two channels, (X \in \mathbb{R}^{4\times 4\times 2}), and a filter (F \in \mathbb{R}^{2\times 2\times 2}), the output is still (2\times 2). The matrix (M_X) becomes (4\times 8):

$$
M_X = \begin{pmatrix}
X_{111} & X_{121} & X_{211} & X_{221} & X_{112} & X_{122} & X_{212} & X_{222}\
X_{131} & X_{141} & X_{231} & X_{241} & X_{132} & X_{142} & X_{232} & X_{242}\
X_{311} & X_{321} & X_{411} & X_{421} & X_{312} & X_{322} & X_{412} & X_{422}\
X_{331} & X_{341} & X_{431} & X_{441} & X_{332} & X_{342} & X_{432} & X_{442}
\end{pmatrix}. \tag{12}
$$

Flatten (F) **channel by channel** (ordering must be consistent with how you form (M_X)):

$$ \operatorname{vec}(F) = \begin{pmatrix} F_{111}& F_{121}& F_{211}& F_{221}& F_{112}& F_{122}& F_{212}& F_{222} \end{pmatrix}^\top. \tag{13} $$

### Gradient to (F)

With (\mathbf{g} = \partial L/\partial \mathbf{h} \in \mathbb{R}^{4}),

$$ \frac{\partial L}{\partial\operatorname{vec}(F)} = M_X^\top \mathbf{g}. \tag{14} $$

For a batch (B):

$$ \frac{\partial L}{\partial\operatorname{vec}(F)} = \frac{1}{|B|}\sum_{(X,y)\in B} M_X^\top, \mathbf{g}_y. \tag{15} $$

---

## Background 3: Apply multiple convolution filters

Apply (n_f=3) filters (F_1,F_2,F_3) efficiently by concatenating them:

$$ H = M_X, F_{\text{all}}, \quad F_{\text{all}} = [\operatorname{vec}(F_1),\operatorname{vec}(F_2),\operatorname{vec}(F_3)] \in \mathbb{R}^{8\times 3}. \tag{16–17} $$

Gradients:

$$ \frac{\partial L}{\partial F_{\text{all}}} = M_X^\top, G, \quad G = [\mathbf{g}_1,\mathbf{g}_2,\mathbf{g}_3] \in \mathbb{R}^{4\times 3}. \tag{18–19} $$

---

## Background 4: Back‑prop equations

Let the mini‑batch loss be

$$ L(B,\Theta) = \frac{1}{|B|}\sum_{(x,y)\in B} \ell_{\text{cross}}\bigl(y, f_{\text{network}}(X,\Theta)\bigr), \quad \Theta={F_{\text{all}}, W_1, W_2}. \tag{20} $$

Given the network outputs for the batch, the gradients w.r.t. (W_1) and (W_2) are as in Assignment 2. The novel quantity is the gradient w.r.t. (F_{\text{all}}). For one input, use (18). For the batch:

$$ \frac{\partial L(B)}{\partial F_{\text{all}}} = \frac{1}{|B|}\sum_{(X,y)\in B} M_X^\top, G_y. \tag{21} $$

In code, store the per‑image (M_X) as a 3D array (M \in \mathbb{R}^{n_p \times 3f^2 \times n}) and the per‑image gradients for the convolution outputs as (G \in \mathbb{R}^{n_p \times n_f \times n}). Then

$$ F^{\text{grad}}*{\text{all}} = \frac{1}{n}\sum*{i=1}^n M(:,:,i)^\top, G(:,:,i) \in \mathbb{R}^{3f^2 \times n_f}. \tag{22} $$

---

## Background 5: Label smoothing

As models get larger, more regularization helps. **Label smoothing** spreads a small probability mass from the ground‑truth class to others. For label (y\in{1,\ldots,K}), let (\mathbf{y}) be the one‑hot vector. For (\varepsilon\in[0,1)), define

$$
\bigl(\mathbf{y}^{\text{smooth}}\bigr)_i =
\begin{cases}
1-\varepsilon, & i=y,\
\varepsilon/(K-1), & i\ne y.
\end{cases} \tag{23}
$$

In back‑prop through softmax‑cross‑entropy, replace

$$ -(\mathbf{y}-\mathbf{p}) ;\Rightarrow; -\bigl(\mathbf{y}^{\text{smooth}}-\mathbf{p}\bigr). \tag{24} $$

---

## Background 6: Cyclical learning rates with increasing step sizes

Use cyclical learning rates (as in Assignment 2) but **double** the cycle length after each cycle:

$$ n_{i+1,s} = 2, n_{i,s}. \tag{25} $$

This approximates cosine annealing with warm restarts (Loshchilov & Hutter, 2017). (In the basic assignment, do **not** decay (\eta_{\max}) across cycles.)

---

## Exercise 1: Implement the convolution efficiently

Write a slow, straightforward dot‑product implementation first (nested loops over non‑overlapping sub‑patches) to verify correctness using provided debug data.

**Debug file:** `debug_conv_info.npz`

```python
load_data = np.load('debug_conv_info.npz')
X  = load_data['X']                   # shape: 3072 x n (n=5)
Fs = load_data['Fs']                  # shape: f x f x 3 x n_f (f=8, n_f=2)
```

Reshape flattened images:

```python
X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
```

For each image `X_ims[:,:,:,i]` and filter `F[:,:,:,k]`, compute dot‑products over each (f\times f\times 3) sub‑patch. Store outputs in an array of shape ((32/f) \times (32/f) \times n_f \times n). Compare to `load_data['conv_outputs']` for correctness.

Once correct, implement the **matrix‑multiplication** version. Build `MX ∈ R^{n_p × (f*f*3) × n}` once at the start:

```python
MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')
```

Flatten filters:

```python
Fs_flat = Fs.reshape((f*f*3, n_f), order='C')
```

Compute convolutions per image by matmul and compare with the loop version (after reshaping):

```python
for i in range(n):
    conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)

conv_outputs_flat = conv_outputs.reshape((n_p, n_f, n), order='C')
```

Finally, remove the image loop with **Einstein summation** for speed:

```python
conv_outputs_mat = np.einsum('ijn, jl -> iln', MX, Fs_flat, optimize=True)
```

> On batches of ~100 images, `einsum` can give >3× speed‑ups on an M1 MacBook Pro.

---

## Exercise 2: Compute gradients

Write the forward pass for Eqs. (1)–(5). After convolution and ReLU, reshape to feed the first FC layer:

```python
conv_flat = np.fmax(conv_outputs_mat.reshape((n_p*n_f, n), order='C'), 0)
```

Use the provided parameters to debug (sizes in parentheses):

* `W1` ((n_h \times n_p n_f)), `W2` ((10 \times n_h))
* `b1` ((n_h \times 1)), `b2` ((10 \times 1))
* Forward intermediates to match: `conv_flat` ((n_p n_f \times n)), `X1` ((n_h \times n)), `P` ((10 \times n))

Backward pass: given labels `Y` ((10\times n)), back‑prop through the FC layers (as in Assignment 2). Let `G_batch ∈ R^{n_p n_f × n}` be the gradient at the flattened conv output. Undo the reshape:

```python
GG = G_batch.reshape((n_p, n_f, n), order='C')
```

Compute the gradient for `Fs_flat` using (22):

```python
MXt = np.transpose(MX, (1, 0, 2))
grad_Fs_flat = np.einsum('ijn, jln -> il', MXt, GG, optimize=True)
```

**Clean up & integrate:**

* Pre‑compute, save, and load `MX` for training and test data.
* Initialize parameters (use **He initialization** for conv filters and FC layers).
* Use `float32` to reduce memory.
* Store conv filters as an array of shape `(3*f*f, n_f)` to match the matrix form.

**Gradient checks:**

* Re‑implement your network in **PyTorch** (CPU), but compute the conv with the **loop** version to avoid ordering issues with `einsum`.
* Compare your numpy gradients to PyTorch’s on a small subset (small (n_f), small batch).
* Then **add a conv bias** vector (size (n_f)) and **L2 regularization** (on FC weights and conv filters). Re‑check gradients.

---

## Exercise 3: Train small networks with cyclic learning rates

Train with cyclical learning rates as in Assignment 2.

**Initial architecture:** (f=4,; n_f=10,; n_h=50)

**Hyper‑parameters:**

* Cyclic LR: `n_cycles=3`, `step=800`, `eta_min=1e-5`, `eta_max=1e-1`, `n_batch=100`
* Regularization: `lam=0.003`

With 49,000 training examples, this model achieves ~**57.61%** test accuracy in under ~50s on an M1 MacBook Pro (see Figure 2 for curves).

> **Figure 2 (conceptual):** Loss and accuracy curves for the initial ConvNet ((f=4, n_f=10, n_h=50)). Final test accuracy: 57.61%.

### Short runs to compare (f) and (n_f)

Train the following (keeping the conv output dimensionality roughly constant):

1. **Arch 1:** (f=2,; n_f=3,; n_h=50)
2. **Arch 2:** (f=4,; n_f=10,; n_h=50) *(already trained)*
3. **Arch 3:** (f=8,; n_f=40,; n_h=50)
4. **Arch 4:** (f=16,; n_f=160,; n_h=50)

Record final test accuracy and training time for each. Make two **bar charts**: (a) test accuracy, (b) run time.

### Train for longer

`f=16` is too slow; `f=2` limits the conv effect. Focus on `f ∈ {4,8}`.

Upgrade to **increasing cycle length** (Background 6): start with `step1 = 800`, keep `n_cycles=3`. Train **Arch 2** and **Arch 3** longer; record final test accuracies and loss plots. At least one should exceed **60%**.

To test **width**, increase **Arch 2** to (n_f=40) and re‑run; increasing width generally helps (see EfficientNet scaling ideas; Tan & Le, 2019).

---

## Exercise 4: Larger networks & regularization with label smoothing

Train a wider network:

* **Arch 5:** (f=4,; n_f=40,; n_h=300)
* Train with `n_cycles=4`, `step1=800`, `lam=0.0025`.

You’ll likely see **loss over‑fitting**. Implement **label smoothing** (Background 5) and re‑run with the same setup. Save loss plots and final test accuracy; comment qualitatively on the difference between **no smoothing** vs **smoothing**.

---

## To complete the assignment (deliverables)

Upload to Canvas:

1. **Code** for the assignment in a single file.
2. A **brief PDF report** including:

   1. How you checked analytic gradients; evidence they’re bug‑free. Report training time for the initial three‑layer ConvNet in Exercise 3.
   2. **Bar charts** of final test accuracy and training times for the 4 architectures in Exercise 3 (short runs).
   3. **Curves** for training/test loss for the **Train for longer** part of Exercise 3 (compute metrics sparsely every (j\cdot\text{step}/2)).
   4. **Curves** for training/test loss for **Arch 5** with and without **label smoothing**; qualitative comparison.
   5. Your proposed **next experiments** to further boost performance given limited (but increased) compute—focus on changes most likely to improve final test accuracy.

---

## Exercise 5 (Optional, bonus points)

### 5.1 Push performance (up to 4 points)

Explore: make the network wider, data augmentation, tune L2 vs label smoothing, decay (\eta_{\max}) across cycles, concatenate multi‑scale conv outputs, etc. A reported (non‑exhaustive) maximum was **67.26%** using tricks from the assignment. Extra bonus: **≥68%** (+1), **≥70%** (+2). (Using the test set as validation is allowed *only* for this bonus exercise.)

### 5.2 Compare training speed with PyTorch (up to 3 points)

Use `torch.nn.functional.conv2d` and autograd on CPU. Compare training timings vs your implementation across a few filter widths and numbers of filters; summarize results and trends.

**Bonus deliverables:** code + a short PDF summarizing best accuracy & improvements (5.1) and timing comparisons & conclusions (5.2).

---

## References

* **Loshchilov, I., & Hutter, F. (2017).** SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.
* **Tan, M., & Le, Q. (2019).** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
