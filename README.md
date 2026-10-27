# Activation Patching & Logit Lens on GPT-2

A hands-on exploration of **Mechanistic Interpretability (Mech. Interp.)** using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). This project investigates how information flows through GPT-2 and how localized interventions affect model behavior.

The goal is to identify which layers and components contribute to specific token predictions and to understand how localized interventions affect model behavior.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Experimental Setup](#-experimental-setup)
- [Baseline Findings](#-baseline-findings)
- [Core Insights](#-core-insights)
- [Methodology](#-methodology)
- [Getting Started](#️-getting-started)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🔍 Project Overview

This repository explores three complementary techniques:

- **Activation Patching:** Intervene on internal activations to investigate which components causally influence a model's output.
- **Logit Lens:** Project intermediate residual stream representations into vocabulary space to examine how token predictions evolve across layers.
- **Direct Logit Attribution (DLA):** Measure how individual attention heads and other components contribute to a target token's logit.

The experiments use clean and corrupted prompts to investigate how GPT-2 represents semantic information and how attention and MLP components contribute to the final prediction.

<p align="center">
  <svg width="960" height="300" viewBox="0 0 960 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram showing activation patching and logit lens experiments on GPT-2">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0f172a"/>
        <stop offset="100%" stop-color="#111827"/>
      </linearGradient>
      <linearGradient id="cleanGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#86efac"/>
        <stop offset="100%" stop-color="#4ade80"/>
      </linearGradient>
      <linearGradient id="corrGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#f9a8d4"/>
        <stop offset="100%" stop-color="#f472b6"/>
      </linearGradient>
      <linearGradient id="patchGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#93c5fd"/>
        <stop offset="100%" stop-color="#60a5fa"/>
      </linearGradient>
      <style>
        .title { font: 700 22px sans-serif; fill: #e5e7eb; }
        .label { font: 600 15px sans-serif; fill: #e5e7eb; }
        .small { font: 500 12px sans-serif; fill: #cbd5e1; }
        .box { stroke: #94a3b8; stroke-width: 2; fill: rgba(15,23,42,0.75); }
        .clean { fill: url(#cleanGrad); opacity: 0.92; }
        .corr { fill: url(#corrGrad); opacity: 0.92; }
        .patch { fill: url(#patchGrad); opacity: 0.92; }
        .arrow { stroke: #cbd5e1; stroke-width: 3; fill: none; marker-end: url(#arrowHead); }
      </style>
      <marker id="arrowHead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L0,6 L9,3 z" fill="#cbd5e1"/>
      </marker>
    </defs>

    <rect x="0" y="0" width="960" height="300" fill="url(#bg)" rx="18"/>

    <text x="480" y="34" text-anchor="middle" class="title">GPT-2: Activation Patching &amp; Logit Lens</text>

    <rect x="40" y="75" width="150" height="110" rx="12" class="box clean"/>
    <text x="115" y="100" text-anchor="middle" class="label">Clean prompt</text>
    <text x="115" y="125" text-anchor="middle" class="small">My dog is a good</text>
    <text x="115" y="160" text-anchor="middle" class="small">Residual stream</text>

    <rect x="260" y="75" width="170" height="110" rx="12" class="box"/>
    <text x="345" y="105" text-anchor="middle" class="label">Transformer layers</text>
    <text x="345" y="130" text-anchor="middle" class="small">Layer-wise features</text>
    <text x="345" y="155" text-anchor="middle" class="small">attention + MLP</text>

    <rect x="500" y="75" width="170" height="110" rx="12" class="box corr"/>
    <text x="585" y="100" text-anchor="middle" class="label">Corrupted prompt</text>
    <text x="585" y="125" text-anchor="middle" class="small">My cat is a good</text>
    <text x="585" y="160" text-anchor="middle" class="small">Competing logits</text>

    <rect x="710" y="75" width="210" height="110" rx="12" class="box patch"/>
    <text x="815" y="100" text-anchor="middle" class="label">Patched activations</text>
    <text x="815" y="125" text-anchor="middle" class="small">Swap clean residuals</text>
    <text x="815" y="160" text-anchor="middle" class="small">Measure recovery</text>

    <path d="M190 130 H260" class="arrow"/>
    <path d="M430 130 H500" class="arrow"/>
    <path d="M670 130 H710" class="arrow"/>

    <rect x="240" y="215" width="480" height="48" rx="12" class="box"/>
    <text x="480" y="245" text-anchor="middle" class="label">Logit Lens / DLA / Causal intervention</text>

    <text x="480" y="286" text-anchor="middle" class="small">Visual summary for interpreting how specific components shift final token predictions.</text>
  </svg>
</p>

This diagram is a working draft to visualize the core experimental flow: compare a clean and corrupted prompt, isolate layerwise feature contributions, and patch activations to quantify recovery in the final prediction.

---

## 📁 Repository Structure

```text
activation-patching-gpt2/
├── notebooks/
│   ├── Activation_Patching_Layer_Specific.ipynb
│   └── Activation_Patching_Model_Pathways.ipynb
├── src/
│   └── patching_sweep.py
├── requirements.txt
├── LICENSE
└── README.md
```

### Notebook Descriptions

| File | Description |
|------|-------------|
| `notebooks/Activation_Patching_Layer_Specific.ipynb` | Investigates residual stream dynamics in `gpt2-small` using `run_with_cache` and a clean-versus-corrupted prompt experiment. |
| `notebooks/Activation_Patching_Model_Pathways.ipynb` | Investigates component-level causal contributions in `gpt2-medium`, comparing Layer 4 attention and MLP activations. |
| `src/patching_sweep.py` | Draft CLI utility intended to automate activation-patching experiments across layers and token positions. |

---

## 🧪 Experimental Setup

The experiments use a simple prompt-corruption setup to investigate how changing a single word affects the model's internal representations.

**Clean prompt**

```text
My dog is a good
```

**Corrupted prompt**

```text
My cat is a good
```

The objective is to understand how replacing `dog` with `cat` affects the residual stream, intermediate token predictions, and downstream model behavior.

Activation patching allows activations from the clean run to be substituted into the corrupted run at selected layers or components. The resulting change in the target metric helps identify which internal pathways are most responsible for the final output.

---

## 📊 Baseline Findings

The following table summarizes the initial observations from the notebook experiments.

| Technique | Component | Result | Interpretation |
|-----------|-----------|--------|----------------|
| Logit Lens | Layer 3 residual stream | Top tokens: `'enough'`, `' luck'` | The target token has not yet emerged among the strongest predictions at this depth. |
| Direct Logit Attribution | Layer 10, Head 10.0 | **+2.70** | Positively contributes to the target token's logit. |
| Direct Logit Attribution | Layer 10, Head 10.7 | **−0.83** | Negatively contributes to the target token's logit. |
| Activation Patching | Layer 4 Attention (`attn_out`) | **1.657%** recovery | The tested intervention produces limited standalone recovery. |
| Activation Patching | Layer 4 MLP (`mlp_out`) | **0.588%** recovery | The tested intervention produces less recovery than the attention intervention. |

> [!NOTE]
> Under the tested intervention, Layer 4 attention produces greater measured causal recovery than the MLP. This result is specific to the experimental setup and does not establish a general ranking of component importance across all prompts or tasks.

These measurements are preliminary experimental findings and should be interpreted in the context of the prompts, model configurations, patching positions, and metric definitions used in the notebooks.

---

## 🔬 Core Insights

### 1. Logit Baseline Shifts and LayerNorm

One counterintuitive observation is that the correct token can receive a **higher** raw logit under a corrupted prompt than under the clean prompt.

For example, the token `' dog'` was observed to receive the following logits:

| Run | Logit for `' dog'` |
|-----|--------------------|
| Clean | 13.741 |
| Corrupted | 13.792 |

Although the prompt changes from `"My dog is a good"` to `"My cat is a good"`, the raw logit for `' dog'` increases slightly.

This illustrates why raw logits should not be interpreted as direct measurements of semantic correctness or causal importance.

**Potential factors include:**

- **Residual stream representations:** The clean and corrupted prompts produce different combinations of semantic and syntactic features.
- **Layer normalization:** GPT-2 applies final layer normalization before the unembedding operation, changing the scale and distribution of the representation.
- **Shared contextual features:** Both `' dog'` and `' cat'` may align with features associated with the continuation pattern *"is a good [noun]"*.

### Layer Normalization (LN)

Layer normalization is applied to the final residual stream before the unembedding step. For a residual vector $x \in \mathbb{R}^{d}$, it is defined as:

$$ \mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^{2} + \epsilon}} + \beta $$

#### Mathematical Formulations

* **Mean** ($\mu$) — Computed across the $d$ features of the vector:
  $$ \mu = \frac{1}{d} \sum_{i=1}^{d} x_i $$

* **Variance** ($\sigma^{2}$) — Computed across the $d$ features of the vector:
  $$ \sigma^{2} = \frac{1}{d} \sum_{i=1}^{d} \left( x_i - \mu \right)^{2} $$

#### Parameter Definitions

* $\gamma \in \mathbb{R}^{d}$ — Learned scale (elementwise gain).
* $\beta \in \mathbb{R}^{d}$ — Learned bias.
* $\epsilon$ — Small constant for numerical stability.
* $\odot$ — Elementwise multiplication (Hadamard product).

The final logits are then obtained by projecting the normalized residual stream through the unembedding matrix:

$$
\mathrm{logits} = \mathrm{LN}\!\left(x_{\text{final}}\right) W_U + b_U
$$

where $W_U$ is the unembedding matrix and $b_U$ is the unembedding bias (zero for GPT-2 in TransformerLens).

Because the normalization divides by $\sqrt{\sigma^{2} + \epsilon}$, any change to the residual stream can rescale every vocabulary logit at once. This is one reason a single token's raw logit can move in a counterintuitive direction.

> **Key takeaway:** A token's raw logit reflects the combined effects of many features. A corrupted prompt can therefore increase a token's logit even when the semantic context becomes less appropriate.

### 2. Semantic Features vs. Syntactic Continuity

The residual stream acts as a shared information pathway through which different layers and components communicate.

Its representations can contain features associated with semantics, syntax, and contextual patterns simultaneously.

In the prompt-corruption experiment:

- The corrupted prompt changes the semantic information associated with the original word.
- Features associated with the canine concept may receive different activation patterns.
- Syntactic and contextual features can continue to support `' dog'` as a plausible continuation.

Consequently, the final logit may remain high even when the original semantic context has been disrupted.

> **Key takeaway:** Token predictions emerge from the interaction of multiple learned features rather than a single isolated semantic representation.

### 3. Why Logit Difference Matters

Absolute logits can be misleading when evaluating localized interventions because changes in the overall representation can affect multiple vocabulary logits simultaneously.

**Notation.** Let $z_v(p)$ denote the logit assigned to vocabulary token $v$ at the final position of prompt $p$. The clean prompt is $p_{\text{clean}}$ and the corrupted prompt is $p_{\text{corr}}$.

#### Target logit shift

A first metric is the change in the target token's logit between the two runs:

$$
\Delta z_t = z_t\!\left(p_{\text{clean}}\right) - z_t\!\left(p_{\text{corr}}\right)
$$

where $t$ denotes the target token. For the `' dog'` example above:

$$
\Delta z_{\texttt{' dog'}} = 13.741 - 13.792 = -0.051
$$

The sign is negative, so the corrupted run assigns `' dog'` a slightly *higher* raw logit. This is the counterintuitive result described in Insight 1, and it shows why the raw target logit alone is a poor indicator of semantic fidelity.

#### Target-versus-foil logit difference

For experiments that compare competing token predictions, a **target-versus-foil logit difference** is a more informative metric:

$$
D(p) = z_t(p) - z_f(p)
$$

where $t$ is the target token and $f$ is a competing token, or *foil*.

This quantity is the log-odds between the two tokens. Since the softmax normalizer cancels in the ratio of probabilities:

$$
D(p) = \log \frac{P(t \mid p)}{P(f \mid p)}
$$

Tracking how $D$ changes during activation patching helps assess whether an intervention restores the model's preference for the intended target.

#### Why shared shifts cancel

Suppose a change to the residual stream shifts every logit by the same constant $c$, so that $z_v \to z_v + c$ for all $v$. The difference is unaffected:

$$
D' = \left(z_t + c\right) - \left(z_f + c\right) = z_t - z_f = D
$$

The raw target logit $z_t$, however, changes by the full amount $c$.

> [!IMPORTANT]
> Logit differences do not automatically eliminate every baseline effect. They control for shared **additive** shifts only when those shifts affect the compared logits equally. A multiplicative rescaling or a more complex representation change can still distort the comparison.

#### Measuring recovery under patching

Let $D_{\text{clean}}$, $D_{\text{corr}}$, and $D_{\text{patched}}$ be the target-versus-foil logit difference on the clean run, the corrupted run, and the corrupted run with a clean activation patch applied. Then:

$$
\mathrm{Recovery} = \frac{D_{\text{patched}} - D_{\text{corr}}}{D_{\text{clean}} - D_{\text{corr}}} \times 100\%
$$

A value of $0\%$ means the patch had no effect relative to the corrupted run, and $100\%$ means the patch fully restored the clean-run behavior.

> **Key takeaway:** Relative logit metrics provide a more interpretable basis for comparing model behavior across clean, corrupted, and patched runs.

---

## 🧠 Methodology

The notebooks follow a component-level investigation workflow:

1. Run the clean prompt and cache intermediate activations.
2. Run the corrupted prompt and cache its activations.
3. Inspect intermediate representations using the Logit Lens.
4. Analyze component contributions using Direct Logit Attribution.
5. Patch selected activations from the clean run into the corrupted run.
6. Measure behavioral recovery using logit-based metrics.
7. Compare interventions across layers, token positions, and component types.

This workflow connects descriptive analysis of internal representations with causal interventions designed to test their influence on model outputs.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3
- Git
- JupyterLab

### 1. Clone the repository

```bash
git clone https://github.com/fahdev789/activation-patching-gpt2.git
cd activation-patching-gpt2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it using the appropriate command for your operating system.

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch JupyterLab

```bash
jupyter lab
```

Open either notebook from the `notebooks/` directory to explore the experiments.

---

## 🚀 Future Work

Potential extensions to this project include:

- **Automated activation-patching sweeps:** Expand `src/patching_sweep.py` to evaluate interventions across layers and token positions.
- **Component-level comparisons:** Compare attention heads, attention outputs, and MLP outputs under consistent experimental conditions.
- **Recovery visualizations:** Plot causal recovery across layers and token positions to identify potential circuit bottlenecks.
- **Metric validation:** Compare absolute target logits, target-versus-foil logit differences, and probability-based measurements.
- **Prompt generalization:** Repeat experiments with additional prompts to test whether observed patterns generalize beyond a single example.

---

## 📚 References

- [TransformerLens — Mechanistic Interpretability Library](https://github.com/TransformerLensOrg/TransformerLens)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Neel Nanda's Mechanistic Interpretability Resources](https://www.neelnanda.io/mechanistic-interpretability)

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
