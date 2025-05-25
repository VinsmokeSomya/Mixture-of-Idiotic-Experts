<h1 align="center" id="top">Mixture of Idiotic Experts 🧠✍️🎭</h1>

<p align="center">
  <strong>Mixture of Idiotic Experts 🧠✍️🎭: Crafting Shakespeare with a Chorus of Experts. Building a Sparse Mixture of Experts language model from scratch!</strong>
</p>

Welcome to **Mixture of Idiotic Experts**! This project is a journey into the world of autoregressive character-level language models, but with a twist: we're using a Sparse Mixture of Experts (MoE) architecture. The goal? To generate Shakespeare-like text, one character at a time, by learning from the Bard's own words.

This implementation is designed for clarity and hackability, making it a great playground for understanding how MoE models work under the hood.

## 📜 Table of Contents

1.  [💡 The Spark: From "Shakespeare's Synaptic Scribes" to "Mixture of Idiotic Experts"](#inspiration)
2.  [📂 What's Inside? (Project Structure)](#project-structure)
3.  [🧠 Under the Hood: The "Mixture of Idiotic Experts" Architecture](#architecture-deep-dive)
    *   [🚪 The Gating Network: Choosing the Wisest Experts](#gating-mechanism)
    *   [🧙‍♂️ The Expert Ensemble: Specialized Language Wizards](#the-experts)
    *   [🔍 Causal Self-Attention: Remembering What's Been Said](#self-attention)
    *   [⚖️ Expert Capacity: Keeping Training Efficient](#expert-capacity)
    *   [⚙️ Workflow Diagram](#workflow-diagram)
4.  [🛠️ Tech Corner (Tech Stack)](#tech-stack)
5.  [🚀 Let's Get Makin'! (Getting Started)](#getting-started)
    *   [1. Prerequisites](#prerequisites)
    *   [2. Clone the Repository](#clone-repository)
    *   [3. Set Up a Virtual Environment (Recommended)](#set-up-virtual-environment)
    *   [4. Install Dependencies](#install-dependencies)
6.  [🏃 Running the Show (How to Run)](#how-to-run)
    *   [🎓 The Core Script: `idiotic_experts.py`](#run-script)
    *   [🎓 The Learning Journey: Jupyter Notebooks](#run-notebooks)
7.  [📜 What to Expect (Output & Behavior)](#expected-output)

---

<h2 id="inspiration">💡 The Spark: Of "Mixture of Idiotic Experts"</h2>

This project is heavily inspired by (and largely based on) Andrej Karpathy's amazing [makemoe](https://github.com/karpathy/makemore) project. We've taken the core ideas of an autoregressive character-level language model and infused it with a **Sparse Mixture of Experts (MoE)** architecture.

**The MoE Twist:**
*   Instead of a single, monolithic feed-forward network, `Mixture of Idiotic Experts` uses multiple smaller "expert" networks.
*   A "gating mechanism" intelligently selects a subset of these experts for each input token.
*   This allows the model to scale its capacity (total number of parameters) while keeping the computation for each token relatively low.

**Key Differences from "Shakespeare's Synaptic Scribes":**
*   **Architecture:** Sparse Mixture of Experts replaces the standard MLP.
*   **Gating:** Implements Top-k and Noisy Top-k gating.
*   **Initialization:** Uses Kaiming He initialization (but it's hackable!).
*   **Expert Capacity:** Includes mechanisms to balance the load across experts for more efficient training.

**What Remains the Same (from "Shakespeare's Synaptic Scribes"):**
*   **The Task:** Generating Shakespeare-like text from the `input.txt` dataset.
*   **Core Components:** The dataset preprocessing, causal self-attention mechanism, training loop, and inference logic are borrowed and adapted.

<h2 id="project-structure">📂 What's Inside? (Project Structure)</h2>

Here's a map of the "Mixture of Idiotic Experts" kingdom:

```
Mixture of Idiotic Experts/
├── .git/                                  # Git's secret scrolls
├── images/
│   └── makemoelogo.png                    # Our project's banner
├── idiotic_experts.py                     # The heart of the MoE model 🐍
├── input.txt                              # The Bard's own words (our dataset) 📜
├── README.md                              # You are here! 🗺️
└── LICENSE                                # The rules of the land 📄
```

<h2 id="architecture-deep-dive">🧠 Under the Hood: The "Mixture of Idiotic Experts" Architecture</h2>

The `idiotic_experts.py` script is where the core logic resides. It defines the building blocks of our sparse expert model.

<h3 id="gating-mechanism">🚪 The Gating Network: Choosing the Wisest Experts</h3>
A small neural network (the "gate") examines each input token and decides which experts are best suited to process it. It outputs scores, and typically, the top 'k' experts are chosen. This project includes implementations for "Top-k gating" and "noisy Top-k gating" (which adds a bit of randomness during training to help with load balancing).

<h3 id="the-experts">🧙‍♂️ The Expert Ensemble: Specialized Language Wizards</h3>
These are individual feed-forward neural networks. Each expert can learn to specialize in different patterns or aspects of the language. The beauty of MoE is that only the selected experts are activated for a given token, making the process efficient.

<h3 id="self-attention">🔍 Causal Self-Attention: Remembering What's Been Said</h3>
Borrowed from the "Shakespeare's Synaptic Scribes" architecture (and transformers in general!), the causal self-attention mechanism allows the model to look at previous characters in the sequence to predict the next one. This is crucial for understanding context. "Causal" means it only looks backward, not into the future.

<h3 id="expert-capacity">⚖️ Expert Capacity: Keeping Training Efficient</h3>
A common challenge in MoE models is ensuring that experts are utilized effectively – not too overwhelmed, not too idle. "Expert Capacity" mechanisms are implemented to help distribute the workload more evenly among the experts during training, leading to better learning and resource utilization.

<h3 id="workflow-diagram">⚙️ Workflow Diagram</h3>

Here's a high-level ASCII diagram illustrating the data flow through the Mixture of Idiotic Experts model:

```text
                             [ Input Text ]
                                   |
                                   v
                [ Token Embedding + Positional Embedding ]
                                   |
                                   v
                --------------------------------------------------
                |              FOR N_LAYER Blocks:               |
                |                                                |
                |                Input (x_block)                 |
                |                      |                         |
                |                      v                         |
                |         norm1_out = LayerNorm(x_block)         |
                |                      |                         |
                |                      v                         |
                |        +-----------------------------+         |
                |        | Self-Attention (Multi-Head) |         |
                |        | Input: norm1_out            |         |
                |        | Output: sa_out              |         |
                |        +-----------------------------+         |
                |                      |                         |
                |           x_block = x_block + sa_out           |
                |                      |                         |
                |                      v                         |
                |         norm2_out = LayerNorm(x_block)         |
                |                      |                         |
                |                      v                         |
                |      +-----------------------------------+     |
                |      |     Sparse Mixture of Experts     |     |
                |      |                                   |     |
                |      |      [Router (Gating Network)]    |     |
                |      |          Input: norm2_out         |     |
                |      |                 |                 |     |
                |      |      (Selects Top-K Experts)      |     |
                |      |                 |                 |     |
                |      |    +----->[Expert 1]<--------+    |     |
                |      |    |      [Expert 2]         |    |     |
                |      |    |      ...                |    |     |
                |      |    +----->[Expert M_total]<--+    |     |
                |      |                 | (Combine Top-K  |     |
                |      |                 |  Expert Outputs |     |
                |      |                 |  using Gating   |     |
                |      |                 |  Scores)        |     |
                |      |                 v                 |     |
                |      |       [Weighted MoE Output]       |     |
                |      |         Output: moe_out           |     |
                |      +-----------------------------------+     |
                |                      |                         |
                |           x_block = x_block + moe_out          | 
                |                                                |
                --------------------------------------------------
                                       | (x_block becomes input for 
                                       | next layer or final output if last layer)
                                       v
                        [ Final Layer Normalization (ln_f) ] 
                      (Applied to the output of the last block)
                                       |
                                       v
                            [ Linear Layer (lm_head) ]      
                           (Projects to Vocabulary Size)
                                       |
                                       v
                                  [ Softmax ]
                                       |
                                       v
                     [ Predicted Next Token Probabilities ]
```

<h2 id="tech-stack">🛠️ Tech Corner (Tech Stack)</h2>

*   **Python 3.x**
*   **PyTorch:** The primary framework for building and training the neural network.
*   **MLFlow (Optional):** For experiment tracking and logging metrics. The code was developed on Databricks where MLFlow is pre-installed, but you can `pip install mlflow` to use it elsewhere.

<h2 id="getting-started">🚀 Let's Get Makin'! (Getting Started)</h2>

Follow these steps to conjure your own Shakespearean prose.

<h3 id="prerequisites">1. Prerequisites</h3>

*   Python 3.6 or higher.
*   `pip` for installing Python packages.

<h3 id="clone-repository">2. Clone the Repository</h3>

```bash
git clone https://github.com/VinsmokeSomya/Mixture-of-Idiotic-Experts.git
cd Mixture of Idiotic Experts
```

<h3 id="set-up-virtual-environment">3. Set Up a Virtual Environment (Recommended)</h3>

Keep your Python dependencies neat and tidy!

```bash
python -m venv .venv
```
Activate it:
*   **Windows (PowerShell/CMD):**
    ```bash
    .venv\Scripts\activate
    ```
*   **macOS/Linux (bash/zsh):**
    ```bash
    source .venv/bin/activate
    ```
Your prompt should now show `(.venv)`.

<h3 id="install-dependencies">4. Install Dependencies</h3>

The main requirement is PyTorch.
```bash
pip install torch torchvision torchaudio # Adjust if you have specific CUDA needs
```
For the full experience with experiment tracking (optional):
```bash
pip install mlflow
```

<h2 id="how-to-run">🏃 Running the Show (How to Run)</h2>

You have a couple of ways to interact with `Mixture of Idiotic Experts`:

<h3 id="run-script">🎓 The Core Script: `idiotic_experts.py`</h3>
This script contains the full, runnable implementation. You can execute it directly to train the model and generate text.
```bash
python idiotic_experts.py
```
Dive into the script to see command-line arguments for controlling training parameters, model size, etc.

<h3 id="run-notebooks">🎓 The Learning Journey: Jupyter Notebooks</h3>

Given that the .ipynb files have been deleted, this section can be removed or updated to reflect the current state of the project.
If you plan to add notebooks back later, you might want to keep a placeholder. For now, I will remove this section.

<h2 id="expected-output">📜 What to Expect (Output & Behavior)</h2>

When you run the training (either via the script or notebooks):
*   You'll see training loss decrease over iterations/epochs.
*   Periodically, the model will generate sample text. Initially, it will be gibberish.
*   As training progresses, the generated text should start to resemble the style of Shakespearean English found in `input.txt` – forming words, simple sentences, and eventually more coherent (though perhaps still whimsical) passages.

The project prioritizes readability and understanding over achieving state-of-the-art performance, so it's a great sandbox for learning!

---

Happy Hacking, and may your experts be ever so eloquent! 🎉
