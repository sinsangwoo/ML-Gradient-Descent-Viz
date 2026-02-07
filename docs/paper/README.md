# Publication-Ready LaTeX Documentation

## Overview

This directory contains publication-quality LaTeX documentation for the Optimization Primitive Library. The documentation includes:

- **Complete mathematical proofs** with Theorem-Proof style
- **Algorithm pseudocode** in algorithmic environment
- **Complexity analysis** (time/space)
- **Experimental validation** with tables and figures
- **Numerical stability analysis**

---

## File Structure

```
docs/paper/
├── main.tex                  # Main document with introduction and preliminaries
├── algorithms.tex            # Adam, AdamW, and complexity comparisons
├── experiments.tex           # Experimental results and benchmarks
├── numerical_analysis.tex    # Floating-point and numerical stability
├── conclusion.tex            # Conclusion and future work
├── references.bib            # BibTeX bibliography
├── README.md                 # This file
├── Makefile                  # Build automation
└── compile.sh                # Compilation script
```

---

## Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Windows
# Install MiKTeX from https://miktex.org/
```

### Compile to PDF

**Option 1: Using Makefile**
```bash
make          # Compile once
make clean    # Remove auxiliary files
make full     # Full compilation with bibliography
```

**Option 2: Using compile script**
```bash
chmod +x compile.sh
./compile.sh
```

**Option 3: Manual compilation**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex  # Second pass for cross-references
```

### Output

The compiled PDF will be: `main.pdf`

---

## Document Structure

### 1. Introduction (`main.tex`)
- Problem formulation
- Key assumptions (L-smoothness, strong convexity)
- Notation and definitions
- Gradient Descent with proof
- Momentum method

### 2. Accelerated Methods (`main.tex`)
- Nesterov Accelerated Gradient
- $O(1/k^2)$ convergence proof
- Optimality discussion

### 3. Adaptive Methods (`algorithms.tex`)
- AdaGrad with regret bound proof
- RMSProp
- Adam with complete algorithm
- AdamW (decoupled weight decay)
- Complexity comparison table

### 4. Experiments (`experiments.tex`)
- MNIST benchmark results
- High-dimensional scalability
- Extreme conditioning tests
- GPU acceleration results
- Numerical stability analysis

### 5. Numerical Analysis (`numerical_analysis.tex`)
- Floating-point precision
- Condition number effects
- Catastrophic cancellation
- Implementation best practices
- Parallel implementation

### 6. Conclusion (`conclusion.tex`)
- Summary of contributions
- Performance comparison
- Future work
- Lessons learned

---

## Key Features

### Theorem-Proof Style

```latex
\begin{theorem}[GD Convergence Rate]\label{thm:gd_convergence}
Suppose $f$ satisfies Assumptions~\ref{assum:smooth} and~\ref{assum:convex}. 
If $\eta = \frac{1}{L}$, then:
\begin{equation}
\norm{\theta_k - \theta^*}^2 \leq \rho^k \norm{\theta_0 - \theta^*}^2
\end{equation}
where $\rho = \frac{\kappa - 1}{\kappa + 1} < 1$.
\end{theorem}

\begin{proof}
[Detailed mathematical proof...]
\end{proof}
```

### Algorithm Pseudocode

```latex
\begin{algorithm}[H]
\caption{Adam (Adaptive Moment Estimation)}
\label{alg:adam}
\begin{algorithmic}[1]
\Require Initial point $\theta_0$, step size $\alpha$
\State $m_0 \gets 0$ (first moment)
\State $v_0 \gets 0$ (second moment)
\For{$k = 1, 2, \ldots, K$}
    \State $g_k \gets \grad f(\theta_{k-1})$
    \State $m_k \gets \beta_1 m_{k-1} + (1-\beta_1) g_k$
    \State ...
\EndFor
\end{algorithmic}
\end{algorithm}
```

### Complexity Analysis

```latex
\begin{remark}[Complexity]
\textbf{Time Complexity}: $O(d)$ per iteration \\
\textbf{Space Complexity}: $O(3d)$ (parameters + moments) \\
\textbf{Iterations to $\epsilon$-accuracy}: $O(\sqrt{K})$ regret
\end{remark}
```

### Experimental Tables

```latex
\begin{table}[h]
\centering
\caption{MNIST Performance}
\begin{tabular}{lccc}
\toprule
\textbf{Optimizer} & \textbf{Time (s)} & \textbf{Final Loss} & \textbf{Test MSE} \\
\midrule
Adam      & 2.8 $\pm$ 0.1 & 8.5e-5 & 1.8e-4 \\
AdamW     & 2.9 $\pm$ 0.1 & 7.2e-5 & 1.6e-4 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Customization

### Adding New Algorithms

1. Add algorithm in `algorithms.tex`:
```latex
\subsection{Your New Optimizer}

\begin{algorithm}[H]
\caption{Your Optimizer}
\begin{algorithmic}[1]
% Your algorithm here
\end{algorithmic}
\end{algorithm}

\begin{theorem}[Convergence]
% Your convergence theorem
\end{theorem}

\begin{proof}
% Your proof
\end{proof}
```

2. Add experiments in `experiments.tex`
3. Recompile: `make`

### Adding Figures

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/convergence_plot.pdf}
\caption{Convergence comparison on MNIST}
\label{fig:convergence}
\end{figure}
```

### Adding References

Edit `references.bib`:
```bibtex
@article{kingma2015adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={ICLR},
  year={2015}
}
```

Cite in text:
```latex
The Adam optimizer \cite{kingma2015adam} combines...
```

---

## Mathematical Notation

The document uses consistent notation:

| Symbol | Meaning |
|--------|----------|
| $\R$ | Real numbers |
| $\theta$ | Parameters |
| $\theta^*$ | Optimal parameters |
| $f(\theta)$ | Objective function |
| $\grad f$ | Gradient |
| $\norm{\cdot}$ | Euclidean norm |
| $\inner{\cdot}{\cdot}$ | Inner product |
| $L$ | Lipschitz constant |
| $\mu$ | Strong convexity parameter |
| $\kappa$ | Condition number ($L/\mu$) |
| $\eta$ | Learning rate |
| $g_k$ | Gradient at iteration $k$ |
| $m_k$ | First moment (Adam) |
| $v_k$ | Second moment (Adam) |

---

## Best Practices

### 1. Reproducible Builds

```bash
# Use exact LaTeX version
pdflatex --version  # Should be TeXLive 2023+

# Clean build from scratch
make clean
make full
```

### 2. Version Control

```bash
# Track only source files
git add *.tex *.bib Makefile README.md
git add compile.sh

# Ignore build artifacts
echo "*.aux" >> .gitignore
echo "*.log" >> .gitignore
echo "*.pdf" >> .gitignore
```

### 3. Collaborative Editing

- Use [Overleaf](https://www.overleaf.com/) for online collaboration
- Or use [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) in VS Code

### 4. Mathematical Typesetting

```latex
% Good: Display equations
\begin{equation}
\theta_{k+1} = \theta_k - \eta \grad f(\theta_k)
\end{equation}

% Avoid: Inline complex equations
$\theta_{k+1} = \theta_k - \eta \grad f(\theta_k)$ (too long)

% Good: Align multiple equations
\begin{align}
\theta_{k+1} &= \theta_k - \eta g_k \\
&= \theta_k - \frac{\eta}{\sqrt{v_k}} g_k
\end{align}
```

---

## Troubleshooting

### Missing Packages

```bash
# Ubuntu/Debian
sudo apt-get install texlive-science  # For algorithm packages
sudo apt-get install texlive-fonts-extra

# macOS (included in MacTeX)

# Windows MiKTeX
# Packages install automatically on first use
```

### Bibliography Not Working

```bash
# Full compilation sequence
pdflatex main.tex
bibtex main       # Note: no .tex extension
pdflatex main.tex
pdflatex main.tex
```

### Cross-References Not Resolved

```bash
# Need two passes
pdflatex main.tex
pdflatex main.tex  # Second pass resolves \ref{} and \cite{}
```

### Overfull/Underfull Boxes

```latex
% Allow slightly more flexibility
\sloppy

% Or adjust specific paragraphs
\begin{sloppypar}
Long paragraph with technical terms...
\end{sloppypar}
```

---

## Converting to Other Formats

### To HTML (for web)

```bash
# Using pandoc
pandoc main.tex -o main.html --mathjax

# Using latex2html
latex2html main.tex
```

### To DOCX (for Word)

```bash
pandoc main.tex -o main.docx
```

### To arXiv-compatible ZIP

```bash
# Include all source files
zip arxiv.zip *.tex *.bib *.bbl figures/*.pdf
```

---

## Page Count

Expected final document:
- **Main paper**: ~30-35 pages
- **With appendices**: ~40-45 pages

Breakdown:
- Introduction: 3 pages
- Preliminaries: 2 pages
- GD & Momentum: 4 pages
- Nesterov: 3 pages
- Adaptive methods: 6 pages
- Experiments: 8 pages
- Numerical analysis: 6 pages
- Conclusion: 4 pages
- References: 2 pages

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sin2025optimization,
  author = {Sangwoo Sin},
  title = {Optimization Primitive Library: Mathematical Foundations and Scalable Implementation},
  year = {2025},
  url = {https://github.com/sinsangwoo/ML-Gradient-Descent-Viz}
}
```

---

## License

MIT License - Free to use, modify, and distribute with attribution.

---

## Contact

For questions or suggestions:
- **Email**: aksrkd7191@gmail.com
- **GitHub**: https://github.com/sinsangwoo/ML-Gradient-Descent-Viz
- **Issues**: https://github.com/sinsangwoo/ML-Gradient-Descent-Viz/issues

---

*Publication-ready documentation with mathematical rigor.*
