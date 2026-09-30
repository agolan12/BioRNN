# BioRNN

Experiments on the vanishing/exploding gradient problem in recurrent networks,
comparing vanilla (Elman) RNNs with leaky, neuroscience-style RNNs, plus
reproductions of biologically motivated RNN papers.

## Layout

```
src/biornn/            reusable code (importable as `biornn`)
  models/rnn.py        VanillaRNNCell, AddingRNN, CopyRNN, set_spectral_radius, total_grad_norm
  models/leaky_rnn.py  LeakyRNNCell, LeakyAddingRNN, LeakyCopyRNN
  models/plastic_rnn.py PlasticRNNCell, PlasticAddingRNN (Hebbian fast weights, Wakhloo et al. 2025)
  tasks.py             adding_problem, frequency_adding_problem, copy_task
  fixes.py             norm_preserving_regularizer
  visualize.py         plotting helpers
notebooks/
  01_vanilla_rnn/        classic Adding Problem + Copy task
  02_frequency_adding/   frequency-encoded Adding Problem + Copy task; 04 = plastic RNN
papers/
  wakhloo2025/         Wakhloo, Clark & Abbott (2025) persistent-oscillation model (JAX/diffrax)
results/               generated figures/checkpoints (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"          # add ",wakhloo" for the paper reproduction
nbstripout --install                   # strip notebook outputs on commit
```

Then, from any notebook:

```python
from biornn.models.rnn import AddingRNN, set_spectral_radius
from biornn.tasks import adding_problem, copy_task
```

## Conventions

- Reusable code goes in `src/biornn/`; notebooks only import and present results.
- New tasks are new functions in `tasks.py`, not copies of the package.
- Notebooks are numbered in reading order: `NN_short_name.ipynb`.
- Lowercase, underscore-separated names; no spaces in paths.
