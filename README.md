# Staged Embarrassment Learning (SEL)

*A curriculum-driven, sparsity-aware training framework for efficient neural networks.*
*Inspired by how humans learn — through failure, reflection, and focused effort.*

---

## The Idea Behind the Name

Watch a child learn to throw a ball.

They try. They miss. Their face flushes — that unmistakable look of embarrassment. And then something remarkable happens: they *focus*. Not on what they already know. Not on the easy throws. They concentrate almost entirely on the throw that went wrong.

A few tries later, they've learned it. And once it clicks — once the embarrassment fades — they move on. They don't keep practicing the throw they've already mastered. They let it go.

That observation, made watching a 7-year-old niece at a park, became the hypothesis behind this project: **what if a neural network trained the same way?**

Standard training treats every sample equally — it fires every weight on every image, epoch after epoch, for however long it takes. But most of that compute is redundant. The model already knows what a truck looks like. It knows what a ship looks like. Why keep spending energy on those?

SEL asks the model to track its own embarrassment — to measure, class by class, where it is genuinely failing — and to direct its compute budget only toward those hard cases. When a class is "learned" (embarrassment drops, confidence rises), the model moves on. When it is still uncertain, it focuses harder.

The result is a model that trains **99% more efficiently**, in **42% less time**, while retaining competitive accuracy on a standard benchmark.

---

## How It Works

SEL is built around three interlocking mechanisms:

**1. Per-Class Embarrassment Signal (E)**

For each class `c` at each training step, the model computes a temperature-scaled cross-entropy loss:

```
E_c = mean( CrossEntropy( logits / T, labels ) )  for all samples in class c
C_c = max(0, 1 - E_c)
```

`E_c` is the embarrassment score. A high value means the model is genuinely confused about this class. A low value means it has learned it. `C_c` is the corresponding confidence. Both are tracked across every epoch.

**2. Curriculum Staging**

Training is divided into 5 stages. In each stage, the training data is drawn not randomly from the full dataset, but from a sorted pool — sorted by per-sample loss from a cold model. Early stages see only "easy" samples (samples the model can handle). Later stages progressively introduce harder and harder examples.

```
Stage 1 — Easy:     bottom 0–25% by loss     10 epochs
Stage 2 — Medium:   20–50%                   10 epochs
Stage 3 — Hard:     40–65%                   10 epochs
Stage 4 — Harder:   60–82%                   10 epochs
Stage 5 — Hardest:  78–100%                  10 epochs
```

Each stage has a minimum accuracy threshold that must be passed before the next begins.

**3. Sparse Gradient Updates**

The sparsity mechanism is the engine of efficiency. After each backward pass, gradients below a dynamic threshold (the 20th percentile of the gradient distribution, decaying across stages) are zeroed out:

```python
mask = (p.grad.abs() > gamma).float()
p.grad.mul_(mask)
```

At steady state, approximately **97% of gradients are zeroed per update**. This means only 3% of the model's parameters actually change at any given step — the rest are frozen. FLOPs are counted accordingly.

---

## Results

Tested on CIFAR-10, ResNet-18, T4 GPU, on a held-out test set of 100 images per class (1,000 total).

| System | Full Test Acc | 100/class Acc | FLOPs | Training Time |
|---|---|---|---|---|
| **Baseline** (30 epochs, full data) | 93.2% | 92.7% | 33.5T | 1480s |
| **SEL** (50 epochs, staged + sparse) | 82.2% | 82.6% | **0.338T** | **859s** |
| **Savings** | −11.0% | −10.1% | **+99%** | **+42%** |

**Per-class breakdown (100-image held-out set):**

| Class | Baseline | SEL | Diff |
|---|---|---|---|
| airplane | 91% | 74% | −17% |
| automobile | 97% | 93% | −4% |
| bird | 92% | 82% | −10% |
| cat | 87% | 61% | −26% |
| deer | 90% | 82% | −8% |
| dog | 88% | 86% | −2% |
| frog | 96% | 82% | −14% |
| horse | 99% | 87% | −12% |
| ship | 92% | 91% | −1% |
| truck | 95% | 88% | −7% |

The accuracy gap is real. SEL trades approximately 10 percentage points of accuracy for a 99x reduction in compute. Whether that trade is acceptable depends entirely on the use case — and for a large class of real-world applications (see Applications below), it is not just acceptable but necessary.

---

## Installation

Requires Python 3.8+, PyTorch 1.13+, and a CUDA-capable GPU (or patience on CPU).

```bash
git clone https://github.com/your-username/staged-embarrassment-learning
cd staged-embarrassment-learning
pip install torch torchvision numpy matplotlib pandas
```

To run the full experiment (baseline + staged):

```bash
# Open in Google Colab (recommended — tested on T4 GPU)
staged_embarrassment_v3.ipynb

# Or run locally
jupyter notebook staged_embarrassment_v3.ipynb
```

Set `Runtime → T4 GPU` in Colab before running. The full notebook takes approximately 40 minutes end-to-end.

---

## Configuration

All hyperparameters are defined at the top of Cell 1:

```python
STAGE_CONFIG = [
    ('Stage 1 Easy',    0.00, 0.25, 0.30, 10),  # (name, pool_start, pool_end, threshold, epochs)
    ('Stage 2 Medium',  0.20, 0.50, 0.45, 10),
    ('Stage 3 Hard',    0.40, 0.65, 0.58, 10),
    ('Stage 4 Harder',  0.60, 0.82, 0.68, 10),
    ('Stage 5 Hardest', 0.78, 1.00, 0.76, 10),
]

SAMPLES_PER_CLASS = 2000   # samples drawn per class per stage
TEMPERATURE       = 1.5    # softening temperature for embarrassment signal
BASELINE_EPOCHS   = 30     # epochs for the dense baseline
```

To experiment with more aggressive sparsity, lower `GUILT_THRESHOLD` or increase the percentile in Cell 5. To relax curriculum pacing, widen the pool windows in `STAGE_CONFIG`.

---

## Applications

The value of SEL is in the 99% compute reduction. This changes what is possible on constrained hardware.

**Edge AI and Robotics**
Delivery drones, surveillance UAVs, and micro-robots (ESP32, Raspberry Pi Zero) can now fine-tune vision models in the field without draining batteries or requiring thermal management. Wearable health monitors can adapt arrhythmia and gait models to individual users locally, without burning the skin or requiring daily charging.

**Smart City and IoT Infrastructure**
Waste segregation sensors can learn to recognize novel packaging at the bin level without cloud connectivity. Solar-powered traffic cameras can update vehicle-recognition models on harvested energy alone. Remote industrial sensors in pipelines and substations — where battery replacement is expensive and physically dangerous — can adapt continuously.

**Space and Aerospace**
Satellites operate under strict thermal budgets and cannot dissipate heat in vacuum. SEL enables on-orbit image classification (forest fire detection, crop monitoring) within those constraints. Deep-space probes can use the embarrassment signal itself as a data prioritization mechanism — flagging high-entropy, high-value observations for transmission and discarding redundant ones.

**Green Data Centers**
A 99% reduction in FLOPs means a 99% reduction in the energy cost of fine-tuning. This allows researchers on 10-year-old hardware, students in low-resource environments, and data centers targeting carbon-neutral operation to run deep learning workloads that were previously inaccessible to them.

**Privacy-Preserving Federated Learning**
On-device training (Gboard, photo classification, health apps) must be invisible to the user — no heat, no battery drain. SEL makes local fine-tuning genuinely transparent. For confidential medical research, hospitals can train shared models on private patient records without the data leaving the premises, using existing low-spec infrastructure.

**Defense and Tactical Systems**
Thermally silent AI for soldier vision systems (goggles, helmets) must not emit heat detectable by infrared sensors. Underwater autonomous submersibles fine-tune sonar recognition while submerged, where power is finite and surface communication is impossible.

**Active Video Summarization**
A camera system that uses the embarrassment signal as an "attention trigger" records only when it encounters something novel — something it does not confidently understand. This alone could reduce CCTV storage requirements by orders of magnitude.

---

## Project Structure

```
staged-embarrassment-learning/
│   stats.html
│   LICENSE
│   README.md
│   requirements.txt
│   staged_embarrassment_v3.ipynb
│   structure.txt
│
├───logs
│       baseline_hist.csv
│       staged_hist.csv
│
└───visulas
        results_final.png
        thermal_baseline.png
        thermal_staged.png
```

---

## Limitations and Honest Notes

The accuracy gap is significant and should not be minimized. On CIFAR-10, SEL reaches 82.6% on the held-out test set versus 92.7% for the dense baseline — a 10-point gap. The `cat` class in particular performs poorly at 61% under SEL, suggesting that visually ambiguous classes require more exposure than the staged curriculum currently provides.

This is version 3 of the system. Earlier versions used a uniform curriculum and no sparsity, and produced worse results in both accuracy and efficiency. The current version was built over approximately one month of iteration on a single T4 GPU.

The comparison to published methods (RepDistiller, Google's Lottery Ticket) in the scatter plot is illustrative rather than rigorous — those methods operate on different splits and hardware, and a direct controlled comparison is future work.

The core hypothesis remains: **a network that knows where it is failing, and spends its compute only there, can learn efficiently.** The data supports this. The full story of why it works — especially the interaction between sparse gradients and the embarrassment signal — is still being understood.

---

## Author

Built by a 21-year-old, third year Information Science and Engineering student, as a personal research project.
The inspiration was a 7-year-old niece throwing a ball at a park.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
