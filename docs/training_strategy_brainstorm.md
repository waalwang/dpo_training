# Training Strategy Brainstorm: Beyond User-Turn Weighting

## Reframing the Problem

Option 1 (mask user turns from loss) is not actually wasting user turns. The
model conditions on them via causal attention - every user token affects the
hidden states of subsequent assistant tokens, and gradient flows through those
hidden states during backprop. User turns already influence the preference
signal indirectly; they're just not being pushed toward a target distribution.

So the real question is not "how to include user turns" but "how to get
stronger, less noisy, more flow-aware preference signal from multi-turn data".
With that framing, there are better levers than weighting user turns.

## Strategic Options

### 1. Swap the Loss Function (IPO)

Your config already exposes `loss_type: sigmoid | hinge | ipo`. For noisy
crawler pairs, IPO is strictly better than vanilla DPO:

```
DPO:  -log sigmoid(beta * delta)         <- saturates, overfits to noise
IPO:  (delta - 1/(2*beta))^2              <- bounded, doesn't overfit
```

IPO is designed for exactly this situation - noisy preference labels where you
don't fully trust every pair. One-line change, highest ROI.

### 2. Mixed SFT + DPO (RPO)

Add an SFT loss on the chosen trajectory alongside the DPO loss:

```
loss = dpo_loss + rpo_alpha * sft_loss_on_chosen
```

TRL supports this natively via `DPOConfig(rpo_alpha=1.0)`. Benefits:

- Positive signal: SFT on chosen teaches the model to generate good multi-turn
  patterns directly
- Preference signal: DPO still separates chosen from rejected
- Stability: the SFT term anchors the model, reducing degeneration risk at low
  beta

This is the cleanest answer to the dialogue-flow learning goal. The SFT
component directly trains "produce this kind of conversation flow", while DPO
refines preferences on top.

### 3. Two-Stage Pipeline: SFT then DPO

The heavyweight version of option 2:

- Stage 1: SFT on chosen trajectories only. Learns multi-turn flow through
  standard next-token prediction. Chat template maintains role boundaries.
- Stage 2: DPO (or IPO) with chosen vs rejected. Refines preference distribution
  on a competent base.

This is the Zephyr/Tulu recipe. DPO assumes a competent base model. If
Qwen2.5-1.5B is not already good at reddit/HN-style multi-turn chains out of the
box, DPO has nothing to refine - SFT fixes that first.

Downside: two training runs, roughly twice the cloud time.

### 4. Length Normalization / SimPO

Trajectories have varying length. In the current setup, longer branches
accumulate more log-prob terms and can dominate the comparison. Fixes:

- Length-normalized DPO: divide `chosen_logps` by token count before the loss
- SimPO: reference-free, length-normalized by design, and removes the reference
  model entirely - saves about 50% VRAM, which lets you fit a 7B model on A100
  40GB without offloading

SimPO gives up the KL anchor though, so it is more aggressive. For noisy data,
try IPO first.

### 5. Score-Delta Filtering

The crawler exports `score_delta` per pair. Weak deltas are noise. Filter them
in `data_loader.py`:

```python
rows = [r for r in rows if r["score_delta"] > threshold]
```

No downside. Threshold is empirical but even `score_delta > 0` (strict positive)
cuts out ties.

### 6. Turn-Level DPO

Per-turn preference comparison instead of one joint comparison per trajectory.
Better credit assignment, but assumes chosen/rejected have comparable turn
structures (they often do not - one branch might be 4 turns, another 7).
Complex to implement correctly and probably not the right call for this data.

## Recommended Path

Stacked in priority order, cheapest to most ambitious:

1. `loss_type: ipo` in config (one line)
2. Score-delta filtering in `data_loader.py` (few lines)
3. `rpo_alpha: 1.0` in the DPO config (one line, enables mixed SFT + DPO)
4. First training run - if the model learns decent flow, stop here
5. If results are weak, two-stage SFT -> DPO as the proper fix

Steps 1-3 together give: robust-to-noise loss + filtered signal + direct SFT on
good trajectories, all in a single training run. That addresses the dialogue
flow goal directly rather than through a proxy.

## Notes on Beta Under These Changes

- IPO is less sensitive to beta than DPO (the quadratic form does not saturate),
  so `beta: 0.1` is safer under IPO than under sigmoid DPO.
- With `rpo_alpha > 0`, the SFT component provides additional anchoring, so low
  beta is even less risky.
- If switching to SimPO, beta needs retuning (typical range is 2.0-2.5 with the
  default length normalization).
