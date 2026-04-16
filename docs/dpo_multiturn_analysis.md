# DPO Multi-Turn Trajectory: How Sequential Patterns Are Captured

## Data Structure

Each training example has three parts:

- `prompt` - shared prefix turns up to the fork point (masked from loss)
- `chosen` - prefix + best-scoring branch
- `rejected` - prefix + worst-scoring branch

The fork is a point in the comment tree where two branches diverged in quality,
scored holistically via discounted cumulative score over the full branch.

## How Causal Patterns Are Captured

TRL tokenizes the full chosen/rejected sequences and computes per-token log-probs
under a causal LM:

```
log P(chosen) = sum of log P(token_t | token_1 ... token_{t-1})
                for all supervised (post-fork) tokens
```

Every token is conditioned on everything before it, so the sequential dependency
is preserved through causal attention. `fork_asst_t2` is conditioned on
`prefix + fork_user_t1 + fork_asst_t1 + fork_user_t2`.

## DPO Loss

```
chosen_logps   = joint log-prob of entire chosen branch
rejected_logps = joint log-prob of entire rejected branch

loss = -log sigmoid(beta * (chosen_logps - rejected_logps))
```

A single scalar comparison of two joint probabilities over the whole trajectory.
The model is trained to assign higher probability to the entire chosen sequence
than the rejected one.

## What This Captures

- Sequential dependency - yes, via causal attention
- Trajectory-level preference - yes, joint probability over the full branch
- Conversational flow - partially; a good turn 2 leading to a bad turn 3 still
  gets penalized at the trajectory level

## What This Misses

- Which specific turn drove the quality difference - gradient is spread across
  all tokens equally
- User turns after the fork - masked by default in TRL, so the model gets no
  feedback on whether the conversation was steered well by the user side

## Open Questions

1. Should user turns after the fork contribute to the loss?
   - Option 1: supervise all post-fork turns (user + assistant) - learns full
     conversational flow but model starts predicting user turns
   - Option 2: lower loss weight on user turns rather than zeroing them out
   - Option 3: keep current, trust that chosen assistant turns implicitly reflect
     surrounding context quality

2. Score delta filtering - pairs with small `score_delta` are weak preference
   signals and potentially noisy. Worth filtering below a threshold before training.

## Beta Considerations

For trajectory data from a crawler, pairs may differ for stylistic reasons
(oral vs formal) rather than substantive quality. A higher beta (0.5-1.0) is
safer for noisy data. Current config uses `beta: 0.1` (aggressive) - monitor
reward margins in wandb during the first run.
