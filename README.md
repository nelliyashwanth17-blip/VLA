# Reinforcement Learning for Real-Time Vision-Language-Action Policies

A CPU-friendly research prototype for image-and-language-conditioned tabletop manipulation using PPO.

## Research scope

This first implementation uses a custom 2D Gymnasium simulator because the development computer has 8 GB RAM and no dedicated GPU. It is a lightweight prototype of the proposed VLA + RL pipeline. It is **not** an RLBench or CALVIN implementation, and results must not be described as benchmark results from either platform.

## Architecture

1. **Visual input:** rendered RGB image of a tabletop containing red and blue blocks.
2. **Visual encoder:** Stable-Baselines3 `CombinedExtractor` applies a CNN to the image observation.
3. **Language input:** a small semantic instruction vector encodes which block should move. This is a controlled prototype encoder, not a pretrained language model.
4. **Scene state:** normalized block coordinates are included as simulator state features.
5. **Feature fusion:** the multi-input policy concatenates the image, instruction and state feature representations.
6. **Action policy:** PPO selects one of four discrete movement actions.
7. **Reward:** progress toward adjacency is rewarded; successful completion receives a larger reward.

## Requirements

- macOS
- Python 3.11 recommended
- 8 GB RAM
- CPU-only execution

## Installation (macOS)

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependency installation fails, check the installed Python version and the current Stable-Baselines3 installation documentation.

## Run the environment test

```bash
python smoke_test.py
```

## Train PPO

Start with a short pilot:

```bash
python train.py --timesteps 2000
```

If the pilot works, run a longer training experiment:

```bash
python train.py --timesteps 12000
```

The trained model is saved as `models/ppo_vla_tabletop.zip`.

## Evaluate

```bash
python evaluate.py
```

Evaluation compares PPO with a rule-based heuristic on the same seeded episodes. It saves:
- `results/evaluation_results.csv`
- Success-rate, latency, episode-length and reward graphs.

## Important evaluation note

The rule-based policy is a transparent reference controller and uses the same compact state and instruction fields. The reported action latency measures policy decision time, not full simulator end-to-end latency. Record your Mac model, macOS version, Python/package versions, training steps and random seeds when reporting results.

Do not report sample or expected metrics as measured findings. Run the scripts and use the generated CSV and plots.

## Limitations and next steps

- The instruction representation is a two-class semantic code, not general natural-language understanding.
- The environment is 2D and does not model robot-arm kinematics, grasping physics or camera noise.
- The image CNN and policy are trained in a small simulation; transfer to RLBench, CALVIN or a physical robot has not been established.
- The next research stage is to strengthen language grounding, test generalization to held-out configurations, and investigate a compatible robotics benchmark if compute resources permit.

## Repository structure

```text
.
├── vla_env.py
├── baseline.py
├── train.py
├── evaluate.py
├── smoke_test.py
├── requirements.txt
├── README.md
├── models/
└── results/
```
