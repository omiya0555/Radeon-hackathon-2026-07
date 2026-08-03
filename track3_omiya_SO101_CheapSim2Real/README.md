# Cheap Sim2Real: teaching a $150 arm a manipulation task with 10 real demonstrations

**Track 3 (Physical AI) — AMD AI DevMaster Hackathon**

A complete sim-to-real pipeline for the **SO-101**, a ~$150 3D-printed robot arm. A digital
twin in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) auto-collects demonstrations
with a scripted expert, [ACT](https://arxiv.org/abs/2304.13705) is trained on an
**AMD Radeon GPU (ROCm)**, and the resulting policy runs on real hardware.

**Result: 8/10 success on the physical robot**, from 50 simulated demonstrations plus
**only 10 human teleoperation episodes** (~35 minutes of a person's time).

<!-- demo video: docs/demo_video.mp4 -->

## Why this matters

Imitation learning for manipulation conventionally needs ~50 demonstrations per task. On real
hardware that is 2-3 hours of a human driving a leader arm, per task — the single biggest cost
in teaching a low-cost arm anything. This project attacks that cost directly: generate the bulk
of the data in simulation where it is free and reproducible, and spend human time only on the
10 episodes that anchor the policy to reality.

The GPU is what makes the trade viable. Each ACT run is 20k-100k optimisation steps over
tens of thousands of frames with two ResNet-18 vision towers; five such runs went into this
submission. On the Radeon box each 20k-step fine-tune took **2h11m at 2.56 step/s** with
3.9 GB VRAM, and the data pipeline never starved the GPU (`data_s` 0.002 s against
`updt_s` 0.388 s — **99.5% of wall-clock was compute**). Iterating on the recipe — three
domain-randomisation ablations, a co-training run and a control run — was only affordable
because the hardware turned each experiment into an overnight job rather than a week.

## What is here

| Path | Contents |
|---|---|
| `docs/Technical_Report.md` / `.pdf` | Full report: method, five experiments, failure analysis |
| `docs/demo_video.mp4` | Sim rollouts and the real-robot success |
| `docs/upstream_contribution_evidence.*` | Upstream issue reported to LeRobot |
| `submission/src/scene/` | Genesis digital twin + scripted expert (5-DOF IK) |
| `submission/src/data/` | Dataset recording, domain randomisation, real+sim merge, **camera liveness validation** |
| `submission/src/train/` | Training commands for ROCm |
| `submission/src/eval/` | Closed-loop sim evaluation + rollout video audit |
| `submission/src/real/` | Real SO-101 runner (unit bridge, camera transport, teleop recorder) |
| `Dockerfile` | Reproducible ROCm environment |

## Results

**Real robot** (SO-101, cube pick-and-place onto a target sheet, 10 fixed placements)

| Policy | Real success | Placement error |
|---|---|---|
| Sim 50 (full DR) pretrain → fine-tune with 10 real episodes | **8/10** | 1-2 cm |
| 10 real episodes only, from scratch | 3/10 | — |

Fisher exact test p = 0.070 (two-tailed) — the direction is clear but n=10 per arm is
underpowered; see the report for the honest treatment.

**Simulation** (fixed-seed evaluation, so checkpoints are compared on identical placements)

- Success saturates at **60% at 60k steps** while the loss is still falling — a data ceiling
  of the 50-episode set, not an optimisation failure.
- Colour generalisation: a policy trained **only on red cubes** keeps working on unseen
  colours. Success correlates **0.956 with ‖Δc‖**, the RGB-space distance between object and
  table, and is **independent of hue** — an orthogonal blue and a parallel wood-brown both
  score 50%. It collapses to 10% only when the cube matches the table colour.
- Domain randomisation is nearly free: final loss 0.028 / 0.028 / 0.030 for no-DR /
  colour-only / full DR.

## Artifacts

- Model: https://huggingface.co/omiya239532/so101_act_cotrain
- Datasets: [`so101_cube_red`](https://huggingface.co/datasets/omiya239532/so101_cube_red) ·
  [`so101_cube_colors`](https://huggingface.co/datasets/omiya239532/so101_cube_colors) ·
  [`so101_cube_dr`](https://huggingface.co/datasets/omiya239532/so101_cube_dr) ·
  [`so101_real_teleop2`](https://huggingface.co/datasets/omiya239532/so101_real_teleop2)

## Reproducing

See [`submission/README.md`](submission/README.md) for the full command sequence. The short
version, on a ROCm host:

```bash
docker build -t so101-sim2real .
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video so101-sim2real

# 1. collect 50 simulated demonstrations with full domain randomisation
python -m src.data.record_dataset_so101 --episodes 50 --dr-all --repo-id <you>/so101_cube_dr

# 2. train ACT on the Radeon GPU
lerobot-train --dataset.repo_id=<you>/so101_cube_dr --policy.type=act \
  --steps=60000 --batch_size=8 --policy.device=cuda --output_dir=outputs/act_dr

# 3. evaluate in simulation on fixed seeds
python -m src.eval.eval_policy_so101 --policy-path outputs/act_dr/checkpoints/060000/pretrained_model \
  --repo-id <you>/so101_cube_dr --episodes 20
```

Steps 4-6 (real teleoperation, merging, fine-tuning) need the physical arm; they are
documented in `submission/README.md`.

## Licence and attribution

Apache-2.0. The scene/expert/recording structure follows the Franka demo in
[`wangxunx/franka_fruit_pick_demo`](https://github.com/wangxunx/franka_fruit_pick_demo);
the SO-101 port, the 5-DOF IK strategy, the real-robot layer and all experiments are this
submission's own work.
