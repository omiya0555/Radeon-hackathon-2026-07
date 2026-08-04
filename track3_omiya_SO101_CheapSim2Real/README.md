# Cheap Sim2Real — teaching a $150 arm with 10 real demonstrations

**Track 3 (Physical AI) · AMD AI DevMaster Hackathon**
Genesis · LeRobot ACT · AMD Radeon GPU / ROCm · SO-101

Imitation learning wants ~50 demonstrations per task. On real hardware that is **2-3 hours of
a person driving a leader arm, for every task** — the dominant cost of teaching a cheap robot
anything. So build a measured digital twin, let a scripted expert generate the bulk for free,
and spend human time only on the handful of episodes that anchor the policy to reality.

![Real SO-101 and its Genesis twin from the same overhead camera](docs/media/sim2real_overhead.png)

<sub>Two frames placed side by side, centre-cropped to a common aspect and otherwise
unmodified — the originals are
[`real_arm_world.png`](docs/media/real_arm_world.png) (real overhead camera) and
[`sim_arm_world.png`](docs/media/sim_arm_world.png) (Genesis, same pose).</sub>

### In 60 seconds

| | |
|---|---|
| **Result** | **85% success on the physical SO-101** (17/20) from **10 real demonstrations** (~35 min of human time) + 50 simulated ones |
| **The control arm** | The same 10 real episodes *without* simulated pretraining: 25% (5/20). Fisher exact **p = 0.00033** |
| **On the GPU** | Simulation stepping, camera rendering, ACT training and closed-loop inference — all on one Radeon (`gs.amdgpu` + ROCm) |
| **Watch** | [`docs/demo_video.mp4`](docs/demo_video.mp4) — 2:48, a real session from `git clone` to the arm placing the cube |
| **Reproduce** | 3 commands, ~3 minutes: [jump to it](#reproducing) |
| **Report** | [`docs/Technical_Report.md`](docs/Technical_Report.md) — method, 5 experiments, failure analysis, stated limits |

### The 10 real demonstrations are what makes it work

<img src="docs/media/real_success.gif" width="49%"> <img src="docs/media/domain_randomisation.gif" width="49%">

**Left:** the physical arm, overhead and wrist cameras. The first grasp slips and the policy
retries — recovery behaviour it learned from the *simulated* scripted expert, never from a
human. **Right:** the same scene under domain randomisation, 12 domains at once (cube colour,
table colour, lighting; friction and mass are re-sampled per episode).

### Why the GPU is load-bearing

Five ACT runs went into this submission — three domain-randomisation ablations, a co-training
run, and the control run that gives the 25% above. Each is 20k-100k optimisation steps over
tens of thousands of frames with two ResNet-18 towers. On the Radeon host a 20k-step fine-tune
takes **2 h 11 m at 2.56 step/s** in 3.9 GB VRAM, and the loader never starves the GPU
(`data_s` 0.002 s against `updt_s` 0.388 s — **99.5% of wall-clock is compute**). Without
that, the honest version of this project is one training run and a claim; with it, two
plausible hypotheses got tested and refuted (§5.2, §5.3) and the headline result got a
control arm.

## What is here

| Path | Contents |
|---|---|
| [`docs/Technical_Report.md`](docs/Technical_Report.md) | Method, 5 experiments, failure analysis, limitations |
| [`docs/demo_video.mp4`](docs/demo_video.mp4) | 2:48 — the reproduction sequence running on the Radeon host, then the real robot |
| [`submission/README.md`](submission/README.md) | Reproduction guide: environment, per-experiment commands, expected results, troubleshooting |
| `submission/src/` | Digital twin + 5-DOF IK expert, dataset/DR tooling, sim & real evaluation, **camera liveness validation** |
| `submission/setup.sh` · `run_pipeline.sh` | One-command environment setup and dataset → train → evaluate |
| `Dockerfile` | Reproducible ROCm environment |

## Results

**Real robot** (SO-101, cube pick-and-place onto a target sheet, 20 fixed placements)

| Policy | Real success | 95% CI | Placement error |
|---|---|---|---|
| Sim 50 (full DR) pretrain → fine-tune with 10 real episodes | **85%** (17/20) | 64-95% | 1-2 cm |
| 10 real episodes only, from scratch (control) | 25% (5/20) | 11-47% | — |

20 episodes per arm on the same 20 placements. **Fisher exact test p = 0.00033**
(two-tailed): the simulated pretraining is what makes 10 real demonstrations enough.

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

Three commands on a fresh ROCm instance. No robot needed — collection, training and
closed-loop evaluation all run on the AMD host (verified headless on the hackathon Radeon
Cloud template).

```bash
git clone -b track3-so101-cheap-sim2real \
  https://github.com/omiya0555/Radeon-hackathon-2026-07.git
cd Radeon-hackathon-2026-07/track3_omiya_SO101_CheapSim2Real/submission

bash setup.sh                    # deps + env, ends with a smoke test (SMOKE_OK)
bash run_pipeline.sh --quick     # dataset -> train -> closed-loop eval, ~3 min
```

`--quick` is a plumbing check (100 training steps, so it reports 0% by design);
`run_pipeline.sh --full` (100k steps, ~11 h) reproduces the 60% above. See
`docs/demo_video.mp4` for this exact sequence running on the Radeon host.

If the clone fails with `server certificate verification failed`, the image's CA bundle is
stale — see [§0 of the reproduction guide](submission/README.md#0-get-the-project).

[`submission/README.md`](submission/README.md) has the per-experiment commands (checkpoint
curve, colour sweep), the real-robot path, expected results and troubleshooting.

## Licence and attribution

Apache-2.0. The scene/expert/recording structure follows the Franka demo in
[`wangxunx/franka_fruit_pick_demo`](https://github.com/wangxunx/franka_fruit_pick_demo);
the SO-101 port, the 5-DOF IK strategy, the real-robot layer and all experiments are this
submission's own work.
