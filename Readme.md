# Causal-Paced Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CP-DRL** (Causal-Paced Deep Reinforcement Learning) is a curriculum learning framework aware of SCM differences between tasks based on interaction data approximation. This signal captures task novelty, which we combine with the agent’s learnability, measured by reward gain, to form a unified objective. Empirically, CP-DRL outperforms existing curriculum methods on the Point Mass benchmark, achieving faster convergence and higher returns. CP-DRL demonstrates reduced variance with comparable final returns in the Bipedal Walker-Trivial setting, and achieves the highest average performance in the Infeasible variant.

## 🚀 Running the Code

### Point Mass Environment

All code for the Point Mass environment lives in the `pointmass` directory. Installation and setup follow the instructions in the [Currot codebase](https://github.com/psclklnk/currot/tree/main). Our CP-DRL implementation is located under [`deep_sprl/teachers/cp_drl`](https://github.com/Cho-Geonwoo/CP-DRL/tree/main/pointmass/deep_sprl/teachers/cp_drl). To launch the Point Mass experiments with CP-DRL, run:

```bash
python run.py --type cp_drl --learner ppo --env point_mass_2d --t_scale 10
```

### Bipedal Walker Environment

You can find the Bipedal Walker code in the `bipedalwalker` directory. Follow the installation steps from the [TeachMyAgent repository](https://github.com/flowersteam/TeachMyAgent), and don’t forget to initialize its submodule in `bipedalwalker`. Our CP-DRL teacher lives in [`TeachMyAgent/teachers/algos/cp_drl.py`](https://github.com/Cho-Geonwoo/CP-DRL/blob/main/bipedalwalker/TeachMyAgent/teachers/algos/cp_drl.py). To run your experiments, use:

```bash
python run.py \
  --test_set parametric_stumps_test_set \
  --keep_periodical_task_samples 250000 \
  --env parametric-continuous-stump-tracks-v0 \
  --max_stump_h 9.0 \
  --max_obstacle_spacing 6.0 \
  --embodiment old_classic_bipedal \
  --student sac_v0.1.1 \
  --backend tf1 \
  --steps_per_ep 500000 \
  --nb_test_episode 100 \
  --nb_env_steps 20 \
  --teacher CP-DRL
```

### Analysis on the Causal World Environment

All scripts for analyzing causal difference versus misalignment score reside in the `causalworld` directory. First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Next, run the main analysis:

```bash
python run.py
```

If you’d like to visualize the environment setup, execute:

```bash
python visualize_env.py
```

## 🤝 Acknowledgements

We build our Pointmass environment experiments on the [Currot codebase](https://github.com/psclklnk/currot/tree/main) (Curriculum Reinforcement Learning via Constrained Optimal Transport). For the Bipedal Walker environment, we leverage the implementation from the [TeachMyAgent repository](https://github.com/flowersteam/TeachMyAgent). We are grateful to the original authors for making their code publicly available.

## Misc

Please note that this codebase may not exactly reproduce the results reported in the paper due to potential human errors during code migration. If you observe any discrepancies in performance, feel free to reach out-we’d appreciate your feedback.

## 📄 License

This project is released under the **MIT License**. Please note that it depends on several third-party libraries, each of which is governed by its own license.
