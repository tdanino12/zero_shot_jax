"""
MEP (Maximum Entropy Population-based Training) in JAX
=======================================================
STRUCTURE OVERVIEW:
-------------------
PHASE 1 — Train Maximum Entropy Population
  - N population agents (MLPs), each trained via self-play PPO
  - Augmented reward per agent i: r_aug = r_task - α * log( π̄(a|s) )
    where π̄(a|s) = (1/N) Σ_i π^(i)(a|s) [mean policy of population]
  - This penalizes taking actions that the average policy also likes → diversity
  - All N agents share the same network architecture but have separate parameters
  - Each agent plays BOTH roles (agent_0 and agent_1) against a copy of itself
PHASE 2 — Train Ego Agent via Prioritized Sampling
  - One ego agent (RNN) trained against frozen population
  - After each rollout with partner i, track avg_return(ego, partner_i)
  - Sampling probability: p(i) ∝ rank(1 / avg_return_i)^β [rank-based prioritization]
  - Higher priority to partners the ego struggles with
  - Ego is trained with standard PPO (no PE bonus needed)
IMPLEMENTATION NOTES:
---------------------
- Population params stored as a leading-axis pytree: params[k] has shape (N, ...)
- Phase 1 uses jax.vmap over population to compute mean policy in one forward pass
- Phase 2 partner sampling is done in Python (outside jit) using numpy weights
- Each PPO update uses the standard clipped surrogate + value loss + entropy bonus
"""
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Tuple
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper
import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import (
    make_counter_circuit_9x9, make_forced_coord_9x9,
    make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9,
)
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
import argparse
import wandb
import os
import orbax.checkpoint
from flax.training import orbax_utils
import imageio
import shutil

# ---------------------------------------------------------------------------
# 1. SHARED NETWORK ARCHITECTURE (used for both population and ego)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Standard MLP actor-critic shared by all agents."""
    action_dim: int
    hidden_size: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if x.ndim == 1:
            pass
        elif x.ndim == 3:
            x = x.reshape(-1)
        elif x.ndim == 4:
            x = x.reshape((x.shape[0], -1))
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Actor head
        a = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        a = act_fn(a)
        a = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a)
        a = act_fn(a)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        pi = distrax.Categorical(logits=logits)

        # Critic head
        v = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        v = act_fn(v)
        v = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(v)
        v = act_fn(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        return pi, jnp.squeeze(v, axis=-1)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        lstm_state = carry
        ins, resets = x
        lstm_state = jax.tree_map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    obs_shape: tuple = (7, 7, 26)

    @nn.compact
    def __call__(self, hidden, obs, dones):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        reshaped_obs = obs.reshape((-1, *self.obs_shape))
        embedding = nn.Conv(
            features=64,
            kernel_size=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(reshaped_obs)
        embedding = nn.relu(embedding)
        embedding = nn.Conv(
            features=32,
            kernel_size=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        batch_size, num_envs, flattened_obs_dim = obs.shape
        embedding = embedding.reshape((batch_size, num_envs, -1))

        embedding = nn.Dense(
            self.hidden_size * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_size * 3 // 4,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_size // 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        if 0:
            routing_logits = nn.Dense(
                3,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0)
            )(actor_mean)
            routing_weights = nn.softmax(routing_logits)
            chosen_expert = jnp.argmax(routing_weights, axis=-1)
            expert_outputs = []
            for _ in range(3):
                expert_out = nn.Dense(
                    self.hidden_size // 4,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0)
                )(actor_mean)
            expert_out = jnp.tanh(expert_out) if self.activation == "tanh" else nn.relu(expert_out)
            expert_outputs.append(expert_out)
            expert_outputs = jnp.stack(expert_outputs, axis=-2)
            one_hot = jax.nn.one_hot(chosen_expert, num_classes=3)
            actor_mean = jnp.sum(expert_outputs * one_hot[..., None], axis=-2)
        else:
            actor_mean = nn.Dense(
                self.hidden_size // 4,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(actor_mean)
            actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        if 1:
            critic = nn.Dense(
                self.hidden_size * 3 // 4,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(critic)
            critic = activation(critic)
            critic = nn.Dense(
                self.hidden_size // 2,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(critic)
            critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)
        return hidden, (pi, jnp.squeeze(critic, axis=-1))

    @staticmethod
    def initialize_carry(hidden_size: int, batch_size: int):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


# ---------------------------------------------------------------------------
# 2. TRANSITION TUPLE (used in rollout buffer)
# ---------------------------------------------------------------------------
class Transition(NamedTuple):
    done: jnp.ndarray       # (num_envs,)
    action: jnp.ndarray     # (num_envs,)
    value: jnp.ndarray      # (num_envs,)
    reward: jnp.ndarray     # (num_envs,) — AUGMENTED reward in Phase 1
    log_prob: jnp.ndarray   # (num_envs,)
    obs: jnp.ndarray        # (num_envs, *obs_shape)
    hstate: Tuple           # (h, c) each (num_envs, hidden_size) — None for Phase 1 MLP
    info: dict


# ---------------------------------------------------------------------------
# 3. PPO LOSS (Phase 1 MLP — stateless forward pass)
# ---------------------------------------------------------------------------
def ppo_loss(params, apply_fn, traj: Transition, advantages, targets, clip_eps, vf_coef, ent_coef):
    """Standard clipped PPO objective for MLP (Phase 1). Returns (total_loss, (vf_loss, pg_loss, entropy))."""
    pi, value = apply_fn(params, traj.obs)
    log_prob = pi.log_prob(traj.action)

    # Value loss (clipped)
    value_clipped = traj.value + jnp.clip(value - traj.value, -clip_eps, clip_eps)
    vf_loss = 0.5 * jnp.maximum(
        jnp.square(value - targets),
        jnp.square(value_clipped - targets)
    ).mean()

    # Policy loss
    ratio = jnp.exp(log_prob - traj.log_prob)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = -jnp.minimum(
        ratio * adv_norm,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    ).mean()

    entropy = pi.entropy().mean()
    total = pg_loss + vf_coef * vf_loss - ent_coef * entropy
    return total, (vf_loss, pg_loss, entropy)


# ---------------------------------------------------------------------------
# 4. GAE COMPUTATION
# ---------------------------------------------------------------------------
def compute_gae(traj: Transition, last_val, gamma, gae_lambda):
    def _step(carry, t):
        gae, next_val = carry
        done, value, reward = t.done, t.value, t.reward
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_val), last_val),
        traj,
        reverse=True,
        unroll=16
    )
    return advantages, advantages + traj.value  # (advantages, targets)


# ---------------------------------------------------------------------------
# 5. PHASE 1 — POPULATION TRAINING  [UNCHANGED]
# Each agent self-plays and gets the PE diversity bonus.
# ---------------------------------------------------------------------------
def make_phase1_train(config, env):
    """
    Returns a function `train_population(rng) -> population_params`
    Inside:
      - Initialise N train_states
      - Collect rollouts: agent i plays BOTH roles against itself
      - Compute augmented reward: r_aug = r_task - α * log( π̄(a|s) )
        where π̄ is evaluated by running ALL N networks on the obs and averaging
      - PPO update for each agent independently
    """
    N = config["POPULATION_SIZE"]
    obs_shape = env.observation_space().shape
    action_dim = env.action_space().n
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_updates = config["PHASE1_TOTAL_TIMESTEPS"] // num_steps // num_envs
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS_POPULATION"]
    alpha = config["ALPHA"]
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    clip_eps = config["CLIP_EPS"]
    vf_coef = config["VF_COEF"]
    ent_coef = config["ENT_COEF"]
    lr = config["LR_POPULATION"]
    max_grad_norm = config["MAX_GRAD_NORM"]

    network = ActorCritic(action_dim, config["HIDDEN_SIZE"], config["ACTIVATION"])

    def train_population(rng):
        # ------------------------------------------------------------------
        # 5a. Initialise N train states (one per population agent)
        # ------------------------------------------------------------------
        rng, *init_rngs = jax.random.split(rng, N + 1)
        dummy_obs = jnp.zeros(obs_shape)

        def init_one(key):
            params = network.init(key, dummy_obs)
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5)
            )
            return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        pop_states = jax.vmap(init_one)(jnp.stack(init_rngs))

        # ------------------------------------------------------------------
        # 5b. Initialise N × num_envs environments
        # ------------------------------------------------------------------
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, N * num_envs)
        init_obs_flat, init_env_state_flat = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

        init_obs = jax.tree_util.tree_map(
            lambda x: x.reshape((N, num_envs) + x.shape[1:]), init_obs_flat
        )
        init_env_state = jax.tree_util.tree_map(
            lambda x: x.reshape((N, num_envs) + x.shape[1:]), init_env_state_flat
        )

        # ------------------------------------------------------------------
        # 5c. Main training loop
        # ------------------------------------------------------------------
        def _update_step(carry, _):
            """One PPO update across ALL N population agents simultaneously."""
            pop_states, env_states, last_obs, rng = carry

            def _env_step(inner_carry, _):
                pop_states, env_states, last_obs, rng = inner_carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                obs_flat = last_obs.reshape((N, num_envs * 2) + obs_shape)

                def get_actions(state, obs):
                    pi, val = state.apply_fn(state.params, obs)
                    return pi, val

                pop_pi, pop_val = jax.vmap(get_actions)(pop_states, obs_flat)

                action_rngs = jax.random.split(action_rng, N * num_envs * 2).reshape(N, num_envs * 2, 2)
                actions = jax.vmap(lambda pi, keys: jax.vmap(lambda d, k: d.sample(seed=k))(pi, keys))(
                    pop_pi, action_rngs
                )
                log_probs = jax.vmap(lambda pi, a: pi.log_prob(a))(pop_pi, actions)

                actions_dict_shaped = actions.reshape(N, num_envs, 2)
                log_probs_shaped = log_probs.reshape(N, num_envs, 2)
                vals_shaped = pop_val.reshape(N, num_envs, 2)

                act_dict = {
                    "agent_0": actions_dict_shaped[:, :, 0],
                    "agent_1": actions_dict_shaped[:, :, 1],
                }

                step_rngs = jax.random.split(step_rng, N * num_envs).reshape(N, num_envs, 2)

                def step_envs(env_state, step_key, act0, act1):
                    a = {"agent_0": act0, "agent_1": act1}
                    obs_new, env_state_new, reward, done, info = jax.vmap(env.step)(
                        step_key, env_state, {k: v for k, v in a.items()}
                    )
                    return obs_new, env_state_new, reward, done, info

                new_obs, new_env_states, rewards, dones, infos = jax.vmap(step_envs)(
                    env_states, step_rngs,
                    act_dict["agent_0"], act_dict["agent_1"]
                )

                new_obs_stack = jnp.stack(
                    [new_obs["agent_0"], new_obs["agent_1"]], axis=2
                )

                # ---- Compute Population Entropy bonus --------------------
                obs_a0 = last_obs[:, :, 0, :]
                act_a0 = actions_dict_shaped[:, :, 0]

                all_obs_flat = obs_a0.reshape((N * num_envs,) + obs_shape)

                def get_probs_for_all_obs(state, all_obs):
                    pi, _ = state.apply_fn(state.params, all_obs)
                    return pi.probs

                probs_all = jax.vmap(get_probs_for_all_obs, in_axes=(0, None))(pop_states, all_obs_flat)
                mean_probs = probs_all.mean(axis=0)
                mean_probs = mean_probs.reshape(N, num_envs, action_dim)

                act_a0_oh = jax.nn.one_hot(act_a0, action_dim)
                mean_prob_taken = (mean_probs * act_a0_oh).sum(-1)
                pe_bonus = -jnp.log(mean_prob_taken + 1e-8)

                task_reward = rewards["agent_0"]
                shaped_reward = infos.get("shaped_reward", {}).get("agent_0", jnp.zeros_like(task_reward))
                aug_reward = task_reward + shaped_reward + alpha * pe_bonus

                val_a0 = vals_shaped[:, :, 0]
                logp_a0 = log_probs_shaped[:, :, 0]
                done_a0 = dones["agent_0"]

                transition = Transition(
                    done=done_a0,
                    action=act_a0,
                    value=val_a0,
                    reward=aug_reward,
                    log_prob=logp_a0,
                    obs=obs_a0,
                    hstate=None,   # MLP — no hstate
                    info=infos,
                )
                inner_carry = (pop_states, new_env_states, new_obs_stack, rng)
                return inner_carry, transition

            (pop_states, env_states, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step,
                (pop_states, env_states, last_obs, rng),
                None,
                length=num_steps
            )

            # ---- Compute GAE for each population agent ----------
            obs_last_a0 = last_obs[:, :, 0, :]

            def get_last_val(state, obs):
                _, val = state.apply_fn(state.params, obs)
                return val

            last_vals = jax.vmap(get_last_val)(pop_states, obs_last_a0)

            traj_transposed = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1) if x is not None else None, traj_batch
            )
            advantages, targets = jax.vmap(compute_gae, in_axes=(0, 0, None, None))(
                traj_transposed, last_vals, gamma, gae_lambda
            )

            # ---- PPO update for each population agent ----------
            batch_size = num_steps * num_envs
            minibatch_size = batch_size // num_minibatches

            def update_one_agent(state, traj_n, adv_n, tgt_n, rng):
                flat_traj = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]) if (x is not None and hasattr(x, 'shape')) else x,
                    traj_n
                )
                flat_adv = adv_n.reshape(batch_size)
                flat_tgt = tgt_n.reshape(batch_size)

                def _epoch(state, rng):
                    perm = jax.random.permutation(rng, batch_size)

                    def _minibatch(state, indices):
                        mb_traj = jax.tree_util.tree_map(
                            lambda x: jnp.take(x, indices, axis=0) if (x is not None and hasattr(x, 'shape')) else x,
                            flat_traj
                        )
                        mb_adv = jnp.take(flat_adv, indices, axis=0)
                        mb_tgt = jnp.take(flat_tgt, indices, axis=0)
                        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
                        (loss, aux), grads = grad_fn(
                            state.params, state.apply_fn,
                            mb_traj, mb_adv, mb_tgt,
                            clip_eps, vf_coef, ent_coef
                        )
                        state = state.apply_gradients(grads=grads)
                        return state, (loss, aux)

                    indices = perm.reshape(num_minibatches, minibatch_size)
                    state, loss_info = jax.lax.scan(_minibatch, state, indices)
                    return state, loss_info

                epoch_rngs = jax.random.split(rng, update_epochs)
                state, loss_info = jax.lax.scan(_epoch, state, epoch_rngs)
                return state, loss_info

            update_rngs = jax.random.split(rng, N)
            pop_states, loss_info = jax.vmap(update_one_agent)(
                pop_states,
                traj_transposed,
                advantages,
                targets,
                update_rngs
            )
            rng, _ = jax.random.split(rng)
            carry = (pop_states, env_states, last_obs, rng)
            return carry, loss_info

        init_obs_stack = jnp.stack(
            [init_obs["agent_0"], init_obs["agent_1"]], axis=2
        )
        carry = (pop_states, init_env_state, init_obs_stack, rng)
        carry, loss_history = jax.lax.scan(
            _update_step, carry, None, length=num_updates
        )
        pop_states, _, _, _ = carry
        return pop_states, loss_history

    return train_population


# ---------------------------------------------------------------------------
# 6. PHASE 2 — EGO AGENT TRAINING WITH PRIORITIZED SAMPLING (RNN ego)
# Ego agent trains against FROZEN population partners.
# ---------------------------------------------------------------------------
def make_phase2_train(config, env, pop_params):
    """
    Returns `train_ego(rng) -> ego_train_state`
    pop_params: pytree with leading axis of size N (frozen after Phase 1)
    Partner sampling:
      - Maintain avg_returns[N] tracking recent avg return per partner
      - p(i) ∝ rank(1 / avg_returns[i])^β
      - Recompute sampling weights in Python every update
    Ego agent is an ActorCriticRNN (LSTM-based).
    """
    N = config["POPULATION_SIZE"]
    obs_shape = env.observation_space().shape
    action_dim = env.action_space().n
    num_envs = config["NUM_ENVS_EGO"]
    num_steps = config["NUM_STEPS"]
    num_updates = config["PHASE2_TOTAL_TIMESTEPS"] // num_steps // num_envs
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS_EGO"]
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    clip_eps = config["CLIP_EPS"]
    vf_coef = config["VF_COEF"]
    ent_coef = config["ENT_COEF"]
    lr = config["LR_EGO"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    beta = config["PRIORITY_BETA"]
    hidden_size = config["HIDDEN_SIZE"]

    # RNN ego network
    ego_network = ActorCriticRNN(
        action_dim=action_dim,
        hidden_size=hidden_size,
        activation=config["ACTIVATION"],
        obs_shape=obs_shape,
    )

    # MLP partner network (frozen population agents)
    partner_network = ActorCritic(action_dim, config["HIDDEN_SIZE"], config["ACTIVATION"])

    obs_dim = int(np.prod(obs_shape))

    # ------------------------------------------------------------------
    # 6a. JIT-compiled rollout: ego (RNN) vs fixed partner (MLP)
    # ------------------------------------------------------------------
    @jax.jit
    def collect_rollout(ego_state, partner_params, env_states, last_obs, last_done, hstate, rng):
        """
        Collect num_steps transitions.
        ego plays as agent_1, partner (fixed MLP params) plays as agent_0.
        Ego uses RNN; hstate is carried across steps and reset on episode done.
        """
        def _step(carry, _):
            ego_state, env_states, last_obs, last_done, hstate, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)

            obs0 = last_obs["agent_0"]  # (num_envs, *obs_shape)
            obs1 = last_obs["agent_1"]  # (num_envs, *obs_shape)

            # Ego RNN forward pass
            # obs1 needs shape (1, num_envs, obs_dim) for the RNN
            obs1_flat = obs1.reshape((num_envs, obs_dim))
            obs1_batched = obs1_flat[None]          # (1, num_envs, obs_dim)
            done_batched = last_done[None]           # (1, num_envs)

            new_hstate, (pi_ego, val_ego) = ego_state.apply_fn(
                ego_state.params, hstate, obs1_batched, done_batched
            )
            pi_ego = jax.tree_util.tree_map(lambda x: x.squeeze(0), pi_ego)
            val_ego = val_ego.squeeze(0)  # (num_envs,)

            # Partner MLP forward pass (stateless)
            pi_partner, _ = partner_network.apply(partner_params, obs0)

            a_rngs_ego = jax.random.split(a_rng, num_envs)
            a_rngs_partner = jax.random.split(a_rng, num_envs)
            act_ego = jax.vmap(lambda d, k: d.sample(seed=k))(pi_ego, a_rngs_ego)
            act_partner = jax.vmap(lambda d, k: d.sample(seed=k))(pi_partner, a_rngs_partner)
            logp_ego = pi_ego.log_prob(act_ego)

            acts = {"agent_0": act_partner, "agent_1": act_ego}
            s_rngs = jax.random.split(s_rng, num_envs)
            new_obs, new_env_states, rewards, dones, infos = jax.vmap(env.step)(
                s_rngs, env_states, acts
            )

            task_r = rewards["agent_1"]
            shaped_r = infos.get("shaped_reward", {}).get("agent_1", jnp.zeros(num_envs))
            total_r = task_r + shaped_r

            new_done = dones["__all__"].astype(jnp.float32)

            transition = Transition(
                done=new_done,
                action=act_ego,
                value=val_ego,
                reward=total_r,
                log_prob=logp_ego,
                obs=obs1_flat,      # store flattened obs (num_envs, obs_dim)
                hstate=hstate,      # hstate at the START of this step (for PPO loss)
                info=infos,
            )
            carry = (ego_state, new_env_states, new_obs, new_done, new_hstate, rng)
            return carry, (transition, rewards["agent_1"])

        (ego_state, env_states, last_obs, last_done, hstate, rng), (traj, ep_task_rews) = jax.lax.scan(
            _step,
            (ego_state, env_states, last_obs, last_done, hstate, rng),
            None,
            length=num_steps
        )

        # Bootstrap value with RNN
        last_obs1_flat = last_obs["agent_1"].reshape((num_envs, obs_dim))
        _, (_, last_val) = ego_state.apply_fn(
            ego_state.params, hstate, last_obs1_flat[None], last_done[None]
        )
        last_val = last_val.squeeze(0)  # (num_envs,)

        advantages, targets = compute_gae(traj, last_val, gamma, gae_lambda)

        # ------------------------------------------------------------------
        # Compute actual episodic return using LogWrapper info keys.
        # returned_episode_returns: (num_steps, num_envs) — cumulative return
        #   at the step each episode ended (0 otherwise).
        # returned_episode: (num_steps, num_envs) bool — True on episode end.
        # We average only over completed episodes; fall back to summing
        # per-step rewards if no episode completed in this rollout window.
        #
        # NOTE: the Python-level `if` branches are resolved at JIT trace time,
        # so we use jnp.where for the "no episodes completed" fallback so that
        # both branches are always well-typed JAX expressions.
        # LogWrapper may nest these keys under the agent name, so we check
        # the flat key first and then the agent_1-namespaced key.
        # ------------------------------------------------------------------
        info = traj.info
        ep_returns  = info.get("returned_episode_returns", None)
        ep_done_mask = info.get("returned_episode", None)

        # Some LogWrapper versions nest stats under the agent key
        if ep_returns is None:
            ep_returns   = (info.get("agent_1") or {}).get("returned_episode_returns", None)
            ep_done_mask = (info.get("agent_1") or {}).get("returned_episode", None)

        if ep_returns is not None and ep_done_mask is not None:
            num_completed = ep_done_mask.sum()
            episode_return = jnp.where(
                num_completed > 0,
                (ep_returns * ep_done_mask).sum() / (num_completed + 1e-8),
                ep_task_rews.sum(axis=0).mean(),   # fallback: no episode finished
            )
        else:
            # Key not present at all — sum per-step rewards across the window
            episode_return = ep_task_rews.sum(axis=0).mean()

        avg_return = episode_return

        return ego_state, env_states, last_obs, last_done, hstate, rng, traj, advantages, targets, avg_return, ep_task_rews

    # ------------------------------------------------------------------
    # 6b. JIT-compiled PPO update for ego (RNN-aware)
    # ------------------------------------------------------------------
    @jax.jit
    def ppo_update(ego_state, traj: Transition, advantages, targets, rng):
        """
        PPO update for the RNN ego agent.
        Uses the stored hstate[0] from each minibatch as the initial carry,
        matching the pattern from the reference RNN training code.
        """
        minibatch_size = num_envs // num_minibatches

        def _update_epoch(update_state, unused):
            def _update_minibatch(ego_state, batch_info):
                traj_mb, advantages_mb, targets_mb = batch_info

                def _loss_fn(params, traj_mb, gae, targets):
                    # Use the first hstate in the minibatch as the initial carry
                    init_hstate = jax.tree_util.tree_map(
                        lambda x: x[0], traj_mb.hstate
                    )
                    # traj_mb.obs: (num_steps, minibatch_size, obs_dim)
                    # traj_mb.done: (num_steps, minibatch_size)
                    _, (pi, value) = ego_network.apply(
                        params,
                        init_hstate,
                        traj_mb.obs,
                        traj_mb.done,
                    )
                    log_prob = pi.log_prob(traj_mb.action)

                    value_pred_clipped = traj_mb.value + (value - traj_mb.value).clip(
                        -clip_eps, clip_eps
                    )
                    value_loss = 0.5 * jnp.maximum(
                        jnp.square(value - targets),
                        jnp.square(value_pred_clipped - targets),
                    ).mean()

                    ratio = jnp.exp(log_prob - traj_mb.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor = -jnp.minimum(
                        ratio * gae_norm,
                        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm,
                    ).mean()

                    entropy = pi.entropy().mean()
                    total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(ego_state.params, traj_mb, advantages_mb, targets_mb)
                ego_state = ego_state.apply_gradients(grads=grads)
                return ego_state, total_loss

            ego_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, num_envs)

            # Shuffle along env axis (axis=1 in the trajectory: (num_steps, num_envs, ...))
            shuffled_traj = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1) if (x is not None and hasattr(x, 'shape') and x.ndim >= 2) else x,
                traj_batch,
            )
            shuffled_advantages = jnp.take(advantages, permutation, axis=1)
            shuffled_targets = jnp.take(targets, permutation, axis=1)

            # Split envs into minibatches
            def split_envs(x):
                # x shape: (num_steps, num_envs, ...) or (num_steps, num_envs)
                s = x.shape
                x = x.reshape((s[0], num_minibatches, minibatch_size) + s[2:])
                # -> (num_minibatches, num_steps, minibatch_size, ...)
                return x.transpose((1, 0, 2) + tuple(range(3, x.ndim)))

            minibatch_traj = jax.tree_util.tree_map(
                lambda x: split_envs(x) if (x is not None and hasattr(x, 'shape') and x.ndim >= 2) else x,
                shuffled_traj,
            )
            minibatch_advantages = split_envs(shuffled_advantages)
            minibatch_targets = split_envs(shuffled_targets)

            ego_state, total_loss = jax.lax.scan(
                _update_minibatch, ego_state, (minibatch_traj, minibatch_advantages, minibatch_targets)
            )
            update_state = (ego_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (ego_state, traj, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, update_epochs
        )
        ego_state = update_state[0]
        rng = update_state[-1]
        return ego_state, rng, loss_info

    # ------------------------------------------------------------------
    # 6c. Compute prioritized sampling weights (pure numpy, outside jit)
    # ------------------------------------------------------------------
    def compute_sampling_weights(avg_returns: np.ndarray, beta: float) -> np.ndarray:
        safe_returns = np.maximum(avg_returns, 1e-8)
        order = np.argsort(safe_returns)
        ranks = np.empty(N, dtype=np.float32)
        ranks[order] = np.arange(1, N + 1, dtype=np.float32)
        weights = ranks ** beta
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # 6d. Main Phase 2 loop (Python-level loop over updates)
    # ------------------------------------------------------------------
    def train_ego(rng):
        # Init ego RNN agent
        rng, init_rng = jax.random.split(rng)

        # RNN init requires (hidden, obs, dones) with correct shapes
        init_obs_rnn = jnp.zeros((1, num_envs, obs_dim))
        init_carry = ActorCriticRNN.initialize_carry(hidden_size, num_envs)
        init_done_rnn = jnp.zeros((1, num_envs))

        ego_params = ego_network.init(init_rng, init_carry, init_obs_rnn, init_done_rnn)
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr, eps=1e-5)
        )
        ego_state = TrainState.create(apply_fn=ego_network.apply, params=ego_params, tx=tx)

        # Init environments
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        last_obs, env_states = jax.vmap(env.reset)(reset_rngs)

        # Init RNN carry and done flags
        hstate = ActorCriticRNN.initialize_carry(hidden_size, num_envs)
        last_done = jnp.zeros((num_envs,))

        avg_returns = np.ones(N, dtype=np.float32)
        ema_alpha = 0.1
        partner_indices = np.arange(N)

        for update_idx in range(num_updates):
            weights = compute_sampling_weights(avg_returns, beta)
            partner_idx = int(np.random.choice(partner_indices, p=weights))
            partner_params = jax.tree_util.tree_map(
                lambda x: x[partner_idx], pop_params
            )

            rng, rollout_rng, update_rng = jax.random.split(rng, 3)

            (ego_state, env_states, last_obs, last_done, hstate, rng,
             traj, adv, tgt, avg_ret, ep_task_rews) = collect_rollout(
                ego_state, partner_params, env_states, last_obs, last_done, hstate, rollout_rng
            )

            avg_returns[partner_idx] = (
                (1 - ema_alpha) * avg_returns[partner_idx] + ema_alpha * float(avg_ret)
            )

            ego_state, rng, loss_info = ppo_update(ego_state, traj, adv, tgt, update_rng)

            # Reset hstate and last_done when switching partners (clean slate for new partner)
            hstate = ActorCriticRNN.initialize_carry(hidden_size, num_envs)
            last_done = jnp.zeros((num_envs,))

            reward_m = ep_task_rews.mean()
            if update_idx % 50 == 0:
                print(f" Phase2 update {update_idx}/{num_updates} | "
                      f"partner={partner_idx} | episode_return={float(avg_ret):.2f} | "
                      f"weights={np.round(weights, 3)}")
                if config.get("USE_WANDB"):
                    wandb.log({
                        "phase2/update": update_idx,
                        "phase2/avg_return_with_partner": float(reward_m),
                        "phase2/episode_return_with_partner": float(avg_ret),
                        "phase2/partner_idx": partner_idx,
                        "phase2/min_avg_return_across_pop": float(avg_returns.min()),
                    })

        return ego_state

    return train_ego


# ---------------------------------------------------------------------------
# 7. POST-PHASE-1 ROLLOUT VISUALISATION  [UNCHANGED]
# ---------------------------------------------------------------------------
def get_single_rollout(params, network, env_raw, rng, max_steps=400):
    """
    Run one episode with `params` controlling BOTH agents.
    """
    rng, reset_rng = jax.random.split(rng)
    obs, state = env_raw.reset(reset_rng)
    state_seq = [state]
    total_reward = 0.0
    for _ in range(max_steps):
        rng, key_a0, key_a1 = jax.random.split(rng, 3)
        obs0 = obs["agent_0"]
        obs1 = obs["agent_1"]
        pi0, _ = network.apply(params, obs0)
        pi1, _ = network.apply(params, obs1)
        act0 = int(pi0.sample(seed=key_a0))
        act1 = int(pi1.sample(seed=key_a1))
        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, _ = env_raw.step(
            step_rng, state, {"agent_0": act0, "agent_1": act1}
        )
        state_seq.append(state)
        total_reward += float(rewards["agent_0"])
        if dones["__all__"]:
            break
    return state_seq, total_reward


def render_state_seq(state_seq, env_raw, tile_size=32):
    """
    Convert a list of raw OvercookedState objects into a uint8 video array.
    """
    padding = env_raw.agent_view_size - 2
    frames = []
    for state in state_seq:
        grid = np.asarray(
            state.maze_map[padding:-padding, padding:-padding, :]
        )
        frame = OvercookedVisualizer._render_grid(
            grid,
            tile_size=tile_size,
            highlight_mask=None,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv,
        )
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_population_rollouts(pop_states, config, save_dir, rng, use_wandb=False):
    """
    Render one self-play episode for each population agent and save as MP4.
    """
    N = config["POPULATION_SIZE"]
    action_dim = config["_ACTION_DIM"]
    network = ActorCritic(action_dim, config["HIDDEN_SIZE"], config["ACTIVATION"])
    env_raw = make_env_raw(config["LAYOUT_NAME"], config, config["ENV_NAME"])
    os.makedirs(save_dir, exist_ok=True)
    video_paths = []
    for i in range(N):
        rng, rollout_rng = jax.random.split(rng)
        agent_params = jax.tree_util.tree_map(lambda x: x[i], pop_states.params)
        print(f" Rendering rollout for population agent {i}/{N-1} ...", end=" ", flush=True)
        state_seq, total_reward = get_single_rollout(
            agent_params, network, env_raw, rollout_rng, max_steps=400
        )
        print(f"total_reward={total_reward:.1f}, frames={len(state_seq)}")
        frames = render_state_seq(state_seq, env_raw, tile_size=32)
        video_path = os.path.join(save_dir, f"agent_{i}_rollout.mp4")
        imageio.mimwrite(video_path, frames, fps=8, quality=8)
        video_paths.append(video_path)
        print(f" Saved → {video_path}")
        if use_wandb:
            wandb.log({
                f"phase1/agent_{i}_rollout": wandb.Video(video_path, fps=8, format="mp4"),
                f"phase1/agent_{i}_total_reward": total_reward,
            })
    return video_paths


# ---------------------------------------------------------------------------
# 8. ENVIRONMENT SETUP HELPER  [UNCHANGED]
# ---------------------------------------------------------------------------
def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config["layout_name"] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["ENV_NAME"] == "overcooked":
        if "9" in layout_name:
            def reset_env(key):
                def reset_sub_dict(key, fn):
                    key, subkey = jax.random.split(key)
                    sampled_layout_dict = fn(subkey, ik=True)
                    temp_o, temp_s = env.custom_reset(
                        key, layout=sampled_layout_dict,
                        random_reset=False, shuffle_inv_and_pot=False
                    )
                    key, subkey = jax.random.split(key)
                    return (temp_o, temp_s), key

                asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
                coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
                counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
                forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
                cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
                layout_resets = [
                    asymm_reset, coord_ring_reset, counter_circuit_reset,
                    forced_coord_reset, cramped_room_reset,
                ]
                stacked = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
                index = jax.random.randint(key, (), minval=0, maxval=5)
                return jax.tree_map(lambda x: x[index], stacked)

            @scan_tqdm(100)
            def gen_held_out(runner_state, unused):
                (i,) = runner_state
                _, ho_state = reset_env(jax.random.key(i))
                res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
                return (i + 1,), res

            _, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
            ho_goal, ho_wall, ho_pot = [], [], []
            for ln, ld in overcooked_layouts.items():
                if "9" in ln:
                    _, ho_state = env.custom_reset(
                        jax.random.PRNGKey(0),
                        random_reset=False, shuffle_inv_and_pot=False, layout=ld
                    )
                    ho_goal.append(ho_state.goal_pos)
                    ho_wall.append(ho_state.wall_map)
                    ho_pot.append(ho_state.pot_pos)
            env.held_out_goal = jnp.concatenate([res[0], jnp.stack(ho_goal)], axis=0)
            env.held_out_wall = jnp.concatenate([res[1], jnp.stack(ho_wall)], axis=0)
            env.held_out_pot = jnp.concatenate([res[2], jnp.stack(ho_pot)], axis=0)
        else:
            env.check_held_out = False
    return env


def make_env(config):
    """Return LogWrapper-wrapped env for training."""
    config_env = OmegaConf.load("../config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"] + "_9"
    config_env["SEED"] = config["SEED"]
    env = initialize_environment(config_env)
    return LogWrapper(env)


def make_env_raw(layout_name, config, env_name="overcooked"):
    """Return unwrapped env for rollout visualisation."""
    config_env = OmegaConf.load("../config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"] + "_9"
    config_env["SEED"] = config["SEED"]
    return initialize_environment(config_env)


# ---------------------------------------------------------------------------
# 9. MAIN ENTRY POINT  [UNCHANGED]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--env_name", type=str, default="overcooked")
    # Population
    parser.add_argument("--population_size", type=int, default=22)
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Weight of Population Entropy bonus (Table 1 in paper)")
    parser.add_argument("--priority_beta", type=float, default=3.0,
                        help="Prioritization exponent β (Appendix D)")
    # Training
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_envs_ego", type=int, default=100,
                        help="Number of envs for Phase 2 ego training (independent of Phase 1)")
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--phase1_total_timesteps", type=int, default=int(35e6))
    parser.add_argument("--phase2_total_timesteps", type=int, default=int(145e6))
    parser.add_argument("--update_epochs_population", type=int, default=4,
                        help="PPO update epochs for Phase 1 population agents")
    parser.add_argument("--update_epochs_ego", type=int, default=15,
                        help="PPO update epochs for Phase 2 ego agent")
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--lr_population", type=float, default=3e-5,
                        help="Learning rate for Phase 1 population agents")
    parser.add_argument("--lr_ego", type=float, default=3e-4,
                        help="Learning rate for Phase 2 ego agent")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=120)
    parser.add_argument("--activation", type=str, default="tanh")
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--save_dir", type=str, default="phase2/mep/")
    args = parser.parse_args()

    config = {
        "ENV_NAME": args.env_name,
        "ENV_KWARGS": {"layout": args.layout},
        "LAYOUT_NAME": args.layout,
        "POPULATION_SIZE": args.population_size,
        "ALPHA": args.alpha,
        "PRIORITY_BETA": args.priority_beta,
        "NUM_ENVS": args.num_envs,
        "NUM_ENVS_EGO": args.num_envs_ego,
        "NUM_STEPS": args.num_steps,
        "PHASE1_TOTAL_TIMESTEPS": args.phase1_total_timesteps,
        "PHASE2_TOTAL_TIMESTEPS": args.phase2_total_timesteps,
        "UPDATE_EPOCHS_POPULATION": args.update_epochs_population,
        "UPDATE_EPOCHS_EGO": args.update_epochs_ego,
        "NUM_MINIBATCHES": args.num_minibatches,
        "LR_POPULATION": args.lr_population,
        "LR_EGO": args.lr_ego,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "VF_COEF": args.vf_coef,
        "ENT_COEF": args.ent_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "HIDDEN_SIZE": args.hidden_size,
        "ACTIVATION": args.activation,
        "SEED": args.seed,
        "layout": args.layout,
        "USE_WANDB": not args.no_wandb,
    }

    if not args.no_wandb:
        wandb.init(project="mep_jax", config=config,
                   group=f"mep_{args.layout}", name=f"seed{args.seed}")

    env = make_env(config)
    config["_ACTION_DIM"] = env.action_space().n

    rng = jax.random.PRNGKey(args.seed)

    # -----------------------------------------------------------------------
    # PHASE 1: Train population
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 1: Training Maximum Entropy Population")
    print(f" N={config['POPULATION_SIZE']} agents | "
          f"α={config['ALPHA']} | "
          f"{config['PHASE1_TOTAL_TIMESTEPS']:.0e} timesteps")
    print("=" * 60)

    rng, phase1_rng = jax.random.split(rng)
    train_population = jax.jit(make_phase1_train(config, env))
    pop_states, phase1_loss = train_population(phase1_rng)
    print("Phase 1 complete.")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {"population": pop_states, "config": config}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    pop_save_path = os.path.join(os.getcwd(), args.save_dir, "population")
    if not os.path.exists(pop_save_path):
        orbax_checkpointer.save(pop_save_path, ckpt, save_args=save_args)
    else:
        shutil.rmtree(pop_save_path)
        orbax_checkpointer.save(pop_save_path, ckpt, save_args=save_args)
    print(f"Population saved to {pop_save_path}")

    # -----------------------------------------------------------------------
    # PHASE 1 ROLLOUT VIDEOS
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Rendering Phase 1 population rollouts ...")
    print("=" * 60)
    rng, render_rng = jax.random.split(rng)
    video_dir = os.path.join(args.save_dir, "phase1_videos")
    render_population_rollouts(
        pop_states,
        config,
        save_dir=video_dir,
        rng=render_rng,
        use_wandb=not args.no_wandb,
    )
    print(f"Videos written to {video_dir}")

    # -----------------------------------------------------------------------
    # PHASE 2: Train ego agent (RNN)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 2: Training RNN Ego Agent with Prioritized Sampling")
    print(f" β={config['PRIORITY_BETA']} | "
          f"{config['PHASE2_TOTAL_TIMESTEPS']:.0e} timesteps")
    print("=" * 60)

    rng, phase2_rng = jax.random.split(rng)
    pop_params_frozen = jax.tree_util.tree_map(lambda x: x, pop_states.params)
    train_ego = make_phase2_train(config, env, pop_params_frozen)
    ego_state = train_ego(phase2_rng)
    print("Phase 2 complete.")

    ego_ckpt = {"ego": ego_state, "config": config}
    ego_save_path = os.path.join(os.getcwd(), str(args.save_dir), args.layout, str(args.seed))
    if not os.path.exists(ego_save_path):
        ego_save_args = orbax_utils.save_args_from_target(ego_ckpt)
        orbax_checkpointer.save(ego_save_path, ego_ckpt, save_args=ego_save_args)
    else:
        shutil.rmtree(ego_save_path)
        ego_save_args = orbax_utils.save_args_from_target(ego_ckpt)
        orbax_checkpointer.save(ego_save_path, ego_ckpt, save_args=ego_save_args)
    print(f"Ego agent saved to {ego_save_path}")

    if not args.no_wandb:
        wandb.finish()
