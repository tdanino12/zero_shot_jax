import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
import collections
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from flax.training import orbax_utils
import orbax
import matplotlib.pyplot as plt
import os
import argparse
import wandb
import haiku as hk
import shutil
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

# ============================================================================
# EMPOWERMENT BUFFER
# ============================================================================
Batch = collections.namedtuple(
    "Batch",
    ["state", "action_robot", "action_human", "next_state", "future_state", "done", "idx", "future_idx", "reward"],
)


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key

            asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
            coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree_map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset

        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res

        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, layout_dict in overcooked_layouts.items():
            if "9" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=layout_dict)
                ho_goal.append(ho_state.goal_pos)
                ho_wall.append(ho_state.wall_map)
                ho_pot.append(ho_state.pot_pos)
        ho_goal = jnp.stack(ho_goal, axis=0)
        ho_wall = jnp.stack(ho_wall, axis=0)
        ho_pot = jnp.stack(ho_pot, axis=0)
        ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
        ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
        ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
        env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)

    return env


class ContrastiveBuffer(NamedTuple):
    state_buffer: jnp.ndarray
    horizon_buffer: jnp.ndarray
    action_human_buffer: jnp.ndarray
    action_robot_buffer: jnp.ndarray
    reward_buffer: jnp.ndarray
    next_pos: int
    max_pos: int
    buflen: int
    obs_dim: int
    size: int
    gamma: float
    batch_size: int

    @staticmethod
    def create(obs_dim, size, gamma, batch_size):
        return ContrastiveBuffer(
            state_buffer=jnp.zeros((size, obs_dim)),
            horizon_buffer=jnp.zeros((size,), dtype=jnp.int32),
            action_human_buffer=jnp.zeros((size,), dtype=jnp.int32),
            action_robot_buffer=jnp.zeros((size,), dtype=jnp.int32),
            reward_buffer=jnp.zeros((size,)),
            next_pos=0, max_pos=0, buflen=0,
            obs_dim=obs_dim, size=size, gamma=gamma, batch_size=batch_size,
        )

    def extend(self, obs, action_robot, action_human, reward):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        t_left = jnp.arange(len(obs))[::-1]
        next_pos = jax.lax.select(self.next_pos + len(obs) > self.size, 0, self.next_pos)
        state_buffer = jax.lax.dynamic_update_slice_in_dim(self.state_buffer, obs, next_pos, axis=0)
        horizon_buffer = jax.lax.dynamic_update_slice_in_dim(self.horizon_buffer, t_left, next_pos, axis=0)
        action_robot_buffer = jax.lax.dynamic_update_slice_in_dim(self.action_robot_buffer, action_robot, next_pos, axis=0)
        action_human_buffer = jax.lax.dynamic_update_slice_in_dim(self.action_human_buffer, action_human, next_pos, axis=0)
        reward_buffer = jax.lax.dynamic_update_slice_in_dim(self.reward_buffer, reward, next_pos, axis=0)
        next_pos = next_pos + len(obs)
        buflen = jnp.minimum(self.buflen + len(obs), self.size)
        next_pos = jax.lax.select(next_pos >= self.size, 0, next_pos)
        max_pos = jnp.maximum(self.max_pos, next_pos)
        return self._replace(
            state_buffer=state_buffer, horizon_buffer=horizon_buffer,
            action_robot_buffer=action_robot_buffer, action_human_buffer=action_human_buffer,
            reward_buffer=reward_buffer, next_pos=next_pos, buflen=buflen, max_pos=max_pos,
        )

    def _sample(self, key):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(key, (), 0, self.max_pos)
        delta = jax.random.geometric(subkey, 1 - self.gamma)
        state = self.state_buffer[idx]
        action_robot = self.action_robot_buffer[idx]
        action_human = self.action_human_buffer[idx]
        reward = self.reward_buffer[idx]
        end = self.horizon_buffer[idx]
        delta = jnp.minimum(delta, end)
        future_idx = (idx + delta) % self.buflen
        future_state = self.state_buffer[future_idx]
        done = idx + 1 == self.next_pos
        next_state = state * done + self.state_buffer[idx + 1] * (1 - done)
        return dict(
            state=state, action_robot=action_robot, action_human=action_human,
            next_state=next_state, future_state=future_state, done=done,
            idx=idx, future_idx=future_idx, reward=reward,
        )

    def sample(self, key, batch_size):
        keys = jax.random.split(key, batch_size)
        return Batch(**jax.vmap(self._sample)(keys))


# ============================================================================
# EMPOWERMENT NETWORKS & FUNCTIONS
# ============================================================================
def make_repr_fn(a_dim, repr_dim, phi_norm, psi_norm):
    def repr_fn(s, ar, ah, g):
        phi = hk.Sequential([
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(repr_dim, w_init=hk.initializers.Orthogonal(1.0), b_init=hk.initializers.Constant(0.0)),
        ])
        psi = hk.Sequential([
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(repr_dim, w_init=hk.initializers.Orthogonal(1.0), b_init=hk.initializers.Constant(0.0)),
        ])
        ar = jax.nn.one_hot(ar, a_dim)
        ah = jax.nn.one_hot(ah, a_dim)
        sa = jnp.concatenate([s, ar, ah], axis=-1)
        s_only = jnp.concatenate([s, ar, jnp.zeros_like(ah)], axis=-1)
        sa, s_only, g = phi(sa), phi(s_only), psi(g)
        if phi_norm:
            sa = sa / jnp.linalg.norm(sa, axis=-1, keepdims=True)
            s_only = s_only / jnp.linalg.norm(s_only, axis=-1, keepdims=True)
        if psi_norm:
            g = g / jnp.linalg.norm(g, axis=-1, keepdims=True)
        if phi_norm and psi_norm:
            temp = hk.get_parameter("temp", shape=(), init=hk.initializers.Constant(1.0))
        else:
            temp = 1.0
        return sa, s_only, g, temp
    return repr_fn


def compute_empowerment_reward(repr_fn, repr_params, s, ar, ah, g, reward_type, repr_dim):
    sa_phi, s_phi, g_psi, temp = repr_fn(repr_params, s, ar, ah, g)
    if reward_type == "dot":
        return jnp.sum((sa_phi - s_phi) * g_psi, axis=-1)
    elif reward_type == "norm":
        return jnp.linalg.norm(sa_phi, axis=-1) - jnp.linalg.norm(s_phi, axis=-1)
    elif reward_type == "diff":
        return jnp.linalg.norm(sa_phi - s_phi, axis=-1)
    else:
        raise ValueError(f"Invalid reward type {reward_type}")


def contrastive_loss(repr_fn, repr_params, s, ar, ah, g, a_dim, repr_dim, psi_reg):
    sa_phi, s_phi, g_psi, tau = repr_fn(repr_params, s, ar, ah, g)

    def infonce(phi, psi):
        logits = jnp.sum(phi[:, None, :] * psi[None, :, :], axis=-1) / repr_dim / tau
        logits1 = jax.nn.log_softmax(logits, axis=1)
        logits2 = jax.nn.log_softmax(logits, axis=0)
        loss = -jnp.mean(jnp.diag(logits1)) - jnp.mean(jnp.diag(logits2))
        loss += psi_reg * jnp.mean(psi ** 2) ** 2
        return loss, {}

    batch = s.shape[0]
    g_tiled = jnp.concatenate([g_psi, g_psi], axis=0)
    s_tiled = jnp.concatenate([s_phi, sa_phi], axis=0)
    loss, info = infonce(s_tiled, g_tiled)
    return loss, info


# ============================================================================
# EMPOWERMENT STATE
# ============================================================================
class EmpowermentState(NamedTuple):
    repr_params: dict
    repr_opt_state: optax.OptState
    buffer: ContrastiveBuffer
    repr_buffer: ContrastiveBuffer
    emp_step: int
    rng: jax.random.PRNGKey


# ============================================================================
# PPO STRUCTURES
# ============================================================================
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ============================================================================
# ROLLOUT & RENDER
# ============================================================================
def get_rollout(train_state_0, train_state_1, config):
    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"] + "_9"
    config_env["SEED"] = config["seed"]
    env = initialize_environment(config_env)

    network = ActorCritic(env.action_space().n, activation=config["activation"])
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)
        obs_flat = {k: v.flatten() for k, v in obs.items()}
        # agent_0 policy now acts as agent_1 in env, agent_1 policy acts as agent_0 in env
        pi_0, _ = network.apply(train_state_0.params, obs_flat["agent_1"])
        pi_1, _ = network.apply(train_state_1.params, obs_flat["agent_0"])
        actions = {
            "agent_0": pi_1.sample(seed=key_a1),
            "agent_1": pi_0.sample(seed=key_a0),
        }
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
        state_seq.append(state)
    return state_seq


def render_state_seq(state_seq, env):
    padding = env.agent_view_size - 2
    def get_frame(state):
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        return OvercookedVisualizer._render_grid(
            grid, tile_size=32, highlight_mask=None,
            agent_dir_idx=state.agent_dir_idx, agent_inv=state.agent_inv)
    frames = np.stack([get_frame(s) for s in state_seq])   # (T, H, W, C)
    return np.moveaxis(frames, -1, 1)                       # (T, C, H, W) for wandb.Video


# ============================================================================
# TRAINING
# ============================================================================
def make_train(config, pretrained_params):
    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"] + "_9"
    config_env["SEED"] = config["seed"]
    env = initialize_environment(config_env)

    config["NUM_ACTORS"] = env.num_agents * config["num_envs"]
    config["NUM_UPDATES"] = config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    # MINIBATCH_SIZE is per-agent
    config["MINIBATCH_SIZE"] = config["num_envs"] * config["num_steps"] // config["num_minibatches"]

    env = LogWrapper(env)

    # batchify layout: [agent_0_env0..agent_0_envN, agent_1_env0..agent_1_envN]
    # n = number of envs = number of agents per slot
    n = config["num_envs"]

    def train(rng):
        network = ActorCritic(env.action_space().n, activation=config["activation"])
        rng, _rng0, _rng1 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.observation_space().shape).flatten()

        def make_tx(lr):
            def schedule(count):
                frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["NUM_UPDATES"]
                return lr * frac
            return optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=schedule if config["anneal_lr"] else lr, eps=1e-5),
            )

        tx_0 = make_tx(3e-5)
        tx_1 = make_tx(3e-5)

        # train_state_0: the ego agent being trained — operates in agent_1's env slot
        train_state_0 = TrainState.create(apply_fn=network.apply, params=network.init(_rng0, init_x), tx=tx_0)
        # train_state_1: pretrained partner — operates in agent_0's env slot
        train_state_1 = TrainState.create(apply_fn=network.apply, params=pretrained_params, tx=tx_1)

        # ── EMPOWERMENT INIT ────────────────────────────────────────────────
        obs_dim = init_x.shape[0]
        a_dim = env.action_space().n

        repr_fn_raw = make_repr_fn(a_dim, config["repr_dim"], config["phi_norm"], config["psi_norm"])
        repr_fn = hk.without_apply_rng(hk.transform(repr_fn_raw))

        rng, _rng = jax.random.split(rng)
        repr_params = repr_fn.init(_rng, jnp.zeros((1, obs_dim)), jnp.array([0]), jnp.array([0]), jnp.zeros((1, obs_dim)))

        repr_opt = optax.adam(config["repr_lr"])
        repr_opt_state = repr_opt.init(repr_params)

        buffer      = ContrastiveBuffer.create(obs_dim, config["buffer_size"],      config["emp_gamma"], config["batch_size"])
        repr_buffer = ContrastiveBuffer.create(obs_dim, config["repr_buffer_size"], config["emp_gamma"], config["batch_size"])

        rng, emp_rng = jax.random.split(rng)
        emp_state = EmpowermentState(
            repr_params=repr_params, repr_opt_state=repr_opt_state,
            buffer=buffer, repr_buffer=repr_buffer, emp_step=0, rng=emp_rng,
        )

        # ── ENV INIT ────────────────────────────────────────────────────────
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # ====================================================================
        # UPDATE STEP
        # ====================================================================
        def _update_step(runner_state, step_idx):

            def save_callback(step, params):
                step = int(step)
                params_0 = jax.tree_util.tree_map(lambda x: x, params)
                ckpt = {"model": params_0, "config": config}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)

                if step == config["NUM_UPDATES"] // 3:
                    s = "init"
                else:
                    s = "mid"
                save_path = os.path.join(os.getcwd(), "phase_1", args.layout, str(args.seed), s, args.output)
                if not os.path.exists(save_path):
                    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
                    print(f"Saved checkpoint at update {step}")
                else:
                    print(f"Checkpoint already exists at update {step}, resave")
                    shutil.rmtree(save_path)
                    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)

            # ── COLLECT TRAJECTORIES ────────────────────────────────────────
            def _env_step(runner_state, unused):
                train_state_0, train_state_1, env_state, last_obs, rng, emp_state = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # obs_batch[:n] = agent_0 env slot observations
                # obs_batch[n:] = agent_1 env slot observations
                #
                # KEY CHANGE: train_state_0 (ego) operates in agent_1's env slot → feeds on obs_batch[n:]
                #             train_state_1 (pretrained) operates in agent_0's env slot → feeds on obs_batch[:n]

                rng, _rng = jax.random.split(rng)
                rng_0, rng_1 = jax.random.split(_rng)

                # ego agent (train_state_0) observes agent_1's slot
                pi_0, value_0 = network.apply(train_state_0.params, obs_batch[n:])
                # pretrained agent (train_state_1) observes agent_0's slot
                pi_1, value_1 = network.apply(train_state_1.params, obs_batch[:n])

                action_0 = pi_0.sample(seed=rng_0)   # will be sent to env as agent_1's action
                action_1 = pi_1.sample(seed=rng_1)   # will be sent to env as agent_0's action
                log_prob_0 = pi_0.log_prob(action_0)
                log_prob_1 = pi_1.log_prob(action_1)

                # Reconstruct in batchify order: [agent_0_slot..., agent_1_slot...]
                # agent_0 slot is driven by train_state_1 (action_1)
                # agent_1 slot is driven by train_state_0 (action_0)
                action   = jnp.concatenate([action_1,   action_0],   axis=0)
                value    = jnp.concatenate([value_1,    value_0],    axis=0)
                log_prob = jnp.concatenate([log_prob_1, log_prob_0], axis=0)

                env_act = unbatchify(action, env.agents, config["num_envs"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                # Shaped rewards for both env slots
                reward["agent_1"] += info["shaped_reward"]["agent_1"]
                reward["agent_0"] += info["shaped_reward"]["agent_0"]

                next_obs_batch = batchify(obsv, env.agents, config["NUM_ACTORS"])

                if config["type"] == "socalizer":
                    # ego (train_state_0) is in agent_1 env slot:
                    #   robot = action_0 (ego action), human = action_1 (partner action)
                    #   obs = obs_batch[n:] (agent_1 slot), next_obs = next_obs_batch[n:]
                    emp_reward = compute_empowerment_reward(
                        repr_fn.apply, emp_state.repr_params,
                        obs_batch[n:],          # ego current obs (agent_1 env slot)
                        action_0,               # ego action (robot)
                        action_1,               # partner action (human, agent_0 env slot)
                        next_obs_batch[n:],     # ego next obs
                        config["reward_type"], config["repr_dim"],
                    )
                    # Apply empowerment to agent_1 env slot reward (ego's reward)
                    reward["agent_1"] = (1 - config["emp_reward_scale"]) * reward["agent_1"] + config["emp_reward_scale"] * emp_reward

                elif config["type"] == "achiever":
                    penalty = jax.random.lognormal(_rng, sigma=20, shape=reward["agent_1"].shape)

                elif config["type"] == "combo":
                    emp_reward = compute_empowerment_reward(
                        repr_fn.apply, emp_state.repr_params,
                        obs_batch[n:],
                        action_0,
                        action_1,
                        next_obs_batch[n:],
                        config["reward_type"], config["repr_dim"],
                    )
                    reward["agent_1"] = (1 - config["emp_reward_scale"]) * reward["agent_1"] + config["emp_reward_scale"] * emp_reward
                    reward["agent_1"] = jnp.exp(config["risk_reward_scale"] * jnp.clip(reward["agent_1"], -10, 10))
                    reward["agent_0"] += config["risk_reward_scale"] * jnp.exp(jnp.clip(reward["agent_0"], -10, 10))

                else:
                    # "none" type — no modification
                    pass

                # Transition stored in batchify order: [agent_0_slot, agent_1_slot]
                # agent_0_slot → train_state_1 data (value_1, log_prob_1, action_1)
                # agent_1_slot → train_state_0 data (value_0, log_prob_0, action_0)
                transition = Transition(
                    batchify(done,   env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,          # [action_1 (agent_0 slot), action_0 (agent_1 slot)]
                    value,           # [value_1,  value_0]
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,        # [log_prob_1, log_prob_0]
                    obs_batch,       # [obs agent_0 slot, obs agent_1 slot]
                    info,
                )

                runner_state = (train_state_0, train_state_1, env_state, obsv, rng, emp_state)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])
            train_state_0, train_state_1, env_state, last_obs, rng, emp_state = runner_state

            # ── UPDATE EMPOWERMENT REPR ──────────────────────────────────────
            # ego (train_state_0) trajectories are in agent_1 env slot → traj_batch.obs[:, n:]
            # action layout: action[:, :n] = agent_0 slot (partner/ts1), action[:, n:] = agent_1 slot (ego/ts0)
            agent_0_obs     = traj_batch.obs[:, n:]       # (T, n, obs_dim) — ego obs (agent_1 slot)
            agent_0_actions = traj_batch.action[:, n:]    # (T, n)          — ego actions
            agent_1_actions = traj_batch.action[:, :n]    # (T, n)          — partner actions (agent_0 slot)

            for env_idx in range(config["num_envs"]):
                obs_seq = agent_0_obs[:, env_idx, :]
                ar_seq  = agent_0_actions[:, env_idx]
                ah_seq  = agent_1_actions[:, env_idx]
                g = jnp.tile(obs_seq[-1][None], (obs_seq.shape[0], 1))
                emp_rewards = compute_empowerment_reward(
                    repr_fn.apply, emp_state.repr_params,
                    obs_seq, ar_seq, ah_seq, g,
                    config["reward_type"], config["repr_dim"],
                )
                emp_state = emp_state._replace(
                    buffer=emp_state.buffer.extend(obs_seq, ar_seq, ah_seq, emp_rewards),
                    repr_buffer=emp_state.repr_buffer.extend(obs_seq, ar_seq, ah_seq, emp_rewards),
                )

            def update_repr(emp_state):
                emp_rng, sample_rng = jax.random.split(emp_state.rng)
                batch = emp_state.repr_buffer.sample(sample_rng, config["batch_size"])
                grad_fn = jax.grad(lambda p: contrastive_loss(
                    repr_fn.apply, p,
                    batch.state, batch.action_robot, batch.action_human, batch.future_state,
                    a_dim, config["repr_dim"], config["psi_reg"],
                ), has_aux=True)
                grads, _ = grad_fn(emp_state.repr_params)
                updates, new_opt_state = repr_opt.update(grads, emp_state.repr_opt_state)
                return emp_state._replace(
                    repr_params=optax.apply_updates(emp_state.repr_params, updates),
                    repr_opt_state=new_opt_state,
                    emp_step=emp_state.emp_step + 1,
                    rng=emp_rng,
                )

            def skip_repr(emp_state):
                return emp_state._replace(emp_step=emp_state.emp_step + 1)

            emp_state = jax.lax.cond(
                (emp_state.emp_step % config["update_repr_freq"] == 0) &
                (emp_state.buffer.buflen > config["batch_size"]),
                update_repr, skip_repr, emp_state,
            )

            # ── CALCULATE ADVANTAGE ─────────────────────────────────────────
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            # train_state_1 (partner) drives agent_0 slot [:n]
            # train_state_0 (ego)     drives agent_1 slot [n:]
            _, last_val_1 = network.apply(train_state_1.params, last_obs_batch[:n])
            _, last_val_0 = network.apply(train_state_0.params, last_obs_batch[n:])
            # Keep batchify order: [agent_0_slot_val, agent_1_slot_val]
            last_val = jnp.concatenate([last_val_1, last_val_0], axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ── UPDATE NETWORKS ─────────────────────────────────────────────
            def _update_epoch(update_state, unused):

                def _update_minbatch(train_states, batch_info):
                    train_state_0, train_state_1 = train_states
                    traj_batch, advantages, targets = batch_info

                    # traj layout: first half = agent_0 slot (partner/ts1), second half = agent_1 slot (ego/ts0)
                    half = traj_batch.obs.shape[0] // 2

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    # agent_0 slot [:half] → train_state_1 (partner/pretrained)
                    mb_1 = Transition(
                        done=traj_batch.done[:half], action=traj_batch.action[:half],
                        value=traj_batch.value[:half], reward=traj_batch.reward[:half],
                        log_prob=traj_batch.log_prob[:half], obs=traj_batch.obs[:half], info=None)
                    _, grads_1 = grad_fn(train_state_1.params, mb_1, advantages[:half], targets[:half])
                    train_state_1 = train_state_1.apply_gradients(grads=grads_1)

                    # agent_1 slot [half:] → train_state_0 (ego)
                    mb_0 = Transition(
                        done=traj_batch.done[half:], action=traj_batch.action[half:],
                        value=traj_batch.value[half:], reward=traj_batch.reward[half:],
                        log_prob=traj_batch.log_prob[half:], obs=traj_batch.obs[half:], info=None)
                    _, grads_0 = grad_fn(train_state_0.params, mb_0, advantages[half:], targets[half:])
                    train_state_0 = train_state_0.apply_gradients(grads=grads_0)

                    return (train_state_0, train_state_1), None

                train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["num_minibatches"]
                assert batch_size == config["num_steps"] * config["num_envs"]

                perm = jax.random.permutation(_rng, batch_size)

                traj_no_info = Transition(
                    done=traj_batch.done, action=traj_batch.action, value=traj_batch.value,
                    reward=traj_batch.reward, log_prob=traj_batch.log_prob, obs=traj_batch.obs, info=None)

                def make_minibatches(sl):
                    t = jax.tree_util.tree_map(
                        lambda x: x[:, sl] if x is not None else None, traj_no_info)
                    adv = advantages[:, sl]
                    tgt = targets[:, sl]
                    t_flat = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]) if x is not None else None, t)
                    adv_flat = adv.reshape((batch_size,))
                    tgt_flat = tgt.reshape((batch_size,))
                    t_shuf = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, perm, axis=0) if x is not None else None, t_flat)
                    adv_shuf = jnp.take(adv_flat, perm, axis=0)
                    tgt_shuf = jnp.take(tgt_flat, perm, axis=0)
                    t_mini = jax.tree_util.tree_map(
                        lambda x: x.reshape([config["num_minibatches"], -1] + list(x.shape[1:])) if x is not None else None,
                        t_shuf)
                    return t_mini, adv_shuf.reshape([config["num_minibatches"], -1]), tgt_shuf.reshape([config["num_minibatches"], -1])

                # agent_0 slot = [:n] = partner (train_state_1) data
                # agent_1 slot = [n:] = ego     (train_state_0) data
                t1, adv1, tgt1 = make_minibatches(slice(None, n))   # partner data
                t0, adv0, tgt0 = make_minibatches(slice(n, None))   # ego data

                # Concatenate: first half = partner (agent_0 slot), second half = ego (agent_1 slot)
                # _update_minbatch will split by half and apply grads to the correct train_state
                minibatches = (
                    Transition(
                        done     = jnp.concatenate([t1.done,     t0.done],     axis=1),
                        action   = jnp.concatenate([t1.action,   t0.action],   axis=1),
                        value    = jnp.concatenate([t1.value,    t0.value],    axis=1),
                        reward   = jnp.concatenate([t1.reward,   t0.reward],   axis=1),
                        log_prob = jnp.concatenate([t1.log_prob, t0.log_prob], axis=1),
                        obs      = jnp.concatenate([t1.obs,      t0.obs],      axis=1),
                        info     = None,
                    ),
                    jnp.concatenate([adv1, adv0], axis=1),
                    jnp.concatenate([tgt1, tgt0], axis=1),
                )

                (train_state_0, train_state_1), _ = jax.lax.scan(
                    _update_minbatch, (train_state_0, train_state_1), minibatches)

                update_state = (train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state)
                return update_state, None

            update_state = (train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])

            train_state_0 = update_state[0]
            train_state_1 = update_state[1]
            rng            = update_state[5]
            emp_state      = update_state[6]

            # Save checkpoint for proficiency
            jax.lax.cond(
                (step_idx == config["NUM_UPDATES"] // 3) |
                (step_idx == (config["NUM_UPDATES"] * 2) // 3),
                lambda p: jax.debug.callback(save_callback, step_idx, p),
                lambda p: None,
                train_state_0.params,
            )

            metric = traj_batch.info
            runner_state = (train_state_0, train_state_1, env_state, last_obs, rng, emp_state)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_0, train_state_1, env_state, obsv, _rng, emp_state)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, jnp.arange(config["NUM_UPDATES"]))
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=25e6)
    parser.add_argument("--update_epochs", type=int, default=40)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--env_name", type=str, default="overcooked")
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--output", type=str, default="empowerment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_play_seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="coord_ring")
    parser.add_argument("--pretrained_agent_path", type=str, default=None)
    # Empowerment args
    parser.add_argument("--repr_lr", type=float, default=1e-5)
    parser.add_argument("--repr_dim", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=300000)
    parser.add_argument("--repr_buffer_size", type=int, default=520000)
    parser.add_argument("--emp_gamma", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_repr_freq", type=int, default=100)
    parser.add_argument("--reward_type", type=str, default="dot", choices=["dot", "norm", "diff"])
    parser.add_argument("--psi_reg", type=float, default=0.0)
    parser.add_argument("--phi_norm", action="store_true")
    parser.add_argument("--psi_norm", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--type", type=str, default="socalizer")
    parser.add_argument("--emp_reward_scale", type=float, default=0.2)
    parser.add_argument("--risk_reward_scale", type=float, default=0.01)
    parser.add_argument("--mask", type=float, default=0.2)
    parser.add_argument("--achiever_type", type=int, default=0)
    args = parser.parse_args()
    args.save = True

    if args.self_play_seed != 0:
        args.pretrained_agent_path = os.path.join(os.getcwd(), "phase0/" + args.layout + "/" + str(args.self_play_seed) + "/self_play")
    else:
        args.pretrained_agent_path = os.path.join(os.getcwd(), "phase0/" + args.layout + "/self_play")

    print("#####################################")
    if args.type == "socalizer":
        print("type is socalizer with scale:", args.emp_reward_scale,
              "with partner seed:", args.self_play_seed,
              "and training seed:", args.seed)
    print("ego agent operates in agent_1 env slot")
    print("#####################################")

    os.environ["WANDB_API_KEY"] = "495b87eba3dbc88f719508680483181c811852ba"
    wandb.init(project="empowerment", config=vars(args),
               id=wandb.util.generate_id(4),
               group="empowerment_" + args.layout,
               mode="disabled" if args.no_wandb else "online")

    config = {"env_kwargs": {"layout": overcooked_layouts[args.layout]}, "num_seeds": 1}
    config.update(vars(args))

    rng = jax.random.PRNGKey(config["seed"])

    # Load pretrained params for agent_1 (partner, will operate in agent_0 env slot)
    print(f"Loading pretrained agent from {config['pretrained_agent_path']}")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(config["pretrained_agent_path"])
    try:
        raw = ckpt["model"]["params"]
    except (KeyError, TypeError):
        raw = ckpt["params"]
    pretrained_params = jax.tree_util.tree_map(
        lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, raw)
    print("Pretrained params loaded.")

    rngs = jax.random.split(rng, config["num_seeds"])
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, pretrained_params)))
        out = train_jit(rngs)

        returns = out["metrics"]["returned_episode_returns"]
        flat = returns[0].mean(-1).reshape(-1)
        nonzero = flat[flat != 0]
        print(f"Nonzero episodes: {len(nonzero)} / {flat.size}")
        if len(nonzero) > 0:
            print(f"Mean of nonzero:  {float(nonzero.mean()):.2f}")
            print(f"Last 10 nonzero:  {[round(float(x), 1) for x in nonzero[-10:]]}")

        wandb.log({"evaluation/reward": float(nonzero.mean()) if len(nonzero) > 0 else 0.0})

        for data in returns:
            flat = data.mean(-1).reshape(-1)
            plt.plot(flat[flat != 0])
            print("Final mean reward (nonzero):", float(flat[flat != 0].mean()) if (flat != 0).any() else 0.0)

    # Record rollout video
    env_vis = jaxmarl.make(config["env_name"], **config["env_kwargs"])

    def squeeze_params(ts):
        return ts.replace(params=jax.tree_util.tree_map(
            lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, ts.params))

    ts0 = squeeze_params(out["runner_state"][0])
    ts1 = squeeze_params(out["runner_state"][1])
    state_seq = get_rollout(ts0, ts1, config)
    video_frames = render_state_seq(state_seq, env_vis)
    wandb.log({"evaluation/final_video": wandb.Video(video_frames, fps=4, format="mp4")})
    print(f"Video uploaded to wandb ({len(state_seq)} frames)")

    if args.save:
        ckpt = {"model": out["runner_state"][0], "config": config}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        save_path = os.path.join(os.getcwd(), "phase_1", args.layout, str(args.seed), "final_opposite", args.output)
        if not os.path.exists(save_path):
            orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
            print("Saved final checkpoint.")
        else:
            print(f"Checkpoint already exists at update, re-save")
            shutil.rmtree(save_path)
            orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
