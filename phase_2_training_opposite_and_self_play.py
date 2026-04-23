import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Tuple
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from flax.training import orbax_utils
import orbax
import matplotlib.pyplot as plt
import os
import argparse
import wandb
import functools
import pickle
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
# ============================================================================
# NETWORKS
# ============================================================================

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


class ActorCritic(nn.Module):
    """Original MLP actor-critic — used for population partners (frozen)."""
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
    obs_shape: tuple = (7,7,26)
    moe: int = 1
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

        if 1:
            routing_logits = nn.Dense(
                3,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0)
            )(actor_mean)

            routing_weights = nn.softmax(routing_logits)
            chosen_expert = jnp.argmax(routing_weights, axis=-1)

            expert_outputs = []
            for _ in range(self.moe):
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hstate: Tuple[jnp.ndarray, jnp.ndarray]
    info: jnp.ndarray


# ============================================================================
# HELPERS
# ============================================================================
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def load_population(population_dir, population_dir2, load_dir, num_agents, layout):
    """Load a list of pretrained partner params from orbax checkpoints."""
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    population_params = []

    self_play_path_0 = os.path.join(os.getcwd(), "phase0/"+layout+ "/self_play")
    self_play_path_1 = os.path.join(os.getcwd(), "phase0/"+layout+ "/1" + "/self_play")
    self_play_path_20 = os.path.join(os.getcwd(), "phase0/"+layout+ "/20" + "/self_play")
    self_play_path = [self_play_path_0, self_play_path_1, self_play_path_20]

    for i in range(3):
        agent_path = self_play_path[i]
        ckpt = orbax_checkpointer.restore(agent_path)
        try:
            params = ckpt["model"]["params"]
        except (KeyError, TypeError):
            params = ckpt["params"]
        params = jax.tree_util.tree_map(
            lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, params
        )
        population_params.append(params)
    
    
    for i in range(num_agents):

        # Either load achiever LR or achiever-reward
        if(load_dir =="phase1_r_achiever") and (layout == "forced_coord" or layout=="asymm_advantages") and (i in [11,17,2,3,7,8]):
            # Reward based achiever
            pop_dir = "/home/tom.danino/zero_shot_jax/phase1_r_achiever/" + layout
        elif(load_dir =="phase1_r_achiever") and (i in [11,7,8]): # all other layouts
            # Reward_based achiever
            pop_dir = "/home/tom.danino/zero_shot_jax/phase1_r_achiever/" + layout
        else:
             # LR based achiever
             pop_dir = "/home/tom.danino/zero_shot_jax/phase1_lr/" + layout

        for curr_stage in ["init", "mid", "final"]:
            agent_path = os.path.join(pop_dir, str(i), curr_stage, "empowerment")
            ckpt = orbax_checkpointer.restore(agent_path)
            try:
                params = ckpt["model"]["params"]
            except (KeyError, TypeError):
                params = ckpt["params"]
            params = jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, params
            )
            params = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (1,)) if x.shape == () else x, params
            )
            if "params" not in params:
                params = {"params": params}
            population_params.append(params)
            print(f"Loaded partner {i}/{curr_stage} from {agent_path}")

    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *population_params)
    return stacked


# ============================================================================
# TRAINING
# ============================================================================
def make_train(config, population_params):
    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"]+"_9"
    config_env["SEED"] = config["seed"]
    env = initialize_environment(config_env)
    pop_size = config["population_size"]*3+3
    config["NUM_ACTORS"] = env.num_agents * config["num_envs"]
    config["NUM_UPDATES"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["MINIBATCH_SIZE"] = (
        config["num_envs"] // config["num_minibatches"]
    )
    updates_per_partner = config["updates_per_partner"]
    num_partner_rounds  = config["NUM_UPDATES"] // (updates_per_partner * pop_size)

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["NUM_UPDATES"]
        return config["lr"] * frac

    def train(rng):
        ego_network = ActorCriticRNN(
            action_dim=env.action_space().n,
            hidden_size=config.get("hidden_size", 128),
            activation=config["activation"],
            obs_shape=env.observation_space().shape,
            moe = config["MOE"]
        )

        partner_network_mlp = ActorCritic(env.action_space().n, activation=config["activation"])

        rng, _rng = jax.random.split(rng)
        n       = config["num_envs"]
        obs_dim = int(np.prod(env.observation_space().shape))

        init_obs   = jnp.zeros((1, 2, obs_dim))
        init_carry = ActorCriticRNN.initialize_carry(config.get("hidden_size", 128), 2)
        init_done  = jnp.zeros((1, 2))

        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            )

        ego_state = TrainState.create(
            apply_fn=ego_network.apply,
            params=ego_network.init(_rng, init_carry, init_obs, init_done),
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, n)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = ActorCriticRNN.initialize_carry(config.get("hidden_size", 128), n)

        # ------------------------------------------------------------------ #
        # GAE  (shared — works for any env-axis size)
        # ------------------------------------------------------------------ #
        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                delta = (
                    transition.reward
                    + config["gamma"] * next_value * (1 - transition.done)
                    - transition.value
                )
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # ------------------------------------------------------------------ #
        # PPO UPDATE  (population partners — env batch size = n)
        # ------------------------------------------------------------------ #
        def _ppo_update(ego_state, traj_batch, advantages, targets, rng):
            """PPO epoch/minibatch update. traj_batch env-axis size = n."""

            def _update_epoch(update_state, unused):
                def _update_minibatch(ego_state, batch_info):
                    traj_mb, advantages_mb, targets_mb = batch_info

                    def _loss_fn(params, traj_mb, gae, targets):
                        init_hstate = jax.tree_util.tree_map(
                            lambda x: x[0], traj_mb.hstate
                        )
                        _, (pi, value) = ego_network.apply(
                            params,
                            init_hstate,
                            traj_mb.obs,
                            traj_mb.done,
                        )
                        log_prob = pi.log_prob(traj_mb.action)
                        value_pred_clipped = traj_mb.value + (value - traj_mb.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()
                        ratio = jnp.exp(log_prob - traj_mb.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae_norm,
                        ).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(ego_state.params, traj_mb, advantages_mb, targets_mb)
                    ego_state = ego_state.apply_gradients(grads=grads)
                    return ego_state, total_loss

                ego_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, n)

                traj_no_info = Transition(
                    done=traj_batch.done,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob,
                    obs=traj_batch.obs,
                    hstate=traj_batch.hstate,
                    info=None,
                )
                shuffled_traj = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1) if x is not None else None,
                    traj_no_info,
                )
                shuffled_advantages = jnp.take(advantages, permutation, axis=1)
                shuffled_targets    = jnp.take(targets,    permutation, axis=1)

                def split_envs(x):
                    s = x.shape
                    x = x.reshape((s[0], config["num_minibatches"], config["MINIBATCH_SIZE"]) + s[2:])
                    return x.transpose((1, 0, 2) + tuple(range(3, x.ndim)))

                minibatch_traj       = jax.tree_util.tree_map(lambda x: split_envs(x) if x is not None else None, shuffled_traj)
                minibatch_advantages = split_envs(shuffled_advantages)
                minibatch_targets    = split_envs(shuffled_targets)

                ego_state, total_loss = jax.lax.scan(
                    _update_minibatch, ego_state, (minibatch_traj, minibatch_advantages, minibatch_targets)
                )
                update_state = (ego_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (ego_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["update_epochs"]
            )
            return update_state[0], update_state[-1], loss_info  # ego_state, rng, loss_info

        # ------------------------------------------------------------------ #
        # PPO UPDATE  (self-play — env batch size = 2*n)
        #
        # Both agent-slot trajectories are concatenated along the env axis,
        # giving 2*n samples per time-step. The number of minibatches stays
        # the same; each minibatch is twice as wide (MINIBATCH_SIZE * 2).
        # ------------------------------------------------------------------ #
        def _ppo_update_selfplay(ego_state, traj_batch, advantages, targets, rng):
            """PPO update for self-play. traj_batch env-axis size = 2*n."""
            n2                = n * 2
            minibatch_size_sp = config["MINIBATCH_SIZE"] * 2   # = n2 // num_minibatches

            def _update_epoch(update_state, unused):
                def _update_minibatch(ego_state, batch_info):
                    traj_mb, advantages_mb, targets_mb = batch_info

                    def _loss_fn(params, traj_mb, gae, targets):
                        init_hstate = jax.tree_util.tree_map(
                            lambda x: x[0], traj_mb.hstate
                        )
                        _, (pi, value) = ego_network.apply(
                            params,
                            init_hstate,
                            traj_mb.obs,
                            traj_mb.done,
                        )
                        log_prob = pi.log_prob(traj_mb.action)
                        value_pred_clipped = traj_mb.value + (value - traj_mb.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()
                        ratio = jnp.exp(log_prob - traj_mb.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae_norm,
                        ).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(ego_state.params, traj_mb, advantages_mb, targets_mb)
                    ego_state = ego_state.apply_gradients(grads=grads)
                    return ego_state, total_loss

                ego_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Shuffle across the full 2*n env axis
                permutation = jax.random.permutation(_rng, n2)

                traj_no_info = Transition(
                    done=traj_batch.done,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob,
                    obs=traj_batch.obs,
                    hstate=traj_batch.hstate,
                    info=None,
                )
                shuffled_traj = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1) if x is not None else None,
                    traj_no_info,
                )
                shuffled_advantages = jnp.take(advantages, permutation, axis=1)
                shuffled_targets    = jnp.take(targets,    permutation, axis=1)

                def split_envs(x):
                    s = x.shape
                    x = x.reshape((s[0], config["num_minibatches"], minibatch_size_sp) + s[2:])
                    return x.transpose((1, 0, 2) + tuple(range(3, x.ndim)))

                minibatch_traj       = jax.tree_util.tree_map(lambda x: split_envs(x) if x is not None else None, shuffled_traj)
                minibatch_advantages = split_envs(shuffled_advantages)
                minibatch_targets    = split_envs(shuffled_targets)

                ego_state, total_loss = jax.lax.scan(
                    _update_minibatch, ego_state, (minibatch_traj, minibatch_advantages, minibatch_targets)
                )
                update_state = (ego_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (ego_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["update_epochs"]
            )
            return update_state[0], update_state[-1], loss_info  # ego_state, rng, loss_info

        # ------------------------------------------------------------------ #
        # UPDATE STEP: SELF-PLAY  (ego vs ego, learn from BOTH agent slots)
        #
        # A single environment rollout is run with ego filling BOTH agent_0
        # and agent_1 simultaneously.  Two separate LSTM hidden states are
        # maintained: hstate_a0 for the agent_0 perspective and hstate_a1 for
        # agent_1.  Each env-step produces two Transition objects — one per
        # slot — sharing the same done signal but with perspective-correct
        # obs / action / value / log_prob / hstate.
        #
        # After num_steps:
        #   - GAE is computed independently for each slot (bootstrapped with
        #     its own hstate and obs).
        #   - The two trajectory batches are concatenated along the env axis
        #     (axis=1), yielding shape (num_steps, 2*n, ...).
        #   - A single _ppo_update_selfplay call updates ego on all 2*n
        #     samples, with minibatches of size MINIBATCH_SIZE*2.
        #
        # runner_state = (ego_state, env_state, last_obs, last_done,
        #                 hstate_a0, hstate_a1, rng)
        # ------------------------------------------------------------------ #
        def _update_step_selfplay(runner_state, unused):
            ego_state, env_state, last_obs, last_done, hstate_a0, hstate_a1, rng = runner_state

            def _env_step(runner_state, unused):
                ego_state, env_state, last_obs, last_done, hstate_a0, hstate_a1, rng = runner_state

                # obs_batch: (2*n, obs_dim) — rows 0..n-1 = agent_0, n..2n-1 = agent_1
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                obs_a0 = obs_batch[:n]   # agent_0 perspective for ego
                obs_a1 = obs_batch[n:]   # agent_1 perspective for ego

                rng, rng_a0, rng_a1 = jax.random.split(rng, 3)

                # Ego forward pass as agent_0
                new_hstate_a0, (pi_a0, value_a0) = ego_network.apply(
                    ego_state.params,
                    hstate_a0,
                    obs_a0[None],       # add time dim: (1, n, obs_dim)
                    last_done[None],    # (1, n)
                )
                pi_a0    = jax.tree_util.tree_map(lambda x: x.squeeze(0), pi_a0)
                value_a0 = value_a0.squeeze(0)

                # Ego forward pass as agent_1
                new_hstate_a1, (pi_a1, value_a1) = ego_network.apply(
                    ego_state.params,
                    hstate_a1,
                    obs_a1[None],
                    last_done[None],
                )
                pi_a1    = jax.tree_util.tree_map(lambda x: x.squeeze(0), pi_a1)
                value_a1 = value_a1.squeeze(0)

                action_a0   = pi_a0.sample(seed=rng_a0)   # (n,)
                action_a1   = pi_a1.sample(seed=rng_a1)   # (n,)
                log_prob_a0 = pi_a0.log_prob(action_a0)
                log_prob_a1 = pi_a1.log_prob(action_a1)

                # Ego fills BOTH slots: agent_0 acts as a0, agent_1 acts as a1
                action_full = jnp.concatenate([action_a0, action_a1], axis=0)  # (2*n,)
                env_act = unbatchify(action_full, env.agents, n, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, n)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                done_all = done["__all__"].astype(jnp.float32)  # (n,) — same for both slots

                # Add shaped rewards per agent slot
                reward_a0 = reward["agent_0"] + info["shaped_reward"]["agent_0"]  # (n,)
                reward_a1 = reward["agent_1"] + info["shaped_reward"]["agent_1"]  # (n,)

                # One Transition per agent slot; env axis = n for each
                transition_a0 = Transition(
                    done=done_all,
                    action=action_a0,
                    value=value_a0,
                    reward=reward_a0,
                    log_prob=log_prob_a0,
                    obs=obs_a0,
                    hstate=hstate_a0,   # hstate at the *start* of this step
                    info=info,
                )
                transition_a1 = Transition(
                    done=done_all,
                    action=action_a1,
                    value=value_a1,
                    reward=reward_a1,
                    log_prob=log_prob_a1,
                    obs=obs_a1,
                    hstate=hstate_a1,
                    info=info,
                )

                runner_state = (ego_state, env_state, obsv, done_all,
                                new_hstate_a0, new_hstate_a1, rng)
                # scan stacks both transitions along the time axis
                return runner_state, (transition_a0, transition_a1)

            runner_state, (traj_a0, traj_a1) = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )
            # traj_aX fields: (num_steps, n, ...)
            ego_state, env_state, last_obs, last_done, hstate_a0, hstate_a1, rng = runner_state

            # Bootstrap values independently for each slot
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

            _, (_, last_val_a0) = ego_network.apply(
                ego_state.params, hstate_a0, last_obs_batch[:n][None], last_done[None],
            )
            last_val_a0 = last_val_a0.squeeze(0)  # (n,)

            _, (_, last_val_a1) = ego_network.apply(
                ego_state.params, hstate_a1, last_obs_batch[n:][None], last_done[None],
            )
            last_val_a1 = last_val_a1.squeeze(0)  # (n,)

            # GAE per slot, then concatenate along env axis (axis=1)
            adv_a0, tgt_a0 = _calculate_gae(traj_a0, last_val_a0)
            adv_a1, tgt_a1 = _calculate_gae(traj_a1, last_val_a1)

            # Combined: (num_steps, 2*n, ...)
            traj_combined = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=1),
                traj_a0, traj_a1,
            )
            adv_combined = jnp.concatenate([adv_a0, adv_a1], axis=1)
            tgt_combined = jnp.concatenate([tgt_a0, tgt_a1], axis=1)

            ego_state, rng, loss_info = _ppo_update_selfplay(
                ego_state, traj_combined, adv_combined, tgt_combined, rng
            )

            # Return agent_1 info as metric (consistent shape with rest of training)
            metric       = traj_a1.info
            runner_state = (ego_state, env_state, last_obs, last_done,
                            hstate_a0, hstate_a1, rng)
            return runner_state, metric

        # ------------------------------------------------------------------ #
        # UPDATE STEP: MLP PARTNER (used for population partners)
        # runner_state = (ego_state, env_state, last_obs, last_done,
        #                 hstate, rng, partner_params)   <- no partner_h_state
        #
        # partner_idx is a traced JAX value inside jax.lax.scan, so we
        # cannot use it in Python if/else or to compute Python integer indices.
        # Use jnp.where / jax.lax.dynamic_slice_in_dim instead.
        # ------------------------------------------------------------------ #
        def _update_step_mlp(runner_state, unused, partner_idx):
            ego_state, env_state, last_obs, last_done, hstate, rng, partner_params = runner_state

            # ego_is_agent1 is a traced bool — must NOT be used in Python if/else
            ego_is_agent1 = (partner_idx >= 1) & (partner_idx <= 10)
            # Use jnp.where for integer index computation instead of Python ternary
            ego_slot     = jnp.where(ego_is_agent1, n, 0)
            partner_slot = jnp.where(ego_is_agent1, 0, n)

            def _env_step(runner_state, unused):
                ego_state, env_state, last_obs, last_done, hstate, rng, partner_params = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                rng, rng_ego, rng_partner = jax.random.split(rng, 3)

                # Ego forward pass — use dynamic slice with traced ego_slot
                ego_obs = jax.lax.dynamic_slice_in_dim(obs_batch, ego_slot, n, axis=0)
                new_hstate, (pi_ego, value_ego) = ego_network.apply(
                    ego_state.params,
                    hstate,
                    ego_obs[None],
                    last_done[None],
                )
                pi_ego    = jax.tree_util.tree_map(lambda x: x.squeeze(0), pi_ego)
                value_ego = value_ego.squeeze(0)

                # Partner forward pass — always MLP (stateless), dynamic slice
                partner_obs = jax.lax.dynamic_slice_in_dim(obs_batch, partner_slot, n, axis=0)
                pi_partner, _ = partner_network_mlp.apply(partner_params, partner_obs)

                action_ego     = pi_ego.sample(seed=rng_ego)
                action_partner = pi_partner.sample(seed=rng_partner)
                log_prob_ego   = pi_ego.log_prob(action_ego)

                # Reconstruct full action array in batchify order [agent_0, agent_1].
                # When ego_is_agent1: action_full = [partner, ego]
                # When ego_is_agent0: action_full = [ego, partner]
                # Use jnp.where on the stacked result to avoid traced Python if/else.
                action_ego_first     = jnp.concatenate([action_ego,     action_partner], axis=0)
                action_partner_first = jnp.concatenate([action_partner, action_ego],     axis=0)
                action_full = jnp.where(ego_is_agent1, action_partner_first, action_ego_first)

                env_act = unbatchify(action_full, env.agents, n, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, n)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                # Apply shaped reward to the correct agent slot using jnp.where
                shaped_agent0 = info["shaped_reward"]["agent_0"]
                shaped_agent1 = info["shaped_reward"]["agent_1"]
                reward["agent_0"] = reward["agent_0"] + jnp.where(ego_is_agent1, 0.0, shaped_agent0)
                reward["agent_1"] = reward["agent_1"] + jnp.where(ego_is_agent1, shaped_agent1, 0.0)

                done_ego = done["__all__"].astype(jnp.float32)

                # Extract ego reward from correct slot using dynamic_slice_in_dim
                reward_batch = batchify(reward, env.agents, config["NUM_ACTORS"])
                reward_ego = jax.lax.dynamic_slice_in_dim(reward_batch, ego_slot, n, axis=0).squeeze()

                transition = Transition(
                    done=done_ego,
                    action=action_ego,
                    value=value_ego,
                    reward=reward_ego,
                    log_prob=log_prob_ego,
                    obs=ego_obs,
                    hstate=hstate,
                    info=info,
                )

                runner_state = (ego_state, env_state, obsv, done_ego, new_hstate, rng, partner_params)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )
            ego_state, env_state, last_obs, last_done, hstate, rng, partner_params = runner_state

            # Bootstrap value — use dynamic slice with traced ego_slot
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_ego_obs = jax.lax.dynamic_slice_in_dim(last_obs_batch, ego_slot, n, axis=0)
            _, (_, last_val) = ego_network.apply(
                ego_state.params, hstate, last_ego_obs[None], last_done[None],
            )
            last_val = last_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val)
            ego_state, rng, loss_info = _ppo_update(ego_state, traj_batch, advantages, targets, rng)

            metric       = traj_batch.info
            runner_state = (ego_state, env_state, last_obs, last_done, hstate, rng, partner_params)
            return runner_state, metric

        # ------------------------------------------------------------------ #
        # MIDDLE LOOP — population partners (MLP)
        # ------------------------------------------------------------------ #
        def _partner_phase(runner_state, partner_idx):
            ego_state, env_state, last_obs, last_done, hstate, rng = runner_state

            partner_params = jax.tree_util.tree_map(
                lambda x: x[partner_idx], population_params
            )

            hstate    = ActorCriticRNN.initialize_carry(config.get("hidden_size", 128), n)
            last_done = jnp.zeros_like(last_done)

            # No partner_h_state needed for MLP
            inner_runner = (ego_state, env_state, last_obs, last_done, hstate, rng, partner_params)
            inner_runner, metrics = jax.lax.scan(
                lambda s, u: _update_step_mlp(s, u, partner_idx),
                inner_runner, None, updates_per_partner
            )
            ego_state, env_state, last_obs, last_done, hstate, rng, _ = inner_runner

            ep_returns = metrics["returned_episode_returns"]
            flat       = ep_returns.reshape(-1)
            n_nonzero  = jnp.sum(flat != 0)
            mean_rew   = jnp.where(n_nonzero > 0,
                            jnp.sum(jnp.where(flat != 0, flat, 0.0)) / jnp.maximum(n_nonzero, 1),
                            0.0)
            jax.debug.print(
                "  Finished partner {partner_idx} | mean reward: {mean_rew} | completed episodes: {n}",
                partner_idx=partner_idx,
                mean_rew=mean_rew,
                n=n_nonzero,
            )

            return (ego_state, env_state, last_obs, last_done, hstate, rng), metrics

        # ------------------------------------------------------------------ #
        # OUTER LOOP
        # ------------------------------------------------------------------ #
        def _population_round(runner_state, round_idx):
            ego_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # --- Self-play phase: ego vs ego, learning from BOTH agent slots ---
            # Fresh hidden states for both perspectives at the start of each round.
            hstate_a0      = ActorCriticRNN.initialize_carry(config.get("hidden_size", 128), n)
            hstate_a1      = ActorCriticRNN.initialize_carry(config.get("hidden_size", 128), n)
            self_play_done = jnp.zeros_like(last_done)

            self_play_runner = (ego_state, env_state, last_obs, self_play_done,
                                hstate_a0, hstate_a1, rng)

            self_play_runner, self_play_metrics = jax.lax.scan(
                _update_step_selfplay, self_play_runner, None, updates_per_partner
            )
            ego_state, env_state, last_obs, last_done, hstate_a0, hstate_a1, rng = self_play_runner
            # Carry forward hstate_a1 as the main hstate for the population phase
            hstate = hstate_a1

            ep_returns = self_play_metrics["returned_episode_returns"]
            flat       = ep_returns.reshape(-1)
            n_nonzero  = jnp.sum(flat != 0)
            mean_rew   = jnp.where(n_nonzero > 0,
                            jnp.sum(jnp.where(flat != 0, flat, 0.0)) / jnp.maximum(n_nonzero, 1),
                            0.0)
            jax.debug.print(
                "  Finished self-play | mean reward: {mean_rew} | completed episodes: {n}",
                mean_rew=mean_rew,
                n=n_nonzero,
            )

            # --- Population phase: ego vs MLP partners ---
            jax.debug.print(
                "Round {round_idx}/{total_rounds}",
                round_idx=round_idx + 1,
                total_rounds=num_partner_rounds,
            )
            partner_indices = jnp.arange(pop_size)
            pop_runner = (ego_state, env_state, last_obs, last_done, hstate, rng)
            pop_runner, metrics = jax.lax.scan(_partner_phase, pop_runner, partner_indices)

            return pop_runner, metrics

        jax.debug.print(
            "Starting Phase 2: {total_rounds} rounds x {pop_size} partners x {ups} updates each",
            total_rounds=num_partner_rounds,
            pop_size=pop_size,
            ups=updates_per_partner,
        )

        rng, _rng = jax.random.split(rng)
        init_done    = jnp.zeros((n,))
        runner_state = (ego_state, env_state, obsv, init_done, init_hstate, _rng)
        runner_state, metrics = jax.lax.scan(
            _population_round, runner_state, jnp.arange(num_partner_rounds)
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=145e6)
    parser.add_argument("--update_epochs", type=int, default=4)
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
    parser.add_argument("--hidden_size", type=int, default=120,
                        help="LSTM hidden size for the RNN ego agent")
    parser.add_argument("--output", type=str, default="phase2_ego")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--population_dir", type=str,
                        default="/home/tom.danino/zero_shot_jax/phase1_lr/")
    parser.add_argument("--population_size", type=int, default=22)
    parser.add_argument("--updates_per_partner", type=int, default=15)
    parser.add_argument("--save", action="store_true", default=True)
    
    parser.add_argument("--load_dir", type=str, default="phase1_r_achiever")
    parser.add_argument("--MOE", type=int, default=1)

    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    args.population_dir = "/home/tom.danino/zero_shot_jax/phase1_lr/" + args.layout
    population_dir2 = "/home/tom.danino/zero_shot_jax/phase1_r_achiever/" + args.layout
    os.environ["WANDB_API_KEY"] = "495b87eba3dbc88f719508680483181c811852ba"
    wandb.init(
        project="empowerment",
        config=vars(args),
        group="phase2_" + args.layout,
        mode="disabled" if args.no_wandb else "online",
    )

    config = {
        "env_kwargs": {"layout": overcooked_layouts[args.layout]},
        "num_seeds": 1,
    }
    config.update(vars(args))

    population_params = load_population(
        config["population_dir"],
        population_dir2,
        config["load_dir"],
        config["population_size"],
        args.layout
    )
    print(f"Loaded population of {config['population_size']} agents.")

    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = args.layout+"_9"
    _test_env = initialize_environment(config_env)

    print("####################################################")
    print(_test_env.observation_space().shape)
    print("#####################################################")

    _test_key = jax.random.PRNGKey(42)
    _test_obs, _test_state = _test_env.reset(_test_key)
    print("Env agents:", _test_env.agents)
    print("Obs shapes:", {k: v.shape for k, v in _test_obs.items()})
    _test_actions = {a: jnp.zeros((), dtype=jnp.int32) for a in _test_env.agents}
    _test_obs2, _test_state2, _test_reward, _test_done, _test_info = _test_env.step(
        _test_key, _test_state, _test_actions
    )
    print("Sanity check reward:", _test_reward)
    print("Sanity check done:", _test_done)
    print("Sanity check info keys:", list(_test_info.keys()))

    _state = _test_state
    for _i in range(400):
        _key, _test_key = jax.random.split(_test_key)
        _acts = {a: jax.random.randint(_key, (), 0, _test_env.action_space(a).n) for a in _test_env.agents}
        _, _state, _rew, _done, _info = _test_env.step(_key, _state, _acts)
        if any(v for v in _done.values()):
            print(f"Episode done at step {_i}: done={_done}, reward={_rew}")
            break
    else:
        print("WARNING: Episode never done after 400 steps — check env config!")

    _state = _test_state
    for _i in range(410):
        _key, _test_key = jax.random.split(_test_key)
        _acts = {a: jax.random.randint(_key, (), 0, _test_env.action_space(a).n) for a in _test_env.agents}
        _, _state, _rew, _done, _info = _test_env.step(_key, _state, _acts)
        if _i >= 395:
            print(f"Step {_i}: done={_done}, reward={_rew}, shaped={_info['shaped_reward']}")

    from jaxmarl.wrappers.baselines import LogWrapper as _LogWrapper
    _log_env = _LogWrapper(_test_env)
    _key = jax.random.PRNGKey(42)
    _obs, _state = _log_env.reset(_key)
    for _i in range(410):
        _key, _test_key = jax.random.split(_key)
        _acts = {a: jax.random.randint(_test_key, (), 0, _log_env.action_space(a).n) for a in _log_env.agents}
        _obs, _state, _rew, _done, _info = _log_env.step(_key, _state, _acts)
        if _i >= 395 or any(bool(v) for k,v in _done.items() if k != "__all__" or bool(_done["__all__"])):
            print(f"LogWrapper step {_i}: done={_done}, info_keys={list(_info.keys())}")
            if "returned_episode_returns" in _info:
                print(f"  returned_episode_returns={_info['returned_episode_returns']}")
            if bool(_done.get("__all__", False)):
                break

    rng  = jax.random.PRNGKey(config["seed"])
    rngs = jax.random.split(rng, config["num_seeds"])

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, population_params)))
        out = train_jit(rngs)

        returns = out["metrics"]["returned_episode_returns"]
        all_nonzero = []
        for data in returns:
            flat    = data.reshape(-1)
            nonzero = flat[flat != 0]
            if len(nonzero) > 0:
                print(f"Final mean reward (nonzero): {float(nonzero[-1000:].mean()):.2f}")
                all_nonzero.append(nonzero)
                plt.plot(nonzero)

        if all_nonzero:
            combined     = np.concatenate(all_nonzero)
            final_reward = float(combined[-1000:].mean())
        else:
            final_reward = 0.0

        wandb.log({"evaluation/reward": final_reward})
    if(args.load_dir == phase1_r_achiever):
        save_type = "standrard"
    else:
        save_type = "LR"
    if args.save:
        ego_train_state = out["runner_state"][0]
        ckpt = {"params": jax.tree_map(lambda x: np.array(x), ego_train_state.params),
                "config": config}

        save_path = os.path.join(os.getcwd(), "phase2", args.layout, str(args.seed), save_type,"MOE_" + str(args.MOE))
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{args.output}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(ckpt, f)

        print(f"Saved ego agent as pickle to {file_path}")

