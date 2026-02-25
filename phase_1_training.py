import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
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
import collections

# ============================================================================
# BUFFER
# ============================================================================
Batch = collections.namedtuple(
    "Batch",
    ["state", "action_robot", "action_human", "next_state", "future_state", "done", "idx", "future_idx", "reward"],
)

class ContrastiveBuffer(NamedTuple):
    """JAX-compatible buffer for contrastive learning"""
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
        """Create a new buffer"""
        return ContrastiveBuffer(
            state_buffer=jnp.zeros((size, obs_dim)),
            horizon_buffer=jnp.zeros((size,), dtype=jnp.int32),
            action_human_buffer=jnp.zeros((size,), dtype=jnp.int32),
            action_robot_buffer=jnp.zeros((size,), dtype=jnp.int32),
            reward_buffer=jnp.zeros((size,)),
            next_pos=0,
            max_pos=0,
            buflen=0,
            obs_dim=obs_dim,
            size=size,
            gamma=gamma,
            batch_size=batch_size,
        )

    def extend(self, obs, action_robot, action_human, reward):
        """Add a trajectory to the buffer"""
        # Convert observations to float32 to match buffer dtype
        obs = jnp.asarray(obs, dtype=jnp.float32)
        
        t_left = jnp.arange(len(obs))[::-1]
        next_pos = jax.lax.select(self.next_pos + len(obs) > self.size, 0, self.next_pos)
        
        state_buffer = jax.lax.dynamic_update_slice_in_dim(self.state_buffer, obs, next_pos, axis=0)
        horizon_buffer = jax.lax.dynamic_update_slice_in_dim(self.horizon_buffer, t_left, next_pos, axis=0)
        action_robot_buffer = jax.lax.dynamic_update_slice_in_dim(
            self.action_robot_buffer, action_robot, next_pos, axis=0
        )
        action_human_buffer = jax.lax.dynamic_update_slice_in_dim(
            self.action_human_buffer, action_human, next_pos, axis=0
        )
        reward_buffer = jax.lax.dynamic_update_slice_in_dim(self.reward_buffer, reward, next_pos, axis=0)
        
        next_pos = next_pos + len(obs)
        buflen = jnp.minimum(self.buflen + len(obs), self.size)
        next_pos = jax.lax.select(next_pos >= self.size, 0, next_pos)
        max_pos = jnp.maximum(self.max_pos, next_pos)
        
        return self._replace(
            state_buffer=state_buffer,
            horizon_buffer=horizon_buffer,
            action_robot_buffer=action_robot_buffer,
            action_human_buffer=action_human_buffer,
            reward_buffer=reward_buffer,
            next_pos=next_pos,
            buflen=buflen,
            max_pos=max_pos,
        )

    def _sample(self, key):
        """Sample a single transition"""
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
            state=state,
            action_robot=action_robot,
            action_human=action_human,
            next_state=next_state,
            future_state=future_state,
            done=done,
            idx=idx,
            future_idx=future_idx,
            reward=reward,
        )

    def sample(self, key, batch_size):
        """Sample a batch of transitions with explicit batch size"""
        keys = jax.random.split(key, batch_size)
        samples = jax.vmap(self._sample)(keys)
        return Batch(**samples)


# ============================================================================
# NETWORKS
# ============================================================================
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
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


def make_repr_fn(a_dim, repr_dim, phi_norm, psi_norm):
    """Create representation network function"""
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


# ============================================================================
# EMPOWERMENT STATE
# ============================================================================
class EmpowermentState(NamedTuple):
    """State for empowerment learning components"""
    repr_params: dict
    repr_opt_state: optax.OptState
    buffer: 'ContrastiveBuffer'
    repr_buffer: 'ContrastiveBuffer'
    emp_step: int
    rng: jax.random.PRNGKey


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
####################################################
# NEW CODE: Updated get_rollout to handle train_state_1
####################################################
def get_rollout(train_state_0, config, train_state_1=None):
    """Generate a single rollout for visualization"""
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])

    network = ActorCritic(env.action_space().n, activation=config["activation"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params_0 = train_state_0.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs = {k: v.flatten() for k, v in obs.items()}

        # Agent 0 uses trained policy
        pi_0, _ = network.apply(network_params_0, obs["agent_0"])
        
        # Agent 1 uses train_state_1 if available, otherwise same as agent 0
        if train_state_1 is not None:
            pi_1, _ = network.apply(train_state_1.params, obs["agent_1"])
        else:
            pi_1, _ = network.apply(network_params_0, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq
####################################################
# END NEW CODE
####################################################


def render_state_seq(state_seq, env):
    """Render a sequence of states into video frames"""
    padding = env.agent_view_size - 2

    def get_frame(state):
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        frame = OvercookedVisualizer._render_grid(
            grid, tile_size=32, highlight_mask=None, 
            agent_dir_idx=state.agent_dir_idx, agent_inv=state.agent_inv
        )
        return frame

    frame_seq = np.stack([get_frame(state) for state in state_seq])
    frame_seq = np.moveaxis(frame_seq, -1, 1)

    return frame_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ============================================================================
# EMPOWERMENT FUNCTIONS
# ============================================================================
def compute_empowerment_reward(repr_fn, repr_params, s, ar, ah, g, reward_type, repr_dim):
    """Compute empowerment reward using representation network"""
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
    """Compute contrastive loss for representation learning"""
    sa_phi, s_phi, g_psi, tau = repr_fn(repr_params, s, ar, ah, g)

    def infonce(phi, psi):
        logits = jnp.sum(phi[:, None, :] * psi[None, :, :], axis=-1)
        logits = logits / repr_dim / tau
        logits1 = jax.nn.log_softmax(logits, axis=1)
        logits2 = jax.nn.log_softmax(logits, axis=0)
        loss1 = -jnp.mean(jnp.diag(logits1))
        loss2 = -jnp.mean(jnp.diag(logits2))
        loss = loss1 + loss2
        acc1 = jnp.mean(jnp.argmax(logits1, axis=1) == jnp.arange(logits1.shape[0]))
        acc2 = jnp.mean(jnp.argmax(logits2, axis=0) == jnp.arange(logits2.shape[1]))
        
        l2_psi = jnp.mean(psi**2)
        l2_phi = jnp.mean(phi**2)

        loss += psi_reg * l2_psi**2

        info = {
            "phi_std": jnp.std(phi, axis=0).mean(),
            "psi_std": jnp.std(psi, axis=0).mean(),
            "l2_phi": l2_phi,
            "l2_psi": l2_psi,
            "diag": jnp.diag(logits),
            "loss1": loss1,
            "loss2": loss2,
            "acc1": acc1,
            "acc2": acc2,
            "loss": loss,
            "temp": tau,
        }

        return loss, info

    batch = s.shape[0]
    g_tiled = jnp.concatenate([g_psi, g_psi], axis=0)
    s_tiled = jnp.concatenate([s_phi, sa_phi], axis=0)

    loss, info = infonce(s_tiled, g_tiled)
    
    info["mutual_info"] = jnp.mean(info["diag"][batch:] - info["diag"][:batch])
    info["action_human"] = ah
    info["action_robot"] = ar

    return loss, info


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def make_train(config, pretrained_params=None):
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])

    config["NUM_ACTORS"] = env.num_agents * config["num_envs"]
    config["NUM_UPDATES"] = config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["num_steps"] // config["num_minibatches"]
    
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["NUM_UPDATES"]
        return config["lr"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["activation"])
        rng, _rng = jax.random.split(rng)

        init_x = jnp.zeros(env.observation_space().shape)
        init_x = init_x.flatten()

        network_params = network.init(_rng, init_x)
        
        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(config["lr"], eps=1e-5))
        
        ####################################################
        # NEW CODE: Create train_state_0 (agent_0 - random init)
        ####################################################
        train_state_0 = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        ####################################################
        # END NEW CODE
        ####################################################

        ####################################################
        # NEW CODE: Create train_state_1 if training pretrained agent
        ####################################################
        train_state_1 = None
        if pretrained_params is not None and config.get("train_pretrained_agent", False):
            # Create second train state with pretrained params
            train_state_1 = TrainState.create(
                apply_fn=network.apply,
                params=pretrained_params,
                tx=tx,
            )
            print("Created train_state_1 with pretrained weights - will train both agents")
        ####################################################
        # END NEW CODE
        ####################################################

        # INIT EMPOWERMENT (if enabled)
        emp_state = None
        if config.get("use_empowerment", False):
            obs_dim = init_x.shape[0]
            a_dim = env.action_space().n
            
            repr_fn_raw = make_repr_fn(
                a_dim, 
                config["repr_dim"],
                config["phi_norm"],
                config["psi_norm"]
            )
            repr_fn = hk.without_apply_rng(hk.transform(repr_fn_raw))
            
            s0 = jnp.zeros((1, obs_dim))
            a0 = jnp.array([0])
            rng, _rng = jax.random.split(rng)
            repr_params = repr_fn.init(_rng, s0, a0, a0, s0)
            
            repr_opt = optax.adam(config["repr_lr"])
            repr_opt_state = repr_opt.init(repr_params)
            
            buffer = ContrastiveBuffer.create(
                obs_dim, 
                config["buffer_size"], 
                config["emp_gamma"],
                config["batch_size"]
            )
            repr_buffer = ContrastiveBuffer.create(
                obs_dim,
                config["repr_buffer_size"],
                config["emp_gamma"],
                config["batch_size"]
            )
            
            rng, emp_rng = jax.random.split(rng)
            
            emp_state = EmpowermentState(
                repr_params=repr_params,
                repr_opt_state=repr_opt_state,
                buffer=buffer,
                repr_buffer=repr_buffer,
                emp_step=0,
                rng=emp_rng
            )
            
            repr_opt_fn = repr_opt
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, step_idx):

            def save_callback(step, params):
                step = int(step)
                params_0 = jax.tree_util.tree_map(lambda x: x, params)  # already single seed inside vmap
                ckpt = {"model": params_0, "config": config}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)

                if(step == config["NUM_UPDATES"] // 3):
                    s = "init"
                else:
                    s = "mid"
                save_path = os.path.join(os.getcwd(), args.layout, str(args.seed), f"{args.output}_{s}")
                # save only if checkpoint does not already exist
                if not os.path.exists(save_path):
                    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
                    print(f"Saved checkpoint at update {step}")
                else:
                    print(f"Checkpoint already exists at update {step}, skipping save.")



            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                ####################################################
                # NEW CODE: Unpack train_state_1 from runner_state
                ####################################################
                train_state_0, train_state_1, env_state, last_obs, rng, emp_state = runner_state
                ####################################################
                # END NEW CODE
                ####################################################

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                ####################################################
                # NEW CODE: Handle three cases for action selection
                ####################################################
                if train_state_1 is not None:
                    # Case: train_pretrained_agent=True - both agents training
                    pi_0, value_0 = network.apply(train_state_0.params, obs_batch[0::2])
                    pi_1, value_1 = network.apply(train_state_1.params, obs_batch[1::2])
                elif pretrained_params is not None:
                    # Case: pretrained agent frozen
                    pi_0, value_0 = network.apply(train_state_0.params, obs_batch[0::2])
                    pi_1, value_1 = network.apply(pretrained_params, obs_batch[1::2])
                else:
                    # Case: no pretrained, both agents share train_state_0
                    pi, value = network.apply(train_state_0.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    
                    env_act = unbatchify(action, env.agents, config["num_envs"], env.num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["num_envs"])

                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                        rng_step, env_state, env_act
                    )

                    shaped_anneal = (jnp.exp(-runner_state[0].step/1000))
                    reward["agent_0"] += info["shaped_reward"]["agent_0"]*shaped_anneal
                    reward["agent_1"] += info["shaped_reward"]["agent_1"]*shaped_anneal
                    
                    if config.get("use_empowerment", False) and emp_state is not None:
                        agent_0_obs_flat = obs_batch[0::2]
                        agent_0_action_flat = action[0::2]
                        agent_1_action_flat = action[1::2]
                        next_obs_batch = batchify(obsv, env.agents, config["NUM_ACTORS"])
                        agent_0_next_obs = next_obs_batch[0::2]
                        
                        emp_reward = compute_empowerment_reward(
                            repr_fn.apply,
                            emp_state.repr_params,
                            agent_0_obs_flat,
                            agent_0_action_flat,
                            agent_1_action_flat,
                            agent_0_next_obs,
                            config["reward_type"],
                            config["repr_dim"]
                        )
                        
                        emp_scale = config.get("emp_reward_scale", 1.0)
                        reward["agent_0"] +=  emp_reward#-10.25*emp_reward
                        #reward["agent_0"] += 0.01*jnp.exp(jnp.clip(reward["agent_0"], -10, 10))#0.01* jnp.exp(-jnp.clip(reward["agent_0"], -10, 10))
                        #reward["agent_0"] += 0.1*jnp.abs(reward["agent_0"] - reward["agent_0"].mean())
                    transition = Transition(
                        batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                    )

                    runner_state = (train_state_0, train_state_1, env_state, obsv, rng, emp_state)
                    return runner_state, transition
                
                # For cases with separate policies for each agent
                rng, rng_0, rng_1 = jax.random.split(rng, 3)
                action_0 = pi_0.sample(seed=rng_0)
                action_1 = pi_1.sample(seed=rng_1)
                
                action = jnp.empty((config["NUM_ACTORS"],), dtype=action_0.dtype)
                action = action.at[0::2].set(action_0)
                action = action.at[1::2].set(action_1)
                
                value = jnp.empty((config["NUM_ACTORS"],), dtype=value_0.dtype)
                value = value.at[0::2].set(value_0)
                value = value.at[1::2].set(value_1)
                
                log_prob_0 = pi_0.log_prob(action_0)
                log_prob_1 = pi_1.log_prob(action_1)
                log_prob = jnp.empty((config["NUM_ACTORS"],), dtype=log_prob_0.dtype)
                log_prob = log_prob.at[0::2].set(log_prob_0)
                log_prob = log_prob.at[1::2].set(log_prob_1)
                ####################################################
                # END NEW CODE
                ####################################################
                
                env_act = unbatchify(action, env.agents, config["num_envs"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                shaped_anneal = (jnp.exp(-runner_state[0].step/1000))
                reward["agent_0"] += info["shaped_reward"]["agent_0"]*shaped_anneal 
                reward["agent_1"] += info["shaped_reward"]["agent_1"]*shaped_anneal
                
                if config.get("use_empowerment", False) and emp_state is not None:
                    agent_0_obs_flat = obs_batch[0::2]
                    agent_0_action_flat = action[0::2]
                    agent_1_action_flat = action[1::2]
                    
                    next_obs_batch = batchify(obsv, env.agents, config["NUM_ACTORS"])
                    agent_0_next_obs = next_obs_batch[0::2]
                    
                    emp_reward = compute_empowerment_reward(
                        repr_fn.apply,
                        emp_state.repr_params,
                        agent_0_obs_flat,
                        agent_0_action_flat,
                        agent_1_action_flat,
                        agent_0_next_obs,
                        config["reward_type"],
                        config["repr_dim"]
                    )
                    
                    emp_scale = config.get("emp_reward_scale", 1.0)
                    
                    reward["agent_0"] += emp_reward#-10.25*emp_reward
                    #reward["agent_0"] += 0.01*jnp.exp(jnp.clip(reward["agent_0"], -10, 10))#0.01* jnp.exp(-jnp.clip(reward["agent_0"], -10, 10))
                    #reward["agent_0"] += 0.1*jnp.abs(reward["agent_0"] - reward["agent_0"].mean())
                    #reward["agent_0"] =  0.5*reward["agent_0"] + 0.5*emp_reward
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )

                ####################################################
                # NEW CODE: Include train_state_1 in runner_state
                ####################################################
                runner_state = (train_state_0, train_state_1, env_state, obsv, rng, emp_state)
                ####################################################
                # END NEW CODE
                ####################################################
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            ####################################################
            # NEW CODE: Unpack train_state_1
            ####################################################
            train_state_0, train_state_1, env_state, last_obs, rng, emp_state = runner_state
            ####################################################
            # END NEW CODE
            ####################################################
            
            if config.get("use_empowerment", False) and emp_state is not None:
                agent_0_obs = traj_batch.obs[0::2]
                agent_0_actions = traj_batch.action[0::2]
                agent_1_actions = traj_batch.action[1::2]
                
                num_envs = config["num_envs"]
                agent_0_obs = agent_0_obs.reshape(config["num_steps"], num_envs, -1)
                agent_0_actions = agent_0_actions.reshape(config["num_steps"], num_envs)
                agent_1_actions = agent_1_actions.reshape(config["num_steps"], num_envs)
                
                for env_idx in range(num_envs):
                    obs_seq = agent_0_obs[:, env_idx, :]
                    ar_seq = agent_0_actions[:, env_idx]
                    ah_seq = agent_1_actions[:, env_idx]
                    
                    g = jnp.tile(obs_seq[-1][None], (obs_seq.shape[0], 1))
                    emp_rewards = compute_empowerment_reward(
                        repr_fn.apply,
                        emp_state.repr_params,
                        obs_seq,
                        ar_seq,
                        ah_seq,
                        g,
                        config["reward_type"],
                        config["repr_dim"]
                    )
                    
                    new_buffer = emp_state.buffer.extend(obs_seq, ar_seq, ah_seq, emp_rewards)
                    new_repr_buffer = emp_state.repr_buffer.extend(obs_seq, ar_seq, ah_seq, emp_rewards)
                    emp_state = emp_state._replace(
                        buffer=new_buffer,
                        repr_buffer=new_repr_buffer
                    )
                
                def update_repr_fn(emp_state):
                    emp_rng, sample_rng = jax.random.split(emp_state.rng)
                    batch = emp_state.repr_buffer.sample(sample_rng, config["batch_size"])
                    
                    def loss_fn(params):
                        return contrastive_loss(
                            repr_fn.apply,
                            params,
                            batch.state,
                            batch.action_robot,
                            batch.action_human,
                            batch.future_state,
                            env.action_space().n,
                            config["repr_dim"],
                            config["psi_reg"]
                        )
                    
                    grad_fn = jax.grad(loss_fn, has_aux=True)
                    grad, emp_info = grad_fn(emp_state.repr_params)
                    
                    updates, new_opt_state = repr_opt_fn.update(grad, emp_state.repr_opt_state)
                    new_repr_params = optax.apply_updates(emp_state.repr_params, updates)
                    
                    return emp_state._replace(
                        repr_params=new_repr_params,
                        repr_opt_state=new_opt_state,
                        emp_step=emp_state.emp_step + 1,
                        rng=emp_rng
                    )
                
                def skip_update_fn(emp_state):
                    return emp_state._replace(emp_step=emp_state.emp_step + 1)
                
                should_update = (
                    (emp_state.emp_step % config["update_repr_freq"] == 0) & 
                    (emp_state.buffer.buflen > config["batch_size"])
                )
                
                emp_state = jax.lax.cond(
                    should_update,
                    update_repr_fn,
                    skip_update_fn,
                    emp_state
                )

            # CALCULATE ADVANTAGE
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            
            ####################################################
            # NEW CODE: Calculate last_val for both agents
            ####################################################
            if train_state_1 is not None:
                # Both agents training with separate policies
                _, last_val_0 = network.apply(train_state_0.params, last_obs_batch[0::2])
                _, last_val_1 = network.apply(train_state_1.params, last_obs_batch[1::2])
                
                last_val = jnp.empty((config["NUM_ACTORS"],), dtype=last_val_0.dtype)
                last_val = last_val.at[0::2].set(last_val_0)
                last_val = last_val.at[1::2].set(last_val_1)
            elif pretrained_params is not None:
                # Pretrained agent frozen
                _, last_val_0 = network.apply(train_state_0.params, last_obs_batch[0::2])
                _, last_val_1 = network.apply(pretrained_params, last_obs_batch[1::2])
                
                last_val = jnp.empty((config["NUM_ACTORS"],), dtype=last_val_0.dtype)
                last_val = last_val.at[0::2].set(last_val_0)
                last_val = last_val.at[1::2].set(last_val_1)
            else:
                # Both agents share train_state_0
                _, last_val = network.apply(train_state_0.params, last_obs_batch)
            ####################################################
            # END NEW CODE
            ####################################################

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, epoch_idx):
                ####################################################
                # NEW CODE: Separate update functions for each agent
                ####################################################
                def _update_minbatch(train_states, batch_info):
                    train_state_0, train_state_1 = train_states
                    traj_batch, advantages, targets = batch_info

                    # Update agent_0
                    def _loss_fn_0(params, traj_batch, gae, targets):
                        agent_0_indices = jnp.arange(0, traj_batch.obs.shape[0], 2)
                        
                        obs_agent_0 = traj_batch.obs[agent_0_indices]
                        action_agent_0 = traj_batch.action[agent_0_indices]
                        value_agent_0 = traj_batch.value[agent_0_indices]
                        log_prob_agent_0 = traj_batch.log_prob[agent_0_indices]
                        gae_agent_0 = gae[agent_0_indices]
                        targets_agent_0 = targets[agent_0_indices]
                        
                        pi, value = network.apply(params, obs_agent_0)
                        log_prob = pi.log_prob(action_agent_0)
                        
                        value_pred_clipped = value_agent_0 + (value - value_agent_0).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets_agent_0)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_agent_0)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - log_prob_agent_0)
                        gae_normalized = (gae_agent_0 - gae_agent_0.mean()) / (gae_agent_0.std() + 1e-8)
                        loss_actor1 = ratio * gae_normalized
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        #ent_coef = 0.02 if config.get("use_empowerment", False) else config["ent_coef"]
                        ent_coef = 0.01 if config.get("use_empowerment", False) else config["ent_coef"]
                        total_loss = loss_actor + config["vf_coef"] * value_loss - ent_coef * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn_0 = jax.value_and_grad(_loss_fn_0, has_aux=True)
                    total_loss_0, grads_0 = grad_fn_0(train_state_0.params, traj_batch, advantages, targets)
                    train_state_0 = train_state_0.apply_gradients(grads=grads_0)

                    # Update agent_1 if train_state_1 exists
                    if train_state_1 is not None:
                        def _loss_fn_1(params, traj_batch, gae, targets):
                            agent_1_indices = jnp.arange(1, traj_batch.obs.shape[0], 2)
                            
                            obs_agent_1 = traj_batch.obs[agent_1_indices]
                            action_agent_1 = traj_batch.action[agent_1_indices]
                            value_agent_1 = traj_batch.value[agent_1_indices]
                            log_prob_agent_1 = traj_batch.log_prob[agent_1_indices]
                            gae_agent_1 = gae[agent_1_indices]
                            targets_agent_1 = targets[agent_1_indices]
                            
                            pi, value = network.apply(params, obs_agent_1)
                            log_prob = pi.log_prob(action_agent_1)
                            
                            value_pred_clipped = value_agent_1 + (value - value_agent_1).clip(
                                -config["clip_eps"], config["clip_eps"]
                            )
                            value_losses = jnp.square(value - targets_agent_1)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets_agent_1)
                            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                            ratio = jnp.exp(log_prob - log_prob_agent_1)
                            gae_normalized = (gae_agent_1 - gae_agent_1.mean()) / (gae_agent_1.std() + 1e-8)
                            loss_actor1 = ratio * gae_normalized
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["clip_eps"],
                                    1.0 + config["clip_eps"],
                                )
                                * gae_normalized
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()
                            
                            ent_coef = config["ent_coef"]  # No empowerment for agent_1
                            
                            total_loss = loss_actor + config["vf_coef"] * value_loss - ent_coef * entropy
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn_1 = jax.value_and_grad(_loss_fn_1, has_aux=True)
                        total_loss_1, grads_1 = grad_fn_1(train_state_1.params, traj_batch, advantages, targets)
                        
                        
                        should_update = (epoch_idx%10)
                        #grads_1 = jax.tree_util.tree_map(
                        #          lambda g: jnp.where(should_update, g, jnp.zeros_like(g)), grads_1
                        #)
                        ''' 
                        train_state_1 = jax.lax.cond(
                                                     should_update,
                                                     lambda ts: ts.apply_gradients(grads=grads_1),
                                                     lambda ts: ts,  # returns unchanged train_state_1
                                                     train_state_1
                        )
                        '''
                        train_state_1 = train_state_1.apply_gradients(grads=grads_1)
                        



                        return (train_state_0, train_state_1), (total_loss_0, total_loss_1)
                    else:
                        return (train_state_0, train_state_1), (total_loss_0, None)
                ####################################################
                # END NEW CODE
                ####################################################

                ####################################################
                # NEW CODE: Handle train_state_1 in update_state
                ####################################################
                train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state = update_state
                ####################################################
                # END NEW CODE
                ####################################################
                
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["num_minibatches"]
                assert (
                    batch_size == config["num_steps"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                
                traj_batch_no_info = Transition(
                    done=traj_batch.done,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob,
                    obs=traj_batch.obs,
                    info=None
                )
                
                batch = (traj_batch_no_info, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]) if x is not None else None, 
                    batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0) if x is not None else None, 
                    batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["num_minibatches"], -1] + list(x.shape[1:])) if x is not None else None,
                    shuffled_batch,
                )
                
                ####################################################
                # NEW CODE: Update both train states
                ####################################################
                (train_state_0, train_state_1), total_loss = jax.lax.scan(
                    _update_minbatch, 
                    (train_state_0, train_state_1), 
                    minibatches
                )
                
                update_state = (train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state)
                ####################################################
                # END NEW CODE
                ####################################################
                return update_state, total_loss

            ####################################################
            # NEW CODE: Include train_state_1 in update_state
            ####################################################
            update_state = (train_state_0, train_state_1, traj_batch, advantages, targets, rng, emp_state)
            ####################################################
            # END NEW CODE
            ####################################################
            
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, jnp.arange(config["update_epochs"]))
            
            ####################################################
            # NEW CODE: Unpack both train states
            ####################################################
            train_state_0 = update_state[0]
            train_state_1 = update_state[1]
            ####################################################
            # END NEW CODE
            ####################################################
            
            # save chackpint for profficiency 
            jax.lax.cond(
                (step_idx == config["NUM_UPDATES"] // 3) |
                (step_idx == (config["NUM_UPDATES"] * 2) // 3),
                lambda p: jax.debug.callback(save_callback, step_idx, p),
                lambda p: None,
                train_state_0.params,
            )



            metric = traj_batch.info
            rng = update_state[5]
            emp_state = update_state[6]

            ####################################################
            # NEW CODE: Include train_state_1 in runner_state
            ####################################################
            runner_state = (train_state_0, train_state_1, env_state, last_obs, rng, emp_state)
            ####################################################
            # END NEW CODE
            ####################################################
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        ####################################################
        # NEW CODE: Include train_state_1 in initial runner_state
        ####################################################
        runner_state = (train_state_0, train_state_1, env_state, obsv, _rng, emp_state)
        ####################################################
        # END NEW CODE
        ####################################################
        runner_state, metric = jax.lax.scan(_update_step, runner_state, jnp.arange(config["NUM_UPDATES"]))
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=5e6)#96e6)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)#0.84)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)#0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--env_name", type=str, default="overcooked")
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--output", type=str, default="self_play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--shaped_reward_scale", type=float, default=0.0)
    parser.add_argument("--initial_checkpoint", type=str, default=None)
    parser.add_argument("--pretrained_agent_path", type=str, default=None, 
                        help="Path to pretrained agent checkpoint (agent_1 will use this)")
    
    ####################################################
    # NEW CODE: Add train_pretrained_agent argument
    ####################################################
    parser.add_argument("--train_pretrained_agent", action="store_true",
                        help="Train the pretrained agent (agent_1) while agent_0 trains from scratch. Requires --pretrained_agent_path")
    ####################################################
    # END NEW CODE
    ####################################################
    
    # Empowerment arguments
    parser.add_argument("--use_empowerment", action="store_true", 
                        help="Enable empowerment learning for agent_0")
    parser.add_argument("--repr_lr", type=float, default=3e-5, 
                        help="Representation learning rate for empowerment")
    parser.add_argument("--repr_dim", type=int, default=32, 
                        help="Representation dimension for empowerment")
    parser.add_argument("--buffer_size", type=int, default=200000, 
                        help="Buffer size for empowerment")
    parser.add_argument("--repr_buffer_size", type=int, default=120000, #80000,
                        help="Representation buffer size for empowerment")
    parser.add_argument("--emp_gamma", type=float, default=0.8,
                        help="Discount factor for empowerment buffer")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for empowerment updates")
    parser.add_argument("--update_repr_freq", type=int, default=100,
                        help="Update frequency for representation network")
    parser.add_argument("--reward_type", type=str, default="dot", choices=["dot", "norm", "diff"],
                        help="Empowerment reward function type")
    parser.add_argument("--psi_reg", type=float, default=0.0,
                        help="Regularization on psi for empowerment")
    parser.add_argument("--phi_norm", action="store_true",
                        help="Normalize phi in empowerment")
    parser.add_argument("--psi_norm", action="store_true",
                        help="Normalize psi in empowerment")
    parser.add_argument("--emp_reward_scale", type=float, default=1.0,
                        help="Scaling factor for empowerment reward")
    parser.add_argument("--risk_reward_scale", type=float, default=1.0,
                        help="Scaling factor for empowerment reward")


    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    args.save=True
    args.pretrained_agent_path = os.path.join(os.getcwd(), f"phase0/counter_circuit/self_play")
    args.use_empowerment = True
    args.train_pretrained_agent = True
    wandbid = wandb.util.generate_id(4)
    wandb_mode = "disabled" if args.no_wandb else "online"
    os.environ["WANDB_API_KEY"] = "495b87eba3dbc88f719508680483181c811852ba"
    
    ####################################################
    # NEW CODE: Updated group name logic
    ####################################################
    group_name = "self_play_" + args.layout
    if args.pretrained_agent_path and not args.train_pretrained_agent:
        group_name = "pretrained_frozen_" + args.layout
    if args.train_pretrained_agent:
        group_name = "pretrained_training_" + args.layout
    if args.use_empowerment:
        group_name = "empowerment_" + args.layout
    ####################################################
    # END NEW CODE
    ####################################################
    
    wandb.init(project="empowerment", config=vars(args), id=wandbid, group=group_name, mode=wandb_mode)
    print("layouts:", overcooked_layouts)
    
    config = {
        "env_kwargs": {
            "layout": overcooked_layouts[args.layout],
        },
        "num_seeds": 1,
    }

    config.update(vars(args))

    rng = jax.random.PRNGKey(config["seed"])

    # Load pretrained agent BEFORE JIT if specified
    pretrained_params = None
    if config.get("pretrained_agent_path") is not None:
        print(f"Loading pretrained agent from {config['pretrained_agent_path']}")
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        pretrained_ckpt = orbax_checkpointer.restore(config["pretrained_agent_path"])
        
        print("Checkpoint keys:", pretrained_ckpt.keys())
        if "model" in pretrained_ckpt:
            print("Model type:", type(pretrained_ckpt["model"]))
            if isinstance(pretrained_ckpt["model"], dict):
                print("Model keys:", pretrained_ckpt["model"].keys())
            elif isinstance(pretrained_ckpt["model"], (list, tuple)):
                print("Model length:", len(pretrained_ckpt["model"]))
        
        try:
            raw_params = pretrained_ckpt["model"]["params"]
        except (KeyError, TypeError):
            try:
                raw_params = pretrained_ckpt["model"][0].params
            except (KeyError, TypeError, IndexError):
                try:
                    raw_params = pretrained_ckpt["model"].params
                except (KeyError, TypeError, AttributeError):
                    raw_params = pretrained_ckpt["params"]
        
        print("Raw params structure (before squeeze):")
        for key, val in jax.tree_util.tree_flatten_with_path(raw_params)[0][:5]:
            print(f"  {key}: {val.shape}")
        
        def squeeze_batch_dim(params):
            """Remove the first dimension from all parameters (batch dimension)"""
            return jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if x.shape[0] == 1 else x,
                params,
            )
        
        pretrained_params = squeeze_batch_dim(raw_params)
        
        ####################################################
        # NEW CODE: Updated loading message
        ####################################################
        if config.get("train_pretrained_agent", False):
            print("Pretrained agent loaded - agent_1 will train from these weights, agent_0 trains from scratch")
        else:
            print("Pretrained agent loaded - agent_1 will be frozen, agent_0 trains from scratch")
        ####################################################
        # END NEW CODE
        ####################################################
        
        print("Pretrained params structure (after squeeze):")
        for key, val in jax.tree_util.tree_flatten_with_path(pretrained_params)[0][:5]:
            print(f"  {key}: {val.shape}")

    wandb.define_metric("evaluation/reward", step_metric="episode")
    rngs = jax.random.split(rng, config["num_seeds"])
    
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, pretrained_params)))
        out = train_jit(rngs)
        
        wandb.log({"evaluation/reward":out["metrics"]["returned_episode_returns"][0].mean(-1).reshape(-1)[-1]}, step=0)

        if not args.plot:
            print(out["metrics"]["returned_episode_returns"])
            for data in out["metrics"]["returned_episode_returns"]:
                plt.plot(data.mean(-1).reshape(-1))
                print("Final true mean reward: ", data.mean(-1).reshape(-1)[-1000:].mean())

        print("Generating final rollout video...")
        env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
        
        ####################################################
        # NEW CODE: Handle train_state_1 extraction
        ####################################################
        
        #final_runner_state = out["runner_state"][0]
        train_state_0 = out["runner_state"][0] #final_runner_state[0]
        train_state_1 = out["runner_state"][1] #final_runner_state[1]
        
        def squeeze_batch_dim(params):
            """Remove the first dimension from all parameters (batch dimension from vmap)"""
            return jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if x.ndim > 1 and x.shape[0] == 1 else x,
                params,
            )
        
        train_state_0_no_batch = train_state_0.replace(
            params=squeeze_batch_dim(train_state_0.params)
        )
        
        train_state_1_no_batch = None
        if train_state_1 is not None:
            train_state_1_no_batch = train_state_1.replace(
                params=squeeze_batch_dim(train_state_1.params)
            )
        
        state_seq = get_rollout(train_state_0_no_batch, config, train_state_1_no_batch)
        ####################################################
        # END NEW CODE
        ####################################################
        
        video_frames = render_state_seq(state_seq, env)
        
        wandb.log({
            "evaluation/final_video": wandb.Video(video_frames, fps=4, format="mp4")
        })
        print(f"Video logged! Rollout length: {len(state_seq)} steps")

    if args.save:
        state = out["runner_state"][0]
        ckpt = {"model": state, "config": config}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        path = os.path.join(os.getcwd(), "phase_1", args.layout, str(args.seed), "final",f"{args.output}")
        if not os.path.exists(path):
            orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
            print(f"Saved final checkpoint")
        else:
            print(f"Final checkpoint already exists, skipping save.")
