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
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer  # ADDED
from flax.training import orbax_utils
import orbax
import matplotlib.pyplot as plt
import os
import argparse
import wandb
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
from omegaconf import OmegaConf
from jax_tqdm import scan_tqdm
import shutil

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
            # stack all layouts
            stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
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
        for layout_name, layout_dict in overcooked_layouts.items():  # add hand crafted ones to heldout set
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(train_state, config):
    #env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"]+"_9"
    config_env["SEED"] = config["seed"]
    env = initialize_environment(config_env)

    network = ActorCritic(env.action_space().n, activation=config["activation"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs = {k: v.flatten() for k, v in obs.items()}

        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


# ADDED
def render_state_seq(state_seq, env):
    padding = env.agent_view_size - 2
    def get_frame(state):
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        return OvercookedVisualizer._render_grid(
            grid, tile_size=32, highlight_mask=None,
            agent_dir_idx=state.agent_dir_idx, agent_inv=state.agent_inv)
    frames = np.stack([get_frame(s) for s in state_seq])   # (T, H, W, C)
    return np.moveaxis(frames, -1, 1)                       # (T, C, H, W) for wandb.Video


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    #env = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    config_env = OmegaConf.load("config/ippo_final.yaml")
    config_env = OmegaConf.to_container(config_env)
    config_env["ENV_KWARGS"]["layout"] = config["layout"]+"_9"
    config_env["SEED"] = config["seed"]
    env = initialize_environment(config_env)
    
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["num_envs"], env.num_agents)

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                shaped_anneal = (jnp.exp(-runner_state[0].step/5000))
                reward["agent_0"] += info["shaped_reward"]["agent_0"] 
                #reward["agent_0"] += config["risk"]*jnp.exp(jnp.clip(reward["agent_0"], -10, 10))
                reward["agent_1"] += info["shaped_reward"]["agent_1"]
                #reward["agent_1"] += config["risk"]*jnp.exp(jnp.clip(reward["agent_1"], -10, 10))

                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )

                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

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
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
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
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=20e6)#55e6)
    parser.add_argument("--update_epochs", type=int, default=20)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default= 0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--env_name", type=str, default="overcooked")
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--output", type=str, default="self_play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="forced_coord")
    parser.add_argument("--shaped_reward_scale", type=float, default=0.0)
    parser.add_argument("--initial_checkpoint", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--risk", type=float, default=0.5)
    parser.add_argument("--type", type=str, default="other")
    args = parser.parse_args()
    args.save=True
    wandbid = wandb.util.generate_id(4)
    wandb_mode = "disabled" if args.no_wandb else "online"


    if(args.type == "achiever"):
        args.lr = 0.1e6

    wandb.init(project="empowerment", config=vars(args), id=wandbid, group="self_play_" + args.layout, mode=wandb_mode)

    config = {
        "env_kwargs": {
            "layout": overcooked_layouts[args.layout],
        },
        "num_seeds": 1,
    }

    config.update(vars(args))

    rng = jax.random.PRNGKey(config["seed"])

    wandb.define_metric("evaluation/reward", step_metric="episode")
    rngs = jax.random.split(rng, config["num_seeds"])
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config)))
        out = train_jit(rngs)

        wandb.log({"evaluation/reward":out["metrics"]["returned_episode_returns"][0].mean(-1).reshape(-1)[-1]}, step=0)

        if not args.plot:
            print(out["metrics"]["returned_episode_returns"])
            for data in out["metrics"]["returned_episode_returns"]:
                plt.plot(data.mean(-1).reshape(-1))
                print("Final true mean reward: ", data.mean(-1).reshape(-1)[-1000:].mean())

    # ADDED: Record rollout video and upload to wandb
    env_vis = jaxmarl.make(config["env_name"], **config["env_kwargs"])
    train_state = out["runner_state"][0]
    train_state = train_state.replace(params=jax.tree_util.tree_map(
        lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, train_state.params))
    state_seq = get_rollout(train_state, config)
    video_frames = render_state_seq(state_seq, env_vis)
    wandb.log({"evaluation/final_video": wandb.Video(video_frames, fps=4, format="mp4")})
    print(f"Video uploaded to wandb ({len(state_seq)} frames)")

    if args.save:
        state = out["runner_state"][0]
        ckpt = {"model": state, "config": config}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        if(args.seed==0):
            save_path = os.path.join(os.getcwd(), "phase0/"+args.layout+ "/self_play")
            if not os.path.exists(save_path):
                print("Saved final checkpoint.")
                orbax_checkpointer.save(os.path.join(os.getcwd(), "phase0/"+args.layout+ "/self_play"), ckpt, save_args=save_args)
            else:
                print(f"Checkpoint already exists at update, re-save")
                shutil.rmtree(save_path)
                orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
        else:
            save_path = os.path.join(os.getcwd(), "phase0" ,args.layout, str(args.seed), f"{args.output}")
            if not os.path.exists(save_path):
                print("Saved final checkpoint.")
                orbax_checkpointer.save(os.path.join(os.getcwd(), "phase0" ,args.layout, str(args.seed), f"{args.output}"), ckpt, save_args=save_args)
            else:
                print(f"Checkpoint already exists at update, re-save")
                shutil.rmtree(save_path)
                orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
