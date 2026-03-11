import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
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
import shutil

# ============================================================================
# NETWORK
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


# ============================================================================
# HELPERS
# ============================================================================
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def load_population(population_dir, num_agents, agent_name_template="agent_{}"):
    """Load a list of pretrained partner params from orbax checkpoints."""
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    population_params = []
    for i in range(num_agents):
        agent_path = os.path.join(population_dir)#, agent_name_template.format(i))
        ckpt = orbax_checkpointer.restore(agent_path)
        try:
            params = ckpt["model"]["params"]
        except (KeyError, TypeError):
            params = ckpt["params"]
        # squeeze leading seed dim if present
        params = jax.tree_util.tree_map(
            lambda x: x.squeeze(0) if x.ndim > 0 and x.shape[0] == 1 else x, params
        )
        population_params.append(params)
        print(f"Loaded partner {i} from {agent_path}")
    # Stack into a single pytree with leading axis = population_size
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *population_params)
    return stacked  # shape: (pop_size, ...)


# ============================================================================
# TRAINING
# ============================================================================
def make_train(config, population_params):
    """
    population_params: stacked pytree with leading axis = population_size.
    Each partner is held fixed (no gradient updates).
    The ego agent (agent_0) is trained via PPO against each partner in turn.
    Partner switches every `updates_per_partner` PPO update steps.
    """
    env = jaxmarl.make(config["env_name"], **config["env_kwargs"])

    pop_size = config["population_size"]
    config["NUM_ACTORS"] = env.num_agents * config["num_envs"]
    # Total updates = outer (pop cycling) * inner (updates per partner)
    config["NUM_UPDATES"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["MINIBATCH_SIZE"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )
    # How many full PPO update steps to spend with each partner before switching
    updates_per_partner = config["updates_per_partner"]
    # Number of full passes through the population
    num_partner_rounds = config["NUM_UPDATES"] // (updates_per_partner * pop_size)

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["NUM_UPDATES"]
        return config["lr"] * frac

    def train(rng):
        network = ActorCritic(env.action_space().n, activation=config["activation"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape).flatten()

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
            apply_fn=network.apply,
            params=network.init(_rng, init_x),
            tx=tx,
        )

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # ------------------------------------------------------------------ #
        # INNER LOOP: one PPO update step against a fixed partner
        # ------------------------------------------------------------------ #
        def _update_step(runner_state, unused):
            ego_state, env_state, last_obs, rng, partner_params = runner_state

            # -- collect trajectories --
            def _env_step(runner_state, unused):
                ego_state, env_state, last_obs, rng, partner_params = runner_state

                n = config["num_envs"]
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # obs_batch[:n] = agent_0 (ego), obs_batch[n:] = agent_1 (partner)

                rng, rng_ego, rng_partner = jax.random.split(rng, 3)

                pi_ego, value_ego = network.apply(ego_state.params, obs_batch[:n])
                pi_partner, _     = network.apply(partner_params,   obs_batch[n:])

                action_ego     = pi_ego.sample(seed=rng_ego)
                action_partner = pi_partner.sample(seed=rng_partner)
                log_prob_ego   = pi_ego.log_prob(action_ego)

                # Build full action dict
                action_full = jnp.concatenate([action_ego, action_partner], axis=0)
                env_act = unbatchify(action_full, env.agents, config["num_envs"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                # shaped reward for ego only
                reward["agent_0"] += info["shaped_reward"]["agent_0"]

                # Store only ego agent's transition (agent_0)
                done_ego   = batchify(done,   env.agents, config["NUM_ACTORS"])[:n].squeeze()
                reward_ego = batchify(reward, env.agents, config["NUM_ACTORS"])[:n].squeeze()

                transition = Transition(
                    done=done_ego,
                    action=action_ego,
                    value=value_ego,
                    reward=reward_ego,
                    log_prob=log_prob_ego,
                    obs=obs_batch[:n],
                    info=info,
                )

                runner_state = (ego_state, env_state, obsv, rng, partner_params)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )
            ego_state, env_state, last_obs, rng, partner_params = runner_state

            # -- compute GAE --
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            n = config["num_envs"]
            _, last_val = network.apply(ego_state.params, last_obs_batch[:n])

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # -- PPO update (ego only) --
            def _update_epoch(update_state, unused):
                def _update_minbatch(ego_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()

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
                    total_loss, grads = grad_fn(ego_state.params, traj_batch, advantages, targets)
                    ego_state = ego_state.apply_gradients(grads=grads)
                    return ego_state, total_loss

                ego_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["num_minibatches"]
                assert batch_size == config["num_steps"] * config["num_envs"]

                permutation = jax.random.permutation(_rng, batch_size)

                traj_no_info = Transition(
                    done=traj_batch.done, action=traj_batch.action,
                    value=traj_batch.value, reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob, obs=traj_batch.obs, info=None,
                )
                batch = (traj_no_info, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]) if x is not None else None, batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0) if x is not None else None, shuffled_batch := batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0) if x is not None else None, batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["num_minibatches"], -1] + list(x.shape[1:])) if x is not None else None,
                    shuffled_batch,
                )

                ego_state, total_loss = jax.lax.scan(_update_minbatch, ego_state, minibatches)
                update_state = (ego_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (ego_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["update_epochs"]
            )
            ego_state = update_state[0]
            rng = update_state[-1]

            metric = traj_batch.info
            runner_state = (ego_state, env_state, last_obs, rng, partner_params)
            return runner_state, metric

        # ------------------------------------------------------------------ #
        # MIDDLE LOOP: hold one partner fixed for `updates_per_partner` steps
        # ------------------------------------------------------------------ #
        def _partner_phase(runner_state, partner_idx):
            ego_state, env_state, last_obs, rng = runner_state

            # Select partner params by index from stacked population
            partner_params = jax.tree_util.tree_map(
                lambda x: x[partner_idx], population_params
            )

            inner_runner = (ego_state, env_state, last_obs, rng, partner_params)
            inner_runner, metrics = jax.lax.scan(
                _update_step, inner_runner, None, updates_per_partner
            )
            ego_state, env_state, last_obs, rng, _ = inner_runner

            # Print partner index and mean episode reward after finishing with this partner.
            # We use returned_episode_returns which is only non-zero at episode boundaries,
            # so we take the mean of nonzero values as the reward estimate.
            ep_returns = metrics["returned_episode_returns"]  # shape: (updates_per_partner, num_envs)
            flat = ep_returns.reshape(-1)
            mean_rew = jnp.where(flat != 0, flat, jnp.nan)
            mean_rew = jnp.nanmean(mean_rew)
            jax.debug.print(
                "  Finished partner {partner_idx} | mean reward: {mean_rew}",
                partner_idx=partner_idx,
                mean_rew=mean_rew,
            )

            return (ego_state, env_state, last_obs, rng), metrics

        # ------------------------------------------------------------------ #
        # OUTER LOOP: cycle through all partners repeatedly
        # ------------------------------------------------------------------ #
        def _population_round(runner_state, round_idx):
            jax.debug.print(
                "Round {round_idx}/{total_rounds}",
                round_idx=round_idx + 1,
                total_rounds=num_partner_rounds,
            )
            partner_indices = jnp.arange(pop_size)
            runner_state, metrics = jax.lax.scan(_partner_phase, runner_state, partner_indices)
            return runner_state, metrics

        jax.debug.print(
            "Starting Phase 2: {total_rounds} rounds x {pop_size} partners x {ups} updates each",
            total_rounds=num_partner_rounds,
            pop_size=pop_size,
            ups=updates_per_partner,
        )

        rng, _rng = jax.random.split(rng)
        runner_state = (ego_state, env_state, obsv, _rng)
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
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=5e6)
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
    parser.add_argument("--output", type=str, default="phase2_ego")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--population_dir", type=str, default="/home/tom.danino/zero_shot_jax/cramped_room/self_play/",
                        help="Directory containing one subdir per population agent")
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--updates_per_partner", type=int, default=10,
                        help="Number of PPO update steps to spend with each partner before switching")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
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

    # Load population
    population_params = load_population(
        config["population_dir"],
        config["population_size"],
    )
    print(f"Loaded population of {config['population_size']} agents.")

    rng = jax.random.PRNGKey(config["seed"])
    rngs = jax.random.split(rng, config["num_seeds"])

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, population_params)))
        out = train_jit(rngs)

        returns = out["metrics"]["returned_episode_returns"]
        for data in returns:
            flat = data.reshape(-1)
            nonzero = flat[flat != 0]
            if len(nonzero) > 0:
                print(f"Final mean reward (nonzero): {float(nonzero[-1000:].mean()):.2f}")
                plt.plot(nonzero)

        wandb.log({
            "evaluation/reward": float(nonzero[-1000:].mean()) if len(nonzero) > 0 else 0.0
        })

    if args.save:
        ego_train_state = out["runner_state"][0]
        ckpt = {"model": ego_train_state, "config": config}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        save_path = os.path.join(os.getcwd(), "phase2", args.layout, str(args.seed), args.output)
        os.makedirs(save_path, exist_ok=True)
        orbax_checkpointer.save(save_path, ckpt, save_args=save_args, force=True)
        print(f"Saved ego agent to {save_path}")
