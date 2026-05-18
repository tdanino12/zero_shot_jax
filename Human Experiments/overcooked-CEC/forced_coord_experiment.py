from dotenv import load_dotenv
from flax import struct
import jax
import jax.numpy as jnp
import os
import pickle

from jaxmarl.viz.overcooked_jitted_visualizer import render_fn as overcooked_render_fn
from jaxmarl.environments.overcooked import Overcooked, Actions, State
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_asymm_advantages_9x9, make_coord_ring_9x9, make_forced_coord_9x9, make_forced_coord_9x9, make_cramped_room_9x9
import jaxmarl

from nicegui import ui, app
import nicewebrl
from nicewebrl import MultiAgentJaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, MultiAgentEnvStage, FeedbackStage, Block, prepare_blocks, generate_stage_order
from nicewebrl import get_logger
from actor_networks import ActorCriticRNN, ScannedRNN, ActorCriticE3T
import flax.linen as nn
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal


class ActorCriticRNN2(nn.Module):
    """MOE-based actor-critic. Call signature: (hidden, obs, dones)."""
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    obs_shape: tuple = (7, 7, 26)
    moe: int = 1
    gumbel_tau: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        reshaped_obs = obs.reshape((-1, *self.obs_shape))
        embedding = nn.Conv(features=64, kernel_size=(2, 2),
                            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(reshaped_obs)
        embedding = nn.relu(embedding)
        embedding = nn.Conv(features=32, kernel_size=(2, 2),
                            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.relu(embedding)
        batch_size, num_envs, _ = obs.shape
        embedding = embedding.reshape((batch_size, num_envs, -1))
        embedding = nn.Dense(self.hidden_size * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size * 3 // 4, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        routing_logits = nn.Dense(self.moe, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        gate = jax.nn.softmax(routing_logits, axis=-1)
        expert_outputs = []
        for k in range(self.moe):
            expert_out = nn.Dense(self.hidden_size // 4, kernel_init=orthogonal(jnp.sqrt(2)),
                                  bias_init=constant(0.0), name=f"expert_{k}")(actor_mean)
            expert_out = jnp.tanh(expert_out) if self.activation == "tanh" else nn.relu(expert_out)
            expert_outputs.append(expert_out)
        expert_outputs = jnp.stack(expert_outputs, axis=-2)
        actor_mean = jnp.sum(expert_outputs * gate[..., None], axis=-2)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(self.hidden_size * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size * 3 // 4, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class ActorCriticRNN3(nn.Module):
    """Plain MLP actor-critic (no MoE). Call signature: (hidden, obs, dones)."""
    action_dim: int
    hidden_size: int = 120
    activation: str = "tanh"
    obs_shape: tuple = (7, 7, 26)

    @nn.compact
    def __call__(self, hidden, obs, dones):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        reshaped_obs = obs.reshape((-1, *self.obs_shape))
        embedding = nn.Conv(features=64, kernel_size=(2, 2),
                            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(reshaped_obs)
        embedding = nn.relu(embedding)
        embedding = nn.Conv(features=32, kernel_size=(2, 2),
                            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.relu(embedding)
        batch_size, num_envs, _ = obs.shape
        embedding = embedding.reshape((batch_size, num_envs, -1))
        embedding = nn.Dense(self.hidden_size * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size * 3 // 4, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.hidden_size // 4, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(self.hidden_size * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size * 3 // 4, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(self.hidden_size // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class ModelWrapper:
    """Adapts ActorCriticRNN2/RNN3 (which take obs, dones separately) to nicewebrl's
    model.apply(params, hidden, (obs, dones, agent_pos)) call signature."""
    def __init__(self, model):
        self._model = model

    def apply(self, params, hidden, x):
        obs, dones, agent_positions = x
        return self._model.apply(params, hidden, obs, dones)

    def __getattr__(self, name):
        return getattr(self._model, name)
import pdb
import asyncio

load_dotenv()

logger = get_logger(__name__)
# Populated by web_app.index() on every page load with the newest stage_container.
# display_fn uses this to render into the correct client's DOM.
_active_containers = {}
VERBOSITY = int(os.environ.get('VERBOSITY', 0))
DEBUG = int(os.environ.get('DEBUG', 0))
WORLD_SEED = int(os.environ.get('WORLD_SEED', 1))
NAME = os.environ.get('NAME', 'forced_coord')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 400   # 400 steps for each model's environment stage
TUTORIAL_EPISODE_TIMESTEPS = 50  # 50 steps for the warm-up / tutorial stage
MIN_SUCCESS_EPISODES = 100

def get_user_save_file_fn():
    return f'{DATA_DIR}/user={app.storage.user.get("seed")}_name={NAME}_debug={DEBUG}.json'


########################################
# Define actions and corresponding keys
########################################
actions = [Actions.up, Actions.down, Actions.left, Actions.right, Actions.stay, Actions.interact]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp", "s", " "]
action_to_name = [a.name for a in actions]

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
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, forced_coord_reset, forced_coord_reset, cramped_room_reset]
            stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree_map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset

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
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env

import yaml
from pathlib import Path
base_config = yaml.safe_load(Path('overcooked_config.yaml').read_text())
print(f"[forced_coord_experiment] YAML base_config FC_DIM_SIZE={base_config.get('FC_DIM_SIZE')} GRU_HIDDEN_DIM={base_config.get('GRU_HIDDEN_DIM')}")
base_config['ENV_NAME'] = 'overcooked'
base_config['ENV_KWARGS']['check_held_out'] = False
base_config['ENV_KWARGS']['shuffle_inv_and_pot'] = False
base_config['ENV_KWARGS']['random_reset'] = False
base_config['ENV_KWARGS']['random_reset_fn'] = 'reset_all'
base_config['ENV_KWARGS']['layout'] = 'forced_coord_9'
base_config['ENV_KWARGS']['max_steps'] = MAX_EPISODE_TIMESTEPS - 1
base_config['GRAPH_NET'] = True

tutorial_config = pickle.loads(pickle.dumps(base_config))
tutorial_config['ENV_KWARGS']['layout'] = 'forced_coord_9'  # same map as experiment


########################################
# Define Overcooked environment
########################################
jax_env = initialize_environment(base_config)
jax_env_tutorial = initialize_environment(tutorial_config)

default_params = {'random_reset_fn': 0}

########################################
# Load agent models
########################################
base_agent_model = ActorCriticRNN(action_dim=len(actions), config=base_config)
e3t_agent_model = ActorCriticE3T(action_dim=len(actions), config=base_config)

# These checkpoints were trained with GRAPH_NET=False (path says 'graphFalse')
# and FC_DIM_SIZE=256, GRU_HIDDEN_DIM=256 (from training config).
no_graph_config = dict(base_config)
no_graph_config['GRAPH_NET'] = False
no_graph_config['FC_DIM_SIZE'] = 128
no_graph_config['GRU_HIDDEN_DIM'] = 128
# Verify the config is correct before building models
assert no_graph_config['FC_DIM_SIZE'] == 128, f"FC_DIM_SIZE mismatch: {no_graph_config['FC_DIM_SIZE']}"
print(f"[forced_coord_experiment] no_graph_config FC_DIM_SIZE={no_graph_config['FC_DIM_SIZE']} GRU_HIDDEN_DIM={no_graph_config['GRU_HIDDEN_DIM']} GRAPH_NET={no_graph_config['GRAPH_NET']}")

ik_finetune_model = ActorCriticRNN(action_dim=len(actions), config=no_graph_config)
sk_e3t_model = ActorCriticE3T(action_dim=len(actions), config=no_graph_config)
# Verify model configs at creation time
print(f"[forced_coord_experiment] ik_finetune_model FC_DIM_SIZE={ik_finetune_model.config['FC_DIM_SIZE']} id={id(ik_finetune_model.config)}")
print(f"[forced_coord_experiment] no_graph_config id={id(no_graph_config)}")

MY_HIDDEN_SIZE = 120   # hidden_size used for both fcp and my_model
MY_MOE = 1             # number of experts for my_model (checkpoint was saved with moe=1)
MY_ACHIEVER = 'standard'  # achiever type for my_model path (adjust if needed)
MY_LAYOUT = 'forced_coord'  # layout name used in the my/fcp checkpoint paths

# Instantiate models
# fcp uses ActorCriticRNN3, my uses ActorCriticRNN2
# Both are wrapped so nicewebrl can call model.apply(params, hidden, (obs, dones, agent_pos))
fcp_model_inner = ActorCriticRNN3(
    action_dim=len(actions),
    hidden_size=MY_HIDDEN_SIZE,
    activation="tanh",
    obs_shape=jax_env.observation_space().shape,
)
my_model_inner = ActorCriticRNN2(
    action_dim=len(actions),
    hidden_size=MY_HIDDEN_SIZE,
    activation="tanh",
    obs_shape=jax_env.observation_space().shape,
    moe=MY_MOE,
)
fcp_model = ModelWrapper(fcp_model_inner)
my_model = ModelWrapper(my_model_inner)

mep_model_inner = ActorCriticRNN3(
    action_dim=len(actions),
    hidden_size=MY_HIDDEN_SIZE,
    activation="tanh",
    obs_shape=jax_env.observation_space().shape,
)
mep_model = ModelWrapper(mep_model_inner)

hsp_model_inner = ActorCriticRNN3(
    action_dim=len(actions),
    hidden_size=MY_HIDDEN_SIZE,
    activation="tanh",
    obs_shape=jax_env.observation_space().shape,
)
hsp_model = ModelWrapper(hsp_model_inner)

model_dict = {
    'ik_finetune': ik_finetune_model,
    'sk_e3t': sk_e3t_model,
    'fcp': fcp_model,
    'my': my_model,
    'mep': mep_model,
    'hsp': hsp_model,
}
param_dict = {
    'ik_finetune': [],
    'sk_e3t': [],
    'fcp': [],
    'my': [],
    'mep': None,  # loaded differently (already stacked dict)
    'hsp': [],
}
SEED = 0
num_seed_dict = {
    'ik_finetune': SEED,
    'sk_e3t': SEED,
    'fcp': SEED,
    'my': SEED,
    'mep': SEED,
    'hsp': SEED,
}

load_path_ik = f"/home/tom.danino/crossEnvCooperation/ckpts/ippo/overcooked/forced_coord_9/ikFalse/reset_all/graphFalse"

with open(f"{load_path_ik}/seed{SEED}_ckpt0_improved_fine_tune.pkl", "rb") as f:
    previous_ckpt = pickle.load(f)
    model_params = previous_ckpt['params']
    # Print all layer shapes to understand training architecture
    def print_shapes(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                print_shapes(v, f"{prefix}/{k}")
        else:
            try:
                print(f"[SHAPE] {prefix}: {d.shape}")
            except:
                pass
    print("[SHAPE DUMP] ik_finetune params:")
    print_shapes(model_params)
    param_dict["ik_finetune"].append(model_params)
    num_seed_dict["ik_finetune"] += 1
param_dict["ik_finetune"] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict["ik_finetune"])

load_path_e3t = f"/home/tom.danino/crossEnvCooperation/ckpts/ippo/overcooked/forced_coord_9/ikFalse/reset_all/graphFalse"

with open(f"{load_path_e3t}/seed{SEED}_ckpt0_e3t.pkl", "rb") as f:
    previous_ckpt = pickle.load(f)
    model_params = previous_ckpt['params']
    param_dict["sk_e3t"].append(model_params)
    num_seed_dict["sk_e3t"] += 1
param_dict["sk_e3t"] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict["sk_e3t"])

# Load fcp checkpoint
# Path: /home/tom.danino/zero_shot_jax/baselines/phase2/fcp/{MY_LAYOUT}/{SEED}/phase2_ego.pkl
load_path_fcp = f"/home/tom.danino/zero_shot_jax/baselines/phase2/fcp/{MY_LAYOUT}"
try:
    with open(f"{load_path_fcp}/{SEED}/phase2_ego.pkl", "rb") as f:
        previous_ckpt = pickle.load(f)
        model_params = previous_ckpt['params']
        model_params = jax.tree_map(lambda x: x.squeeze(0) if x.shape[0] == 1 else x, model_params)
        param_dict["fcp"].append(model_params)
        num_seed_dict["fcp"] += 1
    param_dict["fcp"] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict["fcp"])
    print(f"[forced_coord_experiment] Loaded fcp checkpoint from {load_path_fcp}")
except Exception as _e:
    print(f"[forced_coord_experiment] WARNING: Could not load fcp checkpoint: {_e}")
    param_dict["fcp"] = None

# Load my_model checkpoint
# Path: /home/tom.danino/zero_shot_jax/phase2/{MY_LAYOUT}/{SEED}/{MY_ACHIEVER}/MOE_{MY_MOE}/phase2_ego.pkl
load_path_my = f"/home/tom.danino/zero_shot_jax/phase2/{MY_LAYOUT}"
try:
    with open(f"{load_path_my}/{SEED}/standrard/MOE_1/phase2_ego.pkl", "rb") as f:
        previous_ckpt = pickle.load(f)
        model_params = previous_ckpt['params']
        model_params = jax.tree_map(lambda x: x.squeeze(0) if x.shape[0] == 1 else x, model_params)
        param_dict["my"].append(model_params)
        num_seed_dict["my"] += 1
    param_dict["my"] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict["my"])
    print(f"[forced_coord_experiment] Loaded my checkpoint from {load_path_my}")
except Exception as _e:
    print(f"[forced_coord_experiment] WARNING: Could not load my checkpoint: {_e}")
    param_dict["my"] = None

# Load MEP checkpoint (ActorCriticRNN3, same as fcp)
# Path: /home/tom.danino/zero_shot_jax/baselines/phase2/mep/{MY_LAYOUT}/{SEED}/ego.pkl
# Note: key is 'ego_params'['params'], with safe_squeeze for leading batch dim
load_path_mep = f"/home/tom.danino/zero_shot_jax/baselines/phase2/mep/{MY_LAYOUT}"
try:
    with open(f"{load_path_mep}/{SEED}/ego.pkl", "rb") as f:
        previous_ckpt = pickle.load(f)
        model_params = previous_ckpt['ego_params']['params']

        def safe_squeeze(x):
            if not hasattr(x, 'shape') or x.ndim < 2:
                return x          # leave scalars and 1-D params (e.g. Dense_N bias) alone
            if x.shape[0] == 1:
                return x.squeeze(0)  # remove spurious leading batch dim
            return x

        model_params = jax.tree_map(safe_squeeze, model_params)
        # Stack single seed (nicewebrl expects stacked params even for 1 seed)
        stacked = jax.tree_map(lambda x: jnp.stack([x]), model_params)
        param_dict["mep"] = {'params': stacked}
        num_seed_dict["mep"] = 1
    print(f"[forced_coord_experiment] Loaded mep checkpoint from {load_path_mep}")
except Exception as _e:
    print(f"[forced_coord_experiment] WARNING: Could not load mep checkpoint: {_e}")
    param_dict["mep"] = None

# Load HSP checkpoint (ActorCriticRNN3, same architecture as fcp/mep)
# Path: /home/tom.danino/zero_shot_jax/baselines/phase2/hsp/{MY_LAYOUT}/{SEED}/phase2_ego.pkl
load_path_hsp = f"/home/tom.danino/zero_shot_jax/baselines/phase2/hsp/{MY_LAYOUT}"
try:
    with open(f"{load_path_hsp}/{SEED}/phase2_ego.pkl", "rb") as f:
        previous_ckpt = pickle.load(f)
        model_params = previous_ckpt['params']
        model_params = jax.tree_map(lambda x: x.squeeze(0) if x.shape[0] == 1 else x, model_params)
        param_dict["hsp"].append(model_params)
        num_seed_dict["hsp"] += 1
    param_dict["hsp"] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict["hsp"])
    print(f"[forced_coord_experiment] Loaded hsp checkpoint from {load_path_hsp}")
except Exception as _e:
    print(f"[forced_coord_experiment] WARNING: Could not load hsp checkpoint: {_e}")
    param_dict["hsp"] = None

# init_hidden_state_fn is defined per-stage below to capture the correct num_seeds,
# since each stage needs hidden_state shape (1, num_seeds, *carry_dims).
def make_hidden_state_fn(num_seeds, hidden_dim=None):
    # num_seeds is ignored — the carry shape is independent of num_seeds.
    # hidden_dim: pass the model's config GRU_HIDDEN_DIM explicitly.
    #   e.g. make_hidden_state_fn(n, hidden_dim=no_graph_config['GRU_HIDDEN_DIM'])  # 128
    #        make_hidden_state_fn(n, hidden_dim=base_config['GRU_HIDDEN_DIM'])      # 256
    if hidden_dim is None:
        hidden_dim = no_graph_config['GRU_HIDDEN_DIM']
    def _init():
        return ScannedRNN.initialize_carry(1, hidden_dim)
    return _init

jax_env = TimestepWrapper(jax_env, autoreset=True, reset_w_batch_dim=False, use_params=False)
jax_env_tutorial = TimestepWrapper(jax_env_tutorial, autoreset=True, reset_w_batch_dim=False, use_params=False)

jax_web_env = MultiAgentJaxWebEnv(
    env=jax_env,
    actions=action_array)
jax_web_env_tutorial = MultiAgentJaxWebEnv(
    env=jax_env_tutorial,
    actions=action_array)

jax_web_env.precompile(dummy_env_params=default_params)
jax_web_env_tutorial.precompile(dummy_env_params=default_params)

def render_fn(timestep: nicewebrl.Timestep):
    image = overcooked_render_fn(timestep.state)
    return image.astype(jnp.uint8)

def render_fn_tutorial(timestep: nicewebrl.Timestep):
    image = overcooked_render_fn(timestep.state)
    return image.astype(jnp.uint8)

vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, default_params)
vmap_render_fn_tutorial = jax_web_env_tutorial.precompile_vmap_render_fn(
    render_fn_tutorial, default_params)

render_fn = jax.jit(render_fn).lower(
    jax_web_env.reset(jax.random.PRNGKey(0), default_params)).compile()
render_fn_tutorial = jax.jit(render_fn_tutorial).lower(
    jax_web_env_tutorial.reset(jax.random.PRNGKey(0), default_params)).compile()


async def user_survey_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style('align-items: center;'):
        ui.markdown("## User Survey")

        ui.markdown("Please enter your Prolific ID below.")
        prolific_id = ui.input(placeholder="Your Prolific ID")

        ui.markdown("Please answer the following questions about your experience.")

        questions = [
            "The agent adapted to me when making decisions.",
            "The agent was consistent in its actions.",
            "The agent's actions were human-like.",
            "The agent frequently got in my way.",
            "The agent's behavior was frustrating.",
            "Overall, I enjoyed playing with the agent.",
            "Overall, I felt that the agent's ability to coordinate with me was:"
        ]

        responses = {"prolific_id": prolific_id}
        completed = {}
        completed_all = asyncio.Event()

        def create_on_change(q_idx):
            def on_change(val):
                completed[q_idx] = True
                if len(completed) == len(questions):
                    completed_all.set()
            return on_change

        for i, question in enumerate(questions):
            ui.markdown(question)
            options = {
                'Strongly disagree': 'Strongly disagree',
                'Disagree': 'Disagree',
                'Neutral': 'Neutral',
                'Agree': 'Agree',
                'Strongly agree': 'Strongly agree'
            } if i < len(questions) - 1 else {
                'Very poor': 'Very poor',
                'Poor': 'Poor',
                'Neutral': 'Neutral',
                'Good': 'Good',
                'Very good': 'Very good'
            }
            dropdown = ui.select(options, on_change=create_on_change(i))
            responses[question] = dropdown

        ui.markdown(f"{stage.body}")

        await completed_all.wait()
        return {k: v.value for k, v in responses.items()}

def make_survey_stage(name='User Survey'):
    stage = FeedbackStage(
        name=name,
        body="",
        display_fn=user_survey_display_fn,
        user_save_file_fn=get_user_save_file_fn,
        next_button=True
    )
    return stage


########################################
# Define Stages of experiment
########################################
all_stages = []
all_blocks = []

async def instruction_display_fn(stage, container):
    with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown("You'll be playing a game of Overcooked with an agent. The agent will be trying to help you complete tasks.")
        ui.markdown("You'll be playing as the human, and the agent will be playing as the other player.")
        ui.markdown("You'll be given a task to complete, and the agent will be trying to help you complete it.")
        ui.markdown("Use your arrow keys to move up, down, left, and right.")
        ui.markdown("Press the space bar to interact with the environment.")
        ui.markdown("Press the s key to stay in place.")

async def tutorial_display_fn(stage, container):
    with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        ui.markdown("You will now play a tutorial stage so you can get used to the controls.")
        ui.markdown("Please do not close or leave this page until the experiment is complete, as you will not be able to return.")

async def post_tutorial_display_fn(stage, container):
    with container.style('align-items: center;'):
        ui.markdown(f"## {stage.name}")
        ui.markdown("Now that you've seen how to play the game, the actual experiment will begin.")


env_params = default_params

def make_image_html(src):
    html = f'''
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="max-width: 100%; height: auto; display: block;">
    </div>
    '''
    return html

async def env_stage_display_fn(
        stage: MultiAgentEnvStage,
        container: ui.element,
        timestep: nicewebrl.Timestep):

    stage_state = stage.get_user_data('stage_state')
    human_color = stage.get_user_data('human_color') or 'blue'

    from nicegui import context as _ctx
    try:
        client_id = _ctx.client.id
    except Exception as e:
        client_id = f"ERROR:{e}"

    # Use the latest registered container for this user (may be a newer client).
    # _active_containers is populated by web_app.index() on every page load.
    seed = app.storage.user.get('seed')
    latest = _active_containers.get(seed)
    if latest is not None and latest is not container:
        logger.info(f"env_stage_display_fn: switching to latest container for seed {seed}")
        container = latest

    logger.info(f"env_stage_display_fn called for '{stage.name}', stage_state={stage_state is not None}, human_color={human_color}, client_id={client_id}")

    state_image = stage.render_fn(timestep)
    state_image = base64_npimage(state_image)

    logger.info(f"env_stage_display_fn: rendered image, length={len(state_image)}, container={container}")

    # NiceGUI's ui.html() sends element updates via Vue.js/Socket.IO to the
    # container's owning client session. On reconnect, this session is stale
    # and the browser never sees the update. ui.run_javascript() however
    # always targets the CURRENT active session. So we use JS to inject
    # the image directly into the DOM, bypassing NiceGUI's element system.
    # nicewebrl's basics.js already updates #stateImage on key presses via
    # window.next_states, so this initial injection is all we need.
    if stage_state is not None:
        nepisodes = int(stage_state.nepisodes)
        label_text = f"Try: {nepisodes}/{stage.max_episodes}. You control the {human_color} agent."
    else:
        label_text = f"You control the {human_color} agent."

    # Escape the base64 src for safe JS embedding (base64 chars are safe)
    js_src = state_image.replace("'", "\'")
    try:
        await ui.run_javascript(f"""
            (function() {{
                // Remove old image if present
                var old = document.getElementById('stateImageContainer');
                if (old) old.parentNode.removeChild(old);
                // Build new container + image
                var wrap = document.createElement('div');
                wrap.id = 'stateImageContainer';
                wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;';
                var lbl = document.createElement('div');
                lbl.style.cssText = 'padding:4px 8px;background:#d1fae5;border-radius:4px;margin-bottom:8px;';
                lbl.textContent = '{label_text}';
                var img = document.createElement('img');
                img.id = 'stateImage';
                img.src = '{js_src}';
                img.style.cssText = 'max-width:100%;height:auto;display:block;';
                wrap.appendChild(lbl);
                wrap.appendChild(img);
                // Append to the Quasar card (NiceGUI uses Quasar)
                var card = document.querySelector('.q-card') ||
                           document.querySelector('.nicegui-card') ||
                           document.body;
                card.appendChild(wrap);
            }})();
        """, timeout=5.0)
        logger.info(f"Image injected via JavaScript for stage '{stage.name}'")
    except Exception as _je:
        logger.warning(f"JS image injection failed: {_je}")
        # Fallback: try NiceGUI element system
        nicewebrl.clear_element(container)
        with container:
            ui.label(label_text)
            ui.html(make_image_html(src=state_image))
    logger.info(f"env_stage_display_fn: done rendering")


def evaluate_success_fn(timestep: nicewebrl.Timestep, env_params: struct.PyTreeNode):
    """Episode finishes if person every gets 1 achievement"""
    success = int(timestep.state.terminal)
    return success

async def transition_display_fn(stage, container):
    with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown("After completing the survey, please click the button below to continue.")


instruction_stage = Stage(
    name="Instuctions2",
    display_fn=instruction_display_fn)
tutorial_stage = Stage(
    name="Tutorial2",
    display_fn=tutorial_display_fn)
# Verify model config right before stage creation
print(f"[DEBUG] Before tutorial_env_stage: ik_finetune_model.config FC_DIM_SIZE={ik_finetune_model.config['FC_DIM_SIZE']}")
print(f"[DEBUG] id(ik_finetune_model.config)={id(ik_finetune_model.config)}, id(no_graph_config)={id(no_graph_config)}")
print(f"[DEBUG] no_graph_config FC_DIM_SIZE={no_graph_config['FC_DIM_SIZE']}")
tutorial_env_stage = MultiAgentEnvStage(
    name=f"tutorial2",
    web_env=jax_web_env_tutorial,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn_tutorial,
    vmap_render_fn=vmap_render_fn_tutorial,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    notify_success=False,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    user_save_file_fn=get_user_save_file_fn,
    metadata=dict(
        desc="some description",
        key1="value1",
        key2="value2",
    ),
    model=ik_finetune_model,
    model_params=param_dict['ik_finetune'],
    num_seeds=num_seed_dict['ik_finetune'],
    using_param_stack=True,
    init_hidden_state_fn=make_hidden_state_fn(num_seed_dict['ik_finetune'], hidden_dim=no_graph_config['GRU_HIDDEN_DIM']),
    max_timesteps=TUTORIAL_EPISODE_TIMESTEPS,
    human_id=None,
)

post_tutorial_stage = Stage(
    name="Post-Tutorial",
    display_fn=post_tutorial_display_fn)

all_stages.append(instruction_stage)
all_stages.append(tutorial_stage)
all_stages.append(tutorial_env_stage)
all_stages.append(post_tutorial_stage)
instruction_block = Block(stages=[
    instruction_stage,
    tutorial_stage,
    tutorial_env_stage,
    post_tutorial_stage,
], metadata=dict(desc="Instructions"), randomize=False)
all_blocks.append(instruction_block)


model_names = model_dict.keys()
# Map each model name to the hidden_dim its RNN uses
_model_hidden_dims = {
    'ik_finetune': no_graph_config['GRU_HIDDEN_DIM'],  # 128
    'sk_e3t':      no_graph_config['GRU_HIDDEN_DIM'],  # 128
    'fcp':         MY_HIDDEN_SIZE,                      # 120
    'my':          MY_HIDDEN_SIZE,                      # 120
    'mep':         MY_HIDDEN_SIZE,                      # 120
    'hsp':         MY_HIDDEN_SIZE,                      # 120
}

for model_name, model in model_dict.items():
    # Skip models whose checkpoints failed to load
    if param_dict[model_name] is None:
        print(f"[forced_coord_experiment] Skipping stage for '{model_name}' (no params loaded)")
        continue

    hidden_dim = _model_hidden_dims.get(model_name, no_graph_config['GRU_HIDDEN_DIM'])
    environment_stage = MultiAgentEnvStage(
        name=f"{model_name}_forced_coord2",
        web_env=jax_web_env,
        action_keys=action_keys,
        action_to_name=action_to_name,
        env_params=env_params,
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        notify_success=False,
        min_success=MIN_SUCCESS_EPISODES,
        max_episodes=MAX_STAGE_EPISODES,
        verbosity=VERBOSITY,
        user_save_file_fn=get_user_save_file_fn,
        metadata=dict(
            desc="some description",
            key1="value1",
            key2="value2",
        ),
        model=model,
        model_params=param_dict[model_name],
        num_seeds=num_seed_dict[model_name],
        using_param_stack=True,
        init_hidden_state_fn=make_hidden_state_fn(num_seed_dict[model_name], hidden_dim=hidden_dim),
        max_timesteps=MAX_EPISODE_TIMESTEPS,
        human_id=None,
    )

    transition_stage = Stage(
        name="Post-Survey",
        display_fn=transition_display_fn,
    )
    survey_stage = make_survey_stage(f'{model_name} Forced Coord Survey')

    env_block = Block(stages=[
        environment_stage,
        # transition_stage,
    ], metadata=dict(desc=f"{model_name} Environment"), randomize=False)
    all_blocks.append(env_block)


all_stages = prepare_blocks(all_blocks)

def generate_random_stage_order(seed, all_blocks):
    rng_key = jax.random.PRNGKey(seed)
    block_ids = jnp.arange(len(all_blocks))
    first_block_id = block_ids[0]
    valid_ids = block_ids[1:]
    # Use jax.random.permutation (jax.random.shuffle is deprecated and
    # silently returned unshuffled data in newer JAX versions)
    valid_ids = jax.random.permutation(rng_key, valid_ids)
    block_order = [first_block_id, *valid_ids]
    block_order = [int(b) for b in block_order]
    rng_key, subkey = jax.random.split(rng_key)
    stage_order = generate_stage_order(all_blocks, block_order, subkey)
    return stage_order
