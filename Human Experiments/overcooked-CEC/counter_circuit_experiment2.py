from dotenv import load_dotenv
from flax import struct
import jax
import jax.numpy as jnp
import os
import pickle

from jaxmarl.viz.overcooked_jitted_visualizer import render_fn as overcooked_render_fn
from jaxmarl.environments.overcooked import Overcooked, Actions, State
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_asymm_advantages_9x9, make_coord_ring_9x9, make_counter_circuit_9x9, make_forced_coord_9x9, make_cramped_room_9x9
import jaxmarl

from nicegui import ui, app
import nicewebrl
from nicewebrl import MultiAgentJaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, MultiAgentEnvStage, FeedbackStage, Block, prepare_blocks, generate_stage_order
from nicewebrl import get_logger
from actor_networks import ActorCriticRNN, ScannedRNN, ActorCriticE3T
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
NAME = os.environ.get('NAME', 'counter_circuit')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 201
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
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
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
base_config['ENV_NAME'] = 'overcooked'
base_config['ENV_KWARGS']['check_held_out'] = False
base_config['ENV_KWARGS']['shuffle_inv_and_pot'] = False
base_config['ENV_KWARGS']['random_reset'] = False
base_config['ENV_KWARGS']['random_reset_fn'] = 'reset_all'
base_config['ENV_KWARGS']['layout'] = 'counter_circuit_9'
base_config['ENV_KWARGS']['max_steps'] = MAX_EPISODE_TIMESTEPS - 1
base_config['GRAPH_NET'] = True

tutorial_config = pickle.loads(pickle.dumps(base_config))
tutorial_config['ENV_KWARGS']['layout'] = 'asymm_advantages_9'


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

model_dict = {
    'ik': base_agent_model,
    'ik_finetune': base_agent_model,
    'sk': base_agent_model,
    'sk_e3t': e3t_agent_model,
    'sk_fcp': base_agent_model,
    'coord_sk': base_agent_model,
    'coord_fcp': base_agent_model,
}
param_dict = {
    'ik': [],
    'ik_finetune': [],
    'sk': [],
    'sk_e3t': [],
    'sk_fcp': [],
    'coord_sk': [],
    'coord_fcp': []
}
num_seed_dict = {
    'ik': 0,
    'ik_finetune': 0,
    'sk': 0,
    'sk_e3t': 0,
    'sk_fcp': 0,
    'coord_sk': 0,
    'coord_fcp': 0
}

for model_name in model_dict.keys():
    if 'coord' in model_name:
        continue
    model_dir = f'models/{model_name}/counter_circuit/'
    files = os.listdir(model_dir)
    for file in files:
        with open(os.path.join(model_dir, file), 'rb') as f:
            params = pickle.load(f)['params']
            param_dict[model_name].append(params)
            num_seed_dict[model_name] += 1
    param_dict[model_name] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict[model_name])

for model_name in ['sk', 'sk_fcp']:
    model_dir = f'models/{model_name}/coord_ring/'
    dict_name = 'coord_sk' if model_name == 'sk' else 'coord_fcp'
    files = os.listdir(model_dir)
    for file in files:
        with open(os.path.join(model_dir, file), 'rb') as f:
            params = pickle.load(f)['params']
            param_dict[dict_name].append(params)
            num_seed_dict[dict_name] += 1
    param_dict[dict_name] = jax.tree_map(lambda *x: jnp.stack(x), *param_dict[dict_name])

# init_hidden_state_fn is defined per-stage below to capture the correct num_seeds,
# since each stage needs hidden_state shape (1, num_seeds, *carry_dims).
def make_hidden_state_fn(num_seeds):
    # ScannedRNN.initialize_carry returns a tuple carry.
    # The original call used batch_size=1 and worked on first runs.
    # num_seeds is ignored here — the carry shape is independent of num_seeds.
    # (model_params are already de-stacked per seed before model.apply)
    def _init():
        return ScannedRNN.initialize_carry(1, base_config['GRU_HIDDEN_DIM'])
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
    model=base_agent_model,
    model_params=param_dict['ik'],
    num_seeds=num_seed_dict['ik'],
    using_param_stack=True,
    init_hidden_state_fn=make_hidden_state_fn(num_seed_dict['ik']),
    max_timesteps=MAX_EPISODE_TIMESTEPS,
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
for model_name, model in model_dict.items():
    environment_stage = MultiAgentEnvStage(
        name=f"{model_name}_counter_circuit2",
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
        init_hidden_state_fn=make_hidden_state_fn(num_seed_dict[model_name]),
        max_timesteps=MAX_EPISODE_TIMESTEPS,
        human_id=None,
    )

    transition_stage = Stage(
        name="Post-Survey",
        display_fn=transition_display_fn,
    )
    survey_stage = make_survey_stage(f'{model_name} Counter Circuit Survey')

    env_block = Block(stages=[
        environment_stage,
        survey_stage,
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
