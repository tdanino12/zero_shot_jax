import aiofiles
import msgpack
import os.path
import asyncio
import time
from asyncio import Lock
from datetime import datetime, timedelta
from typing import Callable, Awaitable
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator

# from gcs import save_to_gcs_with_retries
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress, write_msgpack_record
from nicewebrl import stages
import pdb
import sys

experiment_name = sys.argv[1]
if experiment_name == 'counter_circuit':
    import counter_circuit_experiment as experiment
elif experiment_name == 'cramped_room':
    import cramped_room_experiment as experiment
elif experiment_name == 'coord_ring':
    import coord_ring_experiment as experiment
elif experiment_name == 'forced_coord':
    import forced_coord_experiment as experiment
elif experiment_name == 'asymm_advantages':
    import asymm_advantages_experiment as experiment
else:
    print(f"Invalid experiment name: {experiment_name!r}")
    print("Valid options: counter_circuit, cramped_room, coord_ring, forced_coord, asymm_advantages")
    print("Usage: python web_app.py <layout> [port]")
    sys.exit(1)

# ── Port (optional second CLI argument, default 8080) ────────────────────────
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
DATA_DIR = os.environ.get('DATA_DIR', f'data_{PORT}')

DEBUG = int(os.environ.get('DEBUG', 0))
DEBUG_SEED = int(os.environ.get('SEED', 0))

# Default NAME includes the port so two instances are automatically isolated
# (different DB file, different data dir) without any extra env vars.
NAME = os.environ.get('NAME', f'{experiment_name}_{PORT}')

DATABASE_FILE = f'{DATABASE_FILE}_name={NAME}_debug={DEBUG}'

os.makedirs(DATA_DIR, exist_ok=True)

_user_locks = {}
_user_timers = {}          # track timers per user so we can cancel stale ones on re-login
_user_session_start = {}   # track session start time per user so we control the clock
_user_cancel_events = {}   # asyncio.Event per user seed; set on re-login to kill old coroutine  
_user_experiment_tasks = {}  # running start_experiment Task per seed
_user_browser_ids = {}  # browser ID per seed, to detect tab changes vs reconnects
_user_active_containers = {}  # seed -> stage_container (always the latest client's container)
_fresh_start_seeds = set() # seeds that need fresh stage state (cleared on re-login)


#####################################
# Setup logger
#####################################
def log_filename_fn(log_dir, user_id):
    return os.path.join(log_dir, f'log_{user_id}.log')


setup_logging(
    DATA_DIR,
    log_filename_fn=log_filename_fn,
    nicegui_storage_user_key='seed')
logger = get_logger('main')

# Monkeypatch nicewebrl's get_latest_stage_state so that users in
# _fresh_start_seeds always get None (fresh state) instead of loading
# from the DB. This is the only reliable way to prevent "Loading from memory"
# without knowing nicewebrl's internal DB table/field names.
def _patch_nicewebrl_stage_memory():
    from nicewebrl import stages as _nwr_stages
    from nicegui import app as _app

    # Patch activate on ALL stage classes that define it in their own __dict__
    patched = []
    for cls_name in ('Stage', 'EnvStage', 'MultiAgentEnvStage', 'FeedbackStage', 'LLMEnvStage'):
        cls = getattr(_nwr_stages, cls_name, None)
        if cls is None or 'activate' not in cls.__dict__:
            continue
        orig = cls.__dict__['activate']
        def make_patched(orig, cname):
            async def _p(self, container):
                seed = _app.storage.user.get('seed')
                if seed in _fresh_start_seeds:
                    # Only delete the DB row here — do NOT clear user_data.
                    # MultiAgentEnvStage.activate sets human_id/hidden_state BEFORE
                    # calling super().activate() which hits our EnvStage patch.
                    # Clearing user_data in the EnvStage patch would wipe those values.
                    # The DB delete is idempotent so running it twice is fine.
                    try:
                        browser_id = _app.storage.browser.get('id')
                        from tortoise import connections
                        conn = connections.get('default')
                        result = await conn.execute_query(
                            'DELETE FROM stage WHERE session_id=? AND name=?',
                            [str(browser_id), str(self.name)]
                        )
                        logger.info(f"Monkeypatch({cname}): deleted DB rows for '{self.name}' browser={browser_id}: {result}")
                    except Exception as _de:
                        logger.warning(f"Monkeypatch({cname}) DB delete failed: {_de}")
                return await orig(self, container)
            return _p
        cls.activate = make_patched(orig, cls_name)
        patched.append(cls_name)
    return patched

try:
    patched = _patch_nicewebrl_stage_memory()
    logger.info(f"Successfully patched activate on: {patched}")
except Exception as _e:
    logger.warning(f"Could not patch Stage.activate: {_e}")



#####################################
# Helper functions
#####################################
def stage_name(stage):
    return stage.name


def get_user_lock():
    """A function that returns a lock for the current user using their unique seed"""
    user_seed = app.storage.user['seed']
    if user_seed not in _user_locks:
        _user_locks[user_seed] = Lock()
    return _user_locks[user_seed]


async def experiment_not_finished():
    """Check if the experiment is not finished"""
    async with get_user_lock():
        not_finished = not app.storage.user.get('experiment_finished', False)
        not_finished &= app.storage.user['stage_idx'] < len(experiment.all_stages)
    return not_finished


def blob_user_filename():
    """filename structure for user data in GCS (cloud)"""
    seed = app.storage.user['seed']
    worker = app.storage.user.get('worker_id', None)
    if worker is not None:
        return f'user={seed}_worker={worker}_name={NAME}_debug={DEBUG}'
    else:
        return f'user={seed}_name={NAME}_debug={DEBUG}'


async def global_handle_key_press(e, container):
    """Define global key press handler"""
    stage_idx = app.storage.user['stage_idx']
    if app.storage.user['stage_idx'] >= len(experiment.all_stages):
        return

    stage = experiment.all_stages[app.storage.user['stage_order'][stage_idx]]
    if stage.get_user_data('finished', False):
        return

    await stage.handle_key_press(e, container)
    local_handle_key_press = stage.get_user_data('local_handle_key_press')
    if local_handle_key_press is not None:
        await local_handle_key_press()


async def save_data(final_save=True, feedback=None, **kwargs):
    user_data_file = experiment.get_user_save_file_fn()

    if final_save:
        user_storage = nicewebrl.make_serializable(dict(app.storage.user))
        last_line = dict(
            finished=True,
            feedback=feedback,
            user_storage=user_storage,
            **kwargs,
        )
        async with aiofiles.open(user_data_file, 'ab') as f:
            await write_msgpack_record(f, last_line)

    if not DEBUG:
        files_to_save = [
            (user_data_file, f'data/{blob_user_filename()}.json'),
            (log_filename_fn(DATA_DIR, app.storage.user.get('user_id')),
             f'logs/{blob_user_filename()}.log')
        ]
        # await save_to_gcs_with_retries(
        #    files_to_save,
        #    max_retries=5 if final_save else 1,
        # )


#####################################
# Setup database for storing experiment data
#####################################
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


async def init_db() -> None:
    await Tortoise.init(
        db_url=f'sqlite://{DATA_DIR}/{DATABASE_FILE}',
        modules={'models': ['models']})
    await Tortoise.generate_schemas()


async def close_db() -> None:
    await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)


#####################################
# Consent Form and demographic info
#####################################

async def make_consent_form(container):
    consent_given = asyncio.Event()
    with container:
        ui.markdown('## Consent Form')
        with open('consent.md', 'r') as consent_file:
            consent_text = consent_file.read()
        ui.markdown(consent_text)
        ui.checkbox('I agree to participate.',
                    on_change=lambda: consent_given.set())
    await consent_given.wait()


async def collect_demographic_info(container):
    nicewebrl.clear_element(container)
    with container:
        ui.markdown('## Demographic Info')
        ui.markdown('Please fill out the following information.')

        with ui.column():
            with ui.column():
                ui.label('Biological Sex')
                sex_input = ui.radio(
                    ['Male', 'Female'], value='Male').props('inline')
            age_input = ui.input('Age')
            ui.label('Participant Seed (provided by experimenter)')
            participant_seed_input = ui.input('Seed')
            ui.label('Layout')
            layout_input = ui.select(
                ['counter_circuit_9', 'coord_ring_9', 'asymm_advantages_9', 'forced_coord_9', 'cramped_room_9'],
                value='counter_circuit_9',
                label='Layout'
            )

        async def submit():
            age = age_input.value
            sex = sex_input.value
            pseed = participant_seed_input.value.strip()
            layout = layout_input.value
            if not age.isdigit() or not (0 < int(age) < 100):
                ui.notify(
                    "Please enter a valid age between 1 and 99.", type="warning")
                return
            if not pseed:
                ui.notify("Please enter your participant seed.", type="warning")
                return
            app.storage.user['age'] = int(age)
            app.storage.user['sex'] = sex
            app.storage.user['participant_seed'] = pseed
            app.storage.user['selected_layout'] = layout
            logger.info(f"age: {int(age)}, sex: {sex}, participant_seed: {pseed}, layout: {layout}")

        button = ui.button('Submit', on_click=submit)
        await button.clicked()


########################
# Run experiment
########################

async def start_experiment(meta_container, stage_container, button_container):

    #========================================
    # Consent form and demographic info
    #========================================
    # Clear stage_container first (it may have stale content from a previous session)
    nicewebrl.clear_element(stage_container)
    nicewebrl.clear_element(button_container)

    if not (app.storage.user.get('experiment_started', False) or DEBUG):
        await make_consent_form(stage_container)
        await collect_demographic_info(stage_container)
        app.storage.user['experiment_started'] = True

    # Always generate stage_order if missing, regardless of DEBUG or
    # experiment_started. Previously this was inside the consent block so
    # DEBUG mode would never set it, causing a silent KeyError that swallowed
    # the entire experiment loop.
    if 'stage_order' not in app.storage.user:
        app.storage.user['stage_order'] = experiment.generate_random_stage_order(
            app.storage.user['seed'], experiment.all_blocks)
    print(app.storage.user['stage_order'])

    ui.run_javascript('window.require_fullscreen = false')
    ui.on('key_pressed', lambda e: global_handle_key_press(e, meta_container))

    # Wait for the client JS to be ready before proceeding.
    # On first login the consent/demographic forms provide this delay naturally.
    # On re-login we jump straight here, so we must wait explicitly.
    logger.info("Waiting for client to be ready...")
    for _ in range(100):  # wait up to 10 seconds
        try:
            ready = await ui.run_javascript("return window.niceGuiReady || false;", timeout=5.0)
            if ready:
                break
        except Exception:
            pass
        await asyncio.sleep(0.1)
    logger.info("Client ready, starting experiment")

    logger.info("Starting experiment")
    my_token = app.storage.user.get('session_token')
    user_seed = app.storage.user['seed']
    def is_stale():
        """Return True if a genuine re-login reset our stage progress."""
        # Only consider stale if stage_idx was reset to 0 (genuine re-login/reset)
        # while we were past stage 0. Token changes happen on every reconnect
        # so we can't use them directly.
        return app.storage.user.get('session_token') != my_token and app.storage.user.get('stage_idx', 0) < stage_idx

    while True and await experiment_not_finished():
        if is_stale():
            logger.info("Stale coroutine detected, exiting start_experiment")
            return
        stage_idx = app.storage.user['stage_idx']
        # Re-generate stage_order if wiped (e.g. by concurrent initialize_user call)
        if 'stage_order' not in app.storage.user:
            app.storage.user['stage_order'] = experiment.generate_random_stage_order(
                app.storage.user['seed'], experiment.all_blocks)
        stage = experiment.all_stages[app.storage.user['stage_order'][stage_idx]]
        print(f"Beginning stage {stage.name}")

        logger.info("=" * 30)
        logger.info(f"Began stage '{stage.name}'")
        await run_stage(stage, stage_container, button_container)
        logger.info(f"Finished stage '{stage.name}'")
        ui.notify("Loading next page...", type='info', position='top')

        if isinstance(stage, stages.EnvStage):
            await stage.finish_saving_user_data()
            logger.info(f"Saved data for stage '{stage.name}'")

        async with get_user_lock():
            app.storage.user['stage_idx'] = stage_idx + 1

        if app.storage.user['stage_idx'] >= len(experiment.all_stages):
            break

    await finish_experiment(meta_container, stage_container, button_container)


async def finish_experiment(meta_container, stage_container, button_container):
    # Do NOT clear meta_container — that would detach stage_container/button_container
    # from the DOM, making them invisible orphans on the next login.
    nicewebrl.clear_element(stage_container)
    nicewebrl.clear_element(button_container)
    logger.info("Finishing experiment")
    experiment_finished = app.storage.user.get('experiment_finished', False)

    if experiment_finished and not DEBUG:
        return

    # Set immediately to block duplicate calls from the timer before async
    # operations complete
    app.storage.user['experiment_finished'] = True
    _fresh_start_seeds.discard(app.storage.user.get('seed'))  # allow DB restore if they return

    async def submit(feedback):
        with stage_container:
            nicewebrl.clear_element(stage_container)
            ui.markdown("## Saving data. Please wait")
            ui.markdown(
                "**Once the data is uploaded, this app will automatically move to the next screen**")
        await save_data(final_save=True, feedback=feedback)
        app.storage.user['data_saved'] = True
        print("data saved")

    app.storage.user['data_saved'] = app.storage.user.get('data_saved', False)
    if not app.storage.user['data_saved']:
        with stage_container:
            nicewebrl.clear_element(stage_container)
            ui.markdown("Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment.")
            text = ui.textarea().style('width: 80%;')
            button = ui.button("Submit")
            await button.clicked()
            await submit(text.value)

    with stage_container:
        nicewebrl.clear_element(stage_container)
        ui.markdown("# Experiment over")
        ui.markdown("## Data saved")
        ui.markdown(
            "### Please record the following code which you will need to provide for compensation")
        ui.markdown('### socialrl.cook')
        ui.markdown("#### You may close the browser")


async def run_stage(stage, stage_container, button_container):
    stage_over_event = asyncio.Event()

    async def local_handle_key_press():
        async with get_user_lock():
            if stage.get_user_data('finished', False):
                logger.info(f"Finished {stage_name(stage)} via key press")
                ui.notify("Loading next page...", type='info', position='top')
                stage_over_event.set()

    async def handle_button_press():
        if stage.get_user_data('finished', False):
            return
        await stage.handle_button_press(stage_container)
        async with get_user_lock():
            if stage.get_user_data('finished', False):
                logger.info(f"Finished {stage_name(stage)} via button press")
                ui.notify("Loading next page...", type='info', position='top')
                stage_over_event.set()

    nicewebrl.clear_element(stage_container)
    await stage.activate(stage_container)

    if stage.get_user_data('finished', False):
        logger.info(f"Finished {stage_name(stage)} immediately after activation")
        ui.notify("Loading next page...", type='info', position='top')
        stage_over_event.set()

    await stage.set_user_data(local_handle_key_press=local_handle_key_press)

    with button_container.style('align-items: center;'):
        nicewebrl.clear_element(button_container)
        checking_fullscreen = False
        next_button_container = ui.row()

        async def create_button_and_wait():
            with next_button_container:
                nicewebrl.clear_element(next_button_container)
                button = ui.button('Next page').bind_visibility_from(
                    stage, 'next_button')
                await wait_for_button_or_keypress(button)
                logger.info("Button or key pressed")
                await handle_button_press()

        if stage.next_button:
            if checking_fullscreen:
                await create_button_and_wait()
                while not await nicewebrl.utils.check_fullscreen():
                    if await stage_over_event.wait():
                        break
                    logger.info("Waiting for fullscreen")
                    await asyncio.sleep(0.1)
                    await create_button_and_wait()
            else:
                await create_button_and_wait()

    await stage_over_event.wait()
    nicewebrl.clear_element(button_container)


#####################################
# Root page
#####################################

def initialize_user(request: Request):
    nicewebrl.initialize_user(seed=DEBUG_SEED)
    app.storage.user['worker_id'] = request.query_params.get('workerId', None)
    app.storage.user['hit_id'] = request.query_params.get('hitId', None)
    app.storage.user['assignment_id'] = request.query_params.get('assignmentId', None)
    app.storage.user['user_id'] = app.storage.user['seed']
    user_seed = app.storage.user['seed']
    current_browser_id = app.storage.browser.get('id')

    # ?reset=1 forces a full reset regardless of stored cookie state
    if request.query_params.get('reset', '0') == '1':
        app.storage.user['stage_idx'] = 0
        app.storage.user['experiment_finished'] = False
        app.storage.user['experiment_started'] = False
        app.storage.user.pop('stage_order', None)
        logger.info(f"initialize_user: forced reset via URL for seed {user_seed}")

    # --- Determine what kind of visit this is ---
    stage_idx = app.storage.user.get('stage_idx', 0)
    experiment_finished = app.storage.user.get('experiment_finished', False)
    is_mid_experiment = (stage_idx > 0 and not experiment_finished)
    logger.info(f"initialize_user: seed={user_seed} browser={current_browser_id} stage={stage_idx} finished={experiment_finished} mid={is_mid_experiment}")

    # --- Always cancel old task so new client context is used ---
    old_task = _user_experiment_tasks.pop(user_seed, None)
    if old_task and not old_task.done():
        old_task.cancel()
        logger.info(f"initialize_user: cancelled old task for seed {user_seed}")

    # Cancel stale timer
    if user_seed in _user_timers:
        try:
            _user_timers[user_seed].cancel()
        except Exception:
            pass
        del _user_timers[user_seed]

    _user_session_start[user_seed] = time.time()
    _user_browser_ids[user_seed] = current_browser_id
    app.storage.user['_active_browser_id'] = current_browser_id
    app.storage.user['session_duration'] = 0
    app.storage.user['data_saved'] = False

    # Validate stored stage_order against current all_stages length.
    # If the experiment was updated (stages added/removed), the old cookie
    # stage_order will have wrong indices → IndexError. Force regeneration.
    stored_order = app.storage.user.get('stage_order')
    n_stages = len(experiment.all_stages)
    if stored_order is not None:
        if len(stored_order) != n_stages or max(stored_order) >= n_stages:
            logger.info(f"initialize_user: stage_order mismatch (len={len(stored_order)} vs {n_stages}), forcing fresh start")
            app.storage.user.pop('stage_order', None)
            app.storage.user['stage_idx'] = 0
            app.storage.user['experiment_finished'] = False
            app.storage.user['experiment_started'] = False
            is_mid_experiment = False

    if is_mid_experiment:
        # Reconnect mid-experiment: preserve stage progress but DELETE DB rows.
        # The DB may contain stale model params from a previous server run with
        # a different model config — always start fresh from the current model.
        logger.info(f"initialize_user: mid-experiment reconnect at stage {stage_idx}, deleting DB state")
        for stage in experiment.all_stages:
            if isinstance(getattr(stage, 'user_data', None), dict):
                stage.user_data.pop(user_seed, None)
        _fresh_start_seeds.add(user_seed)  # monkeypatch will delete DB rows
    else:
        # Fresh start: reset all progress and delete DB rows
        app.storage.user['stage_idx'] = 0
        app.storage.user['experiment_finished'] = False
        app.storage.user['experiment_started'] = False
        app.storage.user.pop('stage_order', None)
        for stage in experiment.all_stages:
            if isinstance(getattr(stage, 'user_data', None), dict):
                stage.user_data.pop(user_seed, None)
        _fresh_start_seeds.add(user_seed)
        logger.info(f"initialize_user: fresh start for seed {user_seed}")


async def check_if_over(*args, episode_limit=60, **kwargs):
    """If past time limit, finish experiment"""
    # Don't fire if experiment is already finished
    if app.storage.user.get('experiment_finished', False):
        return
    # Compute minutes from our own session start time, not nicewebrl which
    # returns cumulative server uptime and cannot be reset between user sessions
    user_seed = app.storage.user['seed']
    session_start = _user_session_start.get(user_seed, time.time())
    minutes_passed = (time.time() - session_start) / 60.0
    app.storage.user['session_duration'] = minutes_passed
    if minutes_passed > episode_limit:
        logger.info(f"experiment timed out after {minutes_passed} minutes")
        app.storage.user['stage_idx'] = len(experiment.all_stages)
        await finish_experiment(*args, **kwargs)


@ui.page('/')
async def index(request: Request):
    # Each call to index() creates a new NiceGUI client (new WebSocket).
    # If we start_experiment on every call, we get multiple coroutines
    # writing UI to different clients, only one of which the browser sees.
    # Solution: use app.storage.browser to ensure only ONE experiment runs
    # per browser tab at a time. Subsequent calls for the same browser just
    # show a "reconnecting" message and wait.
    from nicegui import context as _ctx
    this_client = _ctx.client

    initialize_user(request)

    ui.run_javascript(f'window.debug = {DEBUG}')

    def print_ping(e):
        logger.info(str(e.args))
    ui.on('ping', print_ping)

    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    card = ui.card(align_items=['center']).classes('fixed-center').style(
        'max-width: 90vw;'
        'max-height: 90vh;'
        'overflow: auto;'
        'display: flex;'
        'flex-direction: column;'
        'justify-content: flex-start;'
        'align-items: center;'
    )
    with card:
        episode_limit = 200
        meta_container = ui.column()
        with meta_container.style('align-items: center;'):
            stage_container = ui.column()
            button_container = ui.column()
            timer = ui.timer(
                interval=1,
                callback=lambda: check_if_over(
                    episode_limit=episode_limit,
                    meta_container=meta_container,
                    stage_container=stage_container,
                    button_container=button_container))
            # Store timer reference so we can cancel it on re-login
            _user_timers[app.storage.user['seed']] = timer
            footer_container = ui.row()
            footer(footer_container)

            user_seed = app.storage.user['seed']
            # Register newest containers — display_fn always uses these
            _user_active_containers[user_seed] = stage_container
            experiment._active_containers[user_seed] = stage_container
            # Track current task so initialize_user can cancel it on next login
            _user_experiment_tasks[user_seed] = asyncio.current_task()

            try:
                await start_experiment(meta_container, stage_container, button_container)
            except asyncio.CancelledError:
                logger.info("start_experiment was cancelled by new login")


def footer(footer_container):
    """Add user information and progress bar to the footer"""
    with footer_container:
        with ui.row():
            ui.label().bind_text_from(
                app.storage.user, 'seed',
                lambda v: f"user id: {v}.")
            ui.label()
            ui.label().bind_text_from(
                app.storage.user, 'stage_idx',
                lambda v: f"stage: {int(v) + 1}/{len(experiment.all_stages)}.")
            ui.label()
            ui.label().bind_text_from(
                app.storage.user, 'session_duration',
                lambda v: f"minutes passed: {int(v)}.")

        stage_progress = lambda: float(
            f"{(app.storage.user['stage_idx']+1)/len(experiment.all_stages):.2f}")
        ui.linear_progress(
            value=stage_progress()).bind_value_from(app.storage.user, 'stage_progress')
        ui.button(
            'Toggle fullscreen', icon='fullscreen',
            on_click=nicewebrl.utils.toggle_fullscreen).props('flat')


ui.run(
    storage_secret='private key to secure the browser session cookie',
    reload=False,  # hot reload is incompatible with multi-user experiment state
    title='Overcooked',
    show=False,
    port=PORT,
)
