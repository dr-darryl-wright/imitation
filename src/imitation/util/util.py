"""Miscellaneous utility methods."""


import datetime
import functools
import itertools
import json
import os
import random
import shutil
import subprocess
import uuid
import warnings
from tempfile import mkdtemp
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import cv2
import gym
import numpy as np
import torch as th
from gym.wrappers import TimeLimit
from PIL import Image
from stable_baselines3.common import monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from imitation.data.types import AnyPath, Trajectory


def oric(x: np.ndarray) -> np.ndarray:
    """Optimal rounding under integer constraints.

    Given a vector of real numbers such that the sum is an integer, returns a vector
    of rounded integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original vectors for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A 1D vector of real numbers that sum to an integer.

    Returns:
        A 1D vector of rounded integers, preserving the sum.
    """
    rounded = np.floor(x)
    shortfall = x - rounded

    # The total shortfall should be *exactly* an integer, but we
    # round to account for numerical error.
    total_shortfall = np.round(shortfall.sum()).astype(int)
    indices = np.argsort(-shortfall)

    # Apportion the total shortfall to the elements in order of
    # decreasing shortfall.
    rounded[indices[:total_shortfall]] += 1
    return rounded.astype(int)


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"


def make_vec_env(
    env_name: str,
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
) -> VecEnv:
    """Makes a vectorized environment.

    Args:
        env_name: The Env's string id in Gym.
        rng: The random state to use to seed the environment.
        n_envs: The number of duplicate environments.
        parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
        log_dir: If specified, saves Monitor output to this directory.
        max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
            with this episode length. If not specified and `max_episode_steps`
            exists for this `env_name` in the Gym registry, uses the registry
            `max_episode_steps` for every TimeLimit wrapper (this automatic
            wrapper is the default behavior when calling `gym.make`). Otherwise
            the environments are passed into the VecEnv unwrapped.
        post_wrappers: If specified, iteratively wraps each environment with each
            of the wrappers specified in the sequence. The argument should be a Callable
            accepting two arguments, the Env to be wrapped and the environment index,
            and returning the wrapped Env.
        env_make_kwargs: The kwargs passed to `spec.make`.

    Returns:
        A VecEnv initialized with `n_envs` environments.
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    spec = gym.spec(env_name)
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i: int, this_seed: int) -> gym.Env:
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        env = spec.make(**env_make_kwargs)

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.seed(int(this_seed))

        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        elif spec.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=spec.max_episode_steps)

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = monitor.Monitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)

        return env

    env_seeds = make_seeds(rng, n_envs)
    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
    ]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)


@overload
def make_seeds(
    rng: np.random.Generator,
) -> int:
    ...


@overload
def make_seeds(rng: np.random.Generator, n: int) -> List[int]:
    ...


def make_seeds(
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> Union[Sequence[int], int]:
    """Generate n random seeds from a random state.

    Args:
        rng: The random state to use to generate seeds.
        n: The number of seeds to generate.

    Returns:
        A list of n random seeds.
    """
    seeds_arr = rng.integers(0, (1 << 31) - 1, (n if n is not None else 1,))
    seeds: List[int] = seeds_arr.tolist()
    if n is None:
        return seeds[0]
    else:
        return seeds


def docstring_parameter(*args, **kwargs):
    """Treats the docstring as a format string, substituting in the arguments."""

    def helper(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return helper


T = TypeVar("T")


def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from `iterable`.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    Args:
        iterable: The non-iterator iterable object to endlessly iterate over.

    Returns:
        An iterator that repeats the elements in `iterable` forever.

    Raises:
        ValueError: if iterable is an iterator -- that will be exhausted, so
            cannot be iterated over endlessly.
    """
    if iter(iterable) == iterable:
        raise ValueError("endless_iter needs a non-iterator Iterable.")

    _, iterable = get_first_iter_element(iterable)
    return itertools.chain.from_iterable(itertools.repeat(iterable))


def safe_to_tensor(array: Union[np.ndarray, th.Tensor], **kwargs) -> th.Tensor:
    """Converts a NumPy array to a PyTorch tensor.

    The data is copied in the case where the array is non-writable. Unfortunately if
    you just use `th.as_tensor` for this, an ugly warning is logged and there's
    undefined behavior if you try to write to the tensor.

    Args:
        array: The array to convert to a PyTorch tensor.
        kwargs: Additional keyword arguments to pass to `th.as_tensor`.

    Returns:
        A PyTorch tensor with the same content as `array`.
    """
    if isinstance(array, th.Tensor):
        return array

    if not array.flags.writeable:
        array = array.copy()

    return th.as_tensor(array, **kwargs)


@overload
def safe_to_numpy(obj: Union[np.ndarray, th.Tensor], warn: bool = False) -> np.ndarray:
    ...


@overload
def safe_to_numpy(obj: None, warn: bool = False) -> None:
    ...


def safe_to_numpy(
    obj: Optional[Union[np.ndarray, th.Tensor]],
    warn: bool = False,
) -> Optional[np.ndarray]:
    """Convert torch tensor to numpy.

    If the object is already a numpy array, return it as is.
    If the object is none, returns none.

    Args:
        obj: torch tensor object to convert to numpy array
        warn: if True, warn if the object is not already a numpy array. Useful for
            warning the user of a potential performance hit if a torch tensor is
            not the expected input type.

    Returns:
        Object converted to numpy array
    """
    if obj is None:
        # We ignore the type due to https://github.com/google/pytype/issues/445
        return None  # pytype: disable=bad-return-type
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        if warn:
            warnings.warn(
                "Converted tensor to numpy array, might affect performance. "
                "Make sure this is the intended behavior.",
            )
        return obj.detach().cpu().numpy()


def tensor_iter_norm(
    tensor_iter: Iterable[th.Tensor],
    ord: Union[int, float] = 2,  # noqa: A002
) -> th.Tensor:
    """Compute the norm of a big vector that is produced one tensor chunk at a time.

    Args:
        tensor_iter: an iterable that yields tensors.
        ord: order of the p-norm (can be any int or float except 0 and NaN).

    Returns:
        Norm of the concatenated tensors.

    Raises:
        ValueError: ord is 0 (unsupported).
    """
    if ord == 0:
        raise ValueError("This function cannot compute p-norms for p=0.")
    norms = []
    for tensor in tensor_iter:
        norms.append(th.norm(tensor.flatten(), p=ord))
    norm_tensor = th.as_tensor(norms)
    # Norm of the norms is equal to the norm of the concatenated tensor.
    # th.norm(norm_tensor) = sum(norm**ord for norm in norm_tensor)**(1/ord)
    # = sum(sum(x**ord for x in tensor) for tensor in tensor_iter)**(1/ord)
    # = sum(x**ord for x in tensor for tensor in tensor_iter)**(1/ord)
    # = th.norm(concatenated tensors)
    return th.norm(norm_tensor, p=ord)


class FFMPEGVideo(object):
    def __init__(self, name: str = None, shape: Tuple[int, int] = None):
        super(FFMPEGVideo, self).__init__()
        self._ffmpeg = shutil.which("ffmpeg")
        self._name = name
        self._workdir = mkdtemp(prefix="FFMPEGVideo.", dir=os.getcwd())
        self._framecounter = 0
        self._shape = shape

        assert self._ffmpeg is not None  # should add ffmpeg\bin directory to PATH

    def __len__(self):
        return self._framecounter

    def add_frame(self, frame):
        filename = os.path.join(
            self._workdir, "frame_{:08d}.png".format(self._framecounter)
        )

        if self._shape is None:
            self._shape = frame.shape[:2].T

        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
            img.save(filename)
        else:
            frame.save(filename)

        self._framecounter += 1

    def save(self, filename: str = None, fps: int = 30, keep_frame_images=False):
        if self._framecounter == 0:
            raise Exception("No frames stored.")

        if filename is None:
            assert self._name is not None
            filename = self._name

        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"

        print("running ffmpeg")
        ffmpeg = subprocess.run(
            [
                self._ffmpeg,
                "-y",  # force overwrite if output file exists
                "-framerate",
                "{}".format(fps),
                "-i",
                os.path.join(self._workdir, "frame_%08d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "17",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                filename,
            ]
        )

        if ffmpeg.returncode != 0:
            # TODO proper logging
            print("error")
            print(
                "Running the following command failed with return code {}:\n\t{}".format(
                    ffmpeg.returncode, " ".join(ffmpeg.args)
                )
            )
        elif not keep_frame_images:
            shutil.rmtree(self._workdir)

    def to_webm(self, filename: str = None, keep_mp4: bool = False):
        if filename is None:
            assert self._name is not None
            filename = self._name

        if not filename.lower().endswith(".mp4"):
            new_filename = filename + ".webm"
        else:
            new_filename = filename[:-4] + ".webm"

        ffmpeg = subprocess.run(
            [
                self._ffmpeg,
                "-i",
                filename,
                "-s",
                f"{self._shape[0]}x{self._shape[1]}",
                "-vcodec",
                "libvpx",
                "-acodec",
                "libvorbis",
                "-crf",
                "5",
                new_filename,
            ]
        )

        if ffmpeg.returncode != 0:
            # TODO proper logging
            print("error")
            print(
                "Running the following command failed with return code {}:\n\t{}".format(
                    ffmpeg.returncode, " ".join(ffmpeg.args)
                )
            )
        else:
            if not keep_mp4:
                os.remove(filename)


def weighted_sample_without_replacement(population, weights, k, rng=random):
    """From https://maxhalford.github.io/blog/weighted-sampling-without-replacement/"""
    v = [rng.random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]


AGENT_RESOLUTION = (128, 128)
ACTION_SIZE = (121, 8641)


CURSOR_FILE = os.path.join(os.getcwd(), "cursors", "mouse_cursor_white_16x16.png")

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}


# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MINEREC_ORIGINAL_HEIGHT_PX = 720
# Matches a number in the MineRL Java code
# search the code Java code for "constructMouseState"
# to find explanations
CAMERA_SCALER = 360.0 / 2400.0

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}


INV_KEYBOARD_BUTTON_MAPPING = {v: k for k, v in KEYBOARD_BUTTON_MAPPING.items()}


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


def env_action_to_json_action(env_action, prev_json_action=None):
    """
    Converts a MineRL action into a json action.
    Requires information from previous json action.
    Returns json_action
    """
    json_action = {}
    json_action["keyboard"] = {}
    json_action["keyboard"]["keys"] = []
    json_action["keyboard"]["newKeys"] = []
    json_action["mouse"] = {}
    json_action["mouse"]["buttons"] = []
    json_action["mouse"]["newButtons"] = []
    json_action["hotbar"] = 0
    json_action["tick"] = prev_json_action["tick"] + 1 if prev_json_action else 0

    # determine whether GUI is open
    e_pressed = (
        "key.keyboard.e" in prev_json_action["keyboard"]["keys"]
        if prev_json_action
        else False
    )
    prev_gui_open = prev_json_action["isGuiOpen"] if prev_json_action else False
    json_action["isGuiOpen"] = e_pressed != prev_gui_open
    gui_was_opened_or_closed = prev_gui_open != json_action["isGuiOpen"]

    # process keyboard actions
    for key, json_key in INV_KEYBOARD_BUTTON_MAPPING.items():
        if key in env_action and env_action[key] == 1:
            json_action["keyboard"]["keys"].append(
                json_key,
            )

            # track newly pressed keys
            if (
                prev_json_action
                and json_key not in prev_json_action["keyboard"]["keys"]
            ):
                json_action["keyboard"]["newKeys"].append(json_key)

            # update hotbar entry (note: we can't update based on mouse wheel)
            if key.startswith("hotbar"):
                json_action["hotbar"] = int(key.split(".")[1]) - 1

    if gui_was_opened_or_closed:
        # reset mouse coords
        json_action["mouse"]["x"] = 640.0
        json_action["mouse"]["y"] = 360.0
    else:
        # normal mouse coord update
        json_action["mouse"]["x"] = (
            prev_json_action["mouse"]["x"] + prev_json_action["mouse"]["dx"]
            if prev_json_action
            else 640.0
        )
        json_action["mouse"]["y"] = (
            prev_json_action["mouse"]["y"] + prev_json_action["mouse"]["dy"]
            if prev_json_action
            else 360.0
        )

    # make sure coordinates are not out of bounds when GUI is open
    if json_action["isGuiOpen"]:
        json_action["mouse"]["x"] = max(0.0, min(json_action["mouse"]["x"], 640.0))
        json_action["mouse"]["y"] = max(0.0, min(json_action["mouse"]["x"], 360.0))

    buttons = ["attack", "use", "pickItem"]
    for i, button in enumerate(buttons):
        if button in env_action and env_action[button] == 1:
            json_action["mouse"]["buttons"].append(i)
            # track newly pressed buttons
            if prev_json_action and i not in prev_json_action["mouse"]["buttons"]:
                json_action["mouse"]["newButtons"].append(i)

    return json_action


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y : y + ch, x : x + cw, :] = (
        image1[y : y + ch, x : x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha
    ).astype(np.uint8)


def get_num_lines(path: AnyPath):
    with open(path) as file:
        num_lines = len(file.readlines())
    return num_lines


def load_jsonl(json_path: AnyPath):
    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    return json_data


def process_frame(frame, action, cursor_image, cursor_alpha):
    if action["isGuiOpen"]:
        camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
        cursor_x = int(action["mouse"]["x"] * camera_scaling_factor)
        cursor_y = int(action["mouse"]["y"] * camera_scaling_factor)
        composite_images_with_alpha(
            frame, cursor_image, cursor_alpha, cursor_x, cursor_y
        )
    cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
    pov_frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
    agent_frame = resize_image(pov_frame, AGENT_RESOLUTION)
    return pov_frame, agent_frame


def load_trajectory(
    video_path: AnyPath,
    json_path: AnyPath,
    from_step: int,
    to_step: int,
    minerl_agent,
    pov_to_infos: bool = False,
) -> Trajectory:
    """
    Load trajectories from data files for the data loader.
    """

    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    video = cv2.VideoCapture(video_path)
    # Note: In some recordings, the game seems to start
    #       with attack always down from the beginning, which
    #       is stuck down until player actually presses attack
    attack_is_stuck = False
    # Scrollwheel is allowed way to change items, but this is
    # not captured by the recorder.
    # Work around this by keeping track of selected hotbar item
    # and updating "hotbar.#" actions when hotbar selection changes.
    last_hotbar = 0

    json_data = load_jsonl(json_path)

    # relevant containers for constructing trajectory
    traj_length = to_step - from_step

    obs = np.empty((traj_length + 1, *AGENT_RESOLUTION, 3))
    acts = np.empty((traj_length, 2))
    infos = np.empty((traj_length,), dtype=object) if pov_to_infos else None

    terminal = False
    if to_step == len(json_data):
        terminal = True

    step_counter = 0
    for i in range(len(json_data)):
        # do not procecss more video than needed
        if i >= to_step:
            break

        # Processing action
        step_data = json_data[i]

        if i == 0:
            # Check if attack will be stuck down
            if step_data["mouse"]["newButtons"] == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            # Check if we press attack down, then it might not be stuck
            if 0 in step_data["mouse"]["newButtons"]:
                attack_is_stuck = False
        # If still stuck, remove the action
        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [
                button for button in step_data["mouse"]["buttons"] if button != 0
            ]

        action, is_null_action = json_action_to_env_action(step_data)

        # Update hotbar selection
        current_hotbar = step_data["hotbar"]
        if current_hotbar != last_hotbar:
            action["hotbar.{}".format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar

        # Read frame even if this is null so we progress forward
        ret, frame = video.read()

        # skip saving items before the selected range
        if i < from_step:
            continue

        if ret:
            # Convert action to ndarray and save
            agent_action = minerl_agent._env_action_to_agent(action)
            array_action = np.concatenate(
                (agent_action["camera"], agent_action["buttons"]), -1
            ).squeeze(0)
            acts[step_counter, ...] = array_action

            # Prepocess frame and save
            pov_frame, agent_frame = process_frame(
                frame, step_data, cursor_image, cursor_alpha
            )
            if pov_to_infos:
                # add original frame to infos
                infos["orig_observation"] = pov_frame
            # add downsampled frame to obs
            obs[step_counter, ...] = agent_frame
            step_counter += 1
        else:
            raise ValueError(f"Could not read frame {i} from video {video_path}")

    # we need one more frame than actions / transitions
    ret, frame = video.read()
    if ret:
        if i + 1 < len(json_data):
            step_data = json_data[i + 1]
        else:
            step_data = {"isGuiOpen": False}
        # Prepocess frame and save
        pov_frame, agent_frame = process_frame(
            frame, step_data, cursor_alpha, cursor_alpha
        )
        if pov_to_infos:
            # add original frame to infos
            infos["orig_observation"] = pov_frame
        # add downsampled frame to obs
        obs[step_counter, ...] = agent_frame
    else:
        raise ValueError(f"Could not read frame {i + 1} from video {video_path}")

    video.release()

    return Trajectory(
        obs=obs,
        acts=acts,
        infos=infos,
        terminal=terminal,
    )


def get_first_iter_element(iterable: Iterable[T]) -> Tuple[T, Iterable[T]]:
    """Get first element of an iterable and a new fresh iterable.

    The fresh iterable has the first element added back using ``itertools.chain``.
    If the iterable is not an iterator, this is equivalent to
    ``(next(iter(iterable)), iterable)``.

    Args:
        iterable: The iterable to get the first element of.

    Returns:
        A tuple containing the first element of the iterable, and a fresh iterable
        with all the elements.

    Raises:
        ValueError: `iterable` is empty -- the first call to it returns no elements.
    """
    iterator = iter(iterable)
    try:
        first_element = next(iterator)
    except StopIteration:
        raise ValueError(f"iterable {iterable} had no elements to iterate over.")

    return_iterable: Iterable[T]
    if iterator == iterable:
        # `iterable` was an iterator. Getting `first_element` will have removed it
        # from `iterator`, so we need to add a fresh iterable with `first_element`
        # added back in.
        return_iterable = itertools.chain([first_element], iterator)
    else:
        # `iterable` was not an iterator; we can just return `iterable`.
        # `iter(iterable)` will give a fresh iterator containing the first element.
        # It's preferable to return `iterable` without modification so that users
        # can generate new iterators from it as needed.
        return_iterable = iterable

    return first_element, return_iterable
