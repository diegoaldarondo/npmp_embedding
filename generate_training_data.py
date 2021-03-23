"""Generate training data for CoMic training.

Attributes:
    STILL_CLUSTER (int): ID for non-moving frames.
"""
import numpy as np
import pickle
from scipy.io import loadmat
from scipy.ndimage.measurements import label
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
import os
import sys
from typing import Dict, List
import random
from stac.util import load_params
from stac.view_stac import setup_visualization

STILL_CLUSTER = 51


def load_stac(stac_path) -> Dict:
    """Load stac file.

    Returns:
        Dict: Dict with kinematic information for each frame.

    Args:
        stac_path (TYPE): Description
    """
    with open(stac_path, "rb") as file:
        in_dict = pickle.load(file)
    return in_dict


class BoutGenerator:
    """Helper class to extract bouts of particular behaviors.

    Attributes:
        beh (np.ndarray): Array of cluster ids for each frame.
        beh_path (str): Path to behavioral clustering file.
        param_path (TYPE): Description
        params (TYPE): Description
        stac (Dict): Dict with kinematic information for each frame.
        stac_path (str): Path to stac file.
    """

    def __init__(self, stac_path: str, beh_path: str, param_path: str):
        """Init BoutGenerator

        Args:
            stac_path (str): Path to stac file.
            beh_path (str): Path to behavioral clustering file.
            param_path (str): Path to stac params.yaml file.
        """
        self.stac_path = stac_path
        self.beh_path = beh_path
        self.param_path = param_path
        self.stac = load_stac(self.stac_path)
        self.beh = self.load_beh()
        self.params = load_params(param_path)

    def load_beh(self) -> np.ndarray:
        """Load behavioral cluster file.

        Returns:
            np.ndarray: Array of cluster ids for each frame.
        """
        M = loadmat(self.beh_path)
        return M["ids"][:]

    def getBouts(
        self,
        bout_ids: List[int],
        description: str = None,
        duration: int = 250,
        dilation_kernel: int = 5,
    ) -> List[Dict]:
        """Get all bouts of specific groups of bout_ids.

        Args:
            bout_ids (List[int]): List of integers specifying bouts to include.
            description (str, optional): Optional description of the bouts.
            duration (int, optional): Duration of time surrounding bout center.
            dilation_kernel (int, optional): Duration of the dilation kernel.
        """
        # Get the indices that are a part of the bouts.
        is_bout = np.in1d(self.beh, bout_ids).astype("int")

        # Dilate the indices a bit to avoid oversegmenting.
        kernel = np.ones((dilation_kernel,), dtype=np.int)
        is_bout = np.convolve(is_bout, kernel, mode="same").astype(bool)

        # Find the connected components
        structure = np.ones((3,), dtype=np.int)
        labeled, n_components = label(is_bout, structure)
        bout_centers = [
            np.mean(np.argwhere(labeled == i)).round().astype("int")
            for i in range(1, n_components)
        ]

        # Only keep bouts that are completely separated from one another.
        overlapping = []
        for n_bout, bc in enumerate(bout_centers):
            if n_bout == 0:
                last_bc = bc
                continue
            if bc < (last_bc + duration):
                overlapping.append(n_bout)
            else:
                last_bc = bc

        for n_bout in reversed(overlapping):
            del bout_centers[n_bout]

        bout_indices = [
            np.arange(
                c - np.round(duration / 2), c + np.round(duration / 2)
            ).astype("int")
            for c in bout_centers
        ]

        # Set some defaults for rendering. 
        self.stac["qpos"] = median_filter(self.stac["qpos"], (5, 1))
        tail_ids = np.argwhere(["walker/vertebra_C" in n for n in self.stac["names_qpos"]])[:]
        tail_ids = [t[0] for t in tail_ids]
        self.stac["qpos"][:, tail_ids] = 0.
        tail_extend_ids = [25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47]
        self.stac["qpos"][:, tail_extend_ids] += .3
        mandible_id = np.argwhere([n == "walker/mandible" for n in self.stac["names_qpos"]])[0][0]
        self.stac["qpos"][:, mandible_id] = self.params["_MANDIBLE_POS"]


        # Get the kinematic information for the bout
        bouts = []
        for ids in bout_indices:
            # Only use in-bounds bouts.
            if any(ids < 0) or any(ids > self.stac["qpos"].shape[0]):
                continue
            bout = {}
            bout["qpos"] = self.stac["qpos"][ids, ...].copy()
            bout["kp_data"] = self.stac["kp_data"][ids, ...].copy()
            bout["names_qpos"] = self.stac["names_qpos"].copy()
            bout["offsets"] = self.stac["offsets"].copy()
            bout["description"] = description
            bout["_ARENA_DIAMETER"] = self.params["_ARENA_DIAMETER"]
            bout["_ARENA_CENTER"] = self.params["_ARENA_CENTER"]
            bout["param_path"] = self.param_path
            bouts.append(bout)
        return bouts


class TrainingSetGenerator:
    """Helper class to generate a training set for CoMic.

    Attributes:
        beh_paths (List[str]): List of paths to behavioral files for all exps.
        param_paths (TYPE): List of paths to stac params.yaml files.
        stac_paths (List[str]): List of paths to stac files for all exps.

    """

    def __init__(
        self,
        stac_paths: List[str],
        beh_paths: List[str],
        param_paths: List[str],
    ):
        """Init TrainingSetGenerator

        Args:
            stac_paths (List[str]): List of paths to stac files.
            beh_paths (List[str]): List of paths to behavioral files.
            param_paths (List[str]): List of paths to stac params.yaml files.
        """
        self.stac_paths = stac_paths
        self.beh_paths = beh_paths
        self.param_paths = param_paths

    def get_candidates(
        self, bout_groups: List[List[int]], descriptions: List[str]
    ) -> List[List]:
        """Get the candidate bouts for each experiment.

        Args:
            bout_groups (List[List[int]]): A list of groups of cluster ids.
            descriptions (List[str]): Description of each group.
        """
        candidates = [[] for _ in range(len(bout_groups))]
        for stac_path, beh_path, param_path in zip(
            self.stac_paths, self.beh_paths, self.param_paths
        ):
            print(stac_path)
            print(beh_path)
            print(param_path)
            bg = BoutGenerator(stac_path, beh_path, param_path)
            for n_bout, bout_ids in enumerate(bout_groups):
                bouts = bg.getBouts(bout_ids, description=descriptions[n_bout])
                for b in bouts:
                    candidates[n_bout].append(b)
        return candidates

    def get_training_set(
        self,
        bout_groups: List[List[int]],
        descriptions: List[str],
        n_bouts: List[int],
        random_state: int = 0,
    ) -> Dict:
        """Generate a training set for CoMic training.

        Args:
            bout_groups (List[List[int]]): A list of groups of cluster ids.
            descriptions (List[str]): Description of each group.
            n_bouts (List[int]): Number of bouts to take from each group.
            random_state (int, optional): Random seed.
        """

        def sort_by_com_speed(bouts: List[Dict]) -> List[Dict]:
            """Sort bouts by COM speed.

            Args:
                bouts (List[Dict]): List of bouts.

            Returns:
                List[Dict]: List of bouts sorted by COM speed.
            """
            speeds = [
                np.nanmean(
                    np.sqrt(
                        np.sum(np.diff(b["qpos"][:, 1:3], axis=0) ** 2, axis=1)
                    )
                )
                for b in bouts
            ]
            sorted_speed_ids = np.argsort(speeds)[::-1].tolist()
            return [bouts[i] for i in sorted_speed_ids]

        candidates = self.get_candidates(bout_groups, descriptions)

        # For the walking bouts, order them by speed.
        walk_group = np.argwhere(["Walk" in d for d in descriptions])[0][
            0
        ].astype("int")
        candidates[walk_group] = sort_by_com_speed(candidates[walk_group])

        # Get an assortment of n_bout examples for each type.
        training_set = {}
        random.seed(random_state)
        for des, can, n_bout in zip(descriptions, candidates, n_bouts):
            if des == "Walk":
                # Just get the fastest walks.
                fast_walks = can[:n_bout].copy()
                for i in range(len(fast_walks)):
                    fast_walks[i]["description"] = "FastWalk"
                training_set["FastWalk"] = fast_walks

                # Get random remaining walks
                training_set["Walk"] = random.sample(
                    can[n_bout:].copy(), n_bout
                )
            else:
                # Get random remaining behaviors
                training_set[des] = random.sample(can.copy(), n_bout)
        return training_set

    def render_training_set(self, training_set: Dict, save_folder: str):
        """Render a video for each bout in the training set.

        Args:
            training_set (Dict): Training set to render
            save_folder (str): Folder in which to save videos
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Make a video for each bout
        for des, bouts in training_set.items():
            for n_bout, b in enumerate(bouts):
                print("Rendering %s %d" % (des, n_bout), flush=True)
                # if des != "FaceGroom":
                #     continue
                q = b["qpos"]
                kp_data = b["kp_data"]
                offsets = b["offsets"]
                n_frames = q.shape[0]
                save_path = os.path.join(
                    save_folder, "%s_%d.mp4" % (des, n_bout)
                )
                setup_visualization(
                    b["param_path"],
                    q,
                    offsets,
                    kp_data,
                    n_frames,
                    render_video=True,
                    save_path=save_path,
                    headless=True,
                )

    def save_training_set(self, training_set: Dict, save_folder: str):
        """Save the training set to a folder.

        Args:
            training_set (Dict): Training set to save
            save_folder (str): Folder in which to save data
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Save each bout
        for des, bouts in training_set.items():
            for n_bout, b in enumerate(bouts):
                save_path = os.path.join(
                    save_folder, "%s_%d.p" % (des, n_bout)
                )
                with open(save_path, "wb") as f:
                    pickle.dump(b, f, protocol=2)


def render_training_set_single_batch():
    """Render training set for a single batch in a job array.

    Args:
        bouts_folder (Text): Path to folder containing Comic training set (.p)
    """
    bout_folder = sys.argv[1]
    save_folder = sys.argv[2]
    bout_names = np.sort(os.listdir(bout_folder))
    bout_paths = [os.path.join(bout_folder, p) for p in bout_names]
    bout_names = [os.path.splitext(n)[0] for n in bout_names]
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    bout = load_stac(bout_paths[task_id])
    n_frames = bout["qpos"].shape[0]
    save_path = os.path.join(save_folder, "%s.mp4" % (bout_names[task_id]))
    setup_visualization(
        bout["param_path"],
        bout["qpos"],
        bout["offsets"],
        bout["kp_data"],
        n_frames,
        render_video=True,
        save_path=save_path,
        headless=True,
    )


if __name__ == "__main__":

    # Pathing params
    beh_folder = "/n/holylfs02/LABS/olveczky_lab/Diego/data/dannce_ephys/art/behavioral_clusters"
    animal_folder = (
        "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art"
    )
    save_data_folder = "/n/holylfs02/LABS/olveczky_lab/Diego/data/dannce_ephys/art/CoMic_training_set/snips"
    save_video_folder = "/n/holylfs02/LABS/olveczky_lab/Diego/data/dannce_ephys/art/CoMic_training_set/videos"

    # Make the bout groups
    n_bouts = [200, 200, 200, 200, 200]
    bout_groups = [
        [33, 35],
        [43, 44, 45, 46, 47, 48, 49, 50],
        [13],
        [36, 37, 38],
        [18, 19, 39],
    ]
    descriptions = ["Walk", "Rear", "LGroom", "RGroom", "FaceGroom"]
    random_state = 0

    # Pathing
    beh_paths = np.sort(os.listdir(beh_folder))
    beh_paths = [os.path.join(beh_folder, p) for p in beh_paths]
    project_folders = np.sort(os.listdir(animal_folder))
    project_folders = project_folders[1:43].tolist()
    project_folders = [os.path.join(animal_folder, p) for p in project_folders]
    del project_folders[10]
    stac_paths = [
        os.path.join(pf, "stac", "total.p") for pf in project_folders
    ]
    param_paths = [
        os.path.join(pf, "stac_params", "params.yaml")
        for pf in project_folders
    ]

    # stac_paths = stac_paths[:2]
    # beh_paths = beh_paths[:2]
    # param_paths = param_paths[:2]
    for s, b in zip(stac_paths, beh_paths):
        print(s, b)

    # Build the dataset, save the dataset, and render the videos
    tsg = TrainingSetGenerator(stac_paths, beh_paths, param_paths)
    training_set = tsg.get_training_set(
        bout_groups, descriptions, n_bouts, random_state=random_state
    )
    tsg.save_training_set(training_set, save_data_folder)

    # Single job rendering.
    # tsg.render_training_set(training_set, save_video_folder)

    # Multi job rendering
    n_data_files = len(os.listdir(save_data_folder))
    command = "sbatch --array=0-%d render_training_set_videos.sh %s %s" % (
        n_data_files - 1,
        save_data_folder,
        save_video_folder,
    )

    # For testing
    # command = "sbatch --array=0-%d render_training_set_videos.sh %s %s" % (
    #     1,
    #     save_data_folder,
    #     save_video_folder,
    # )

    os.system(command)
