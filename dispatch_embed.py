"""Dispatch jobs for npmp/comic embedding."""
import os
import pickle
import h5py
import numpy as np
import yaml
import argparse
from typing import Text, List, Dict, Tuple, Union
import subprocess

N_PROJECT_FOLDERS_SIMULTANEOUSLY = 1

EMBED_SCRIPT = (
    lambda n_array, batch_file: f"""#!/bin/bash
#SBATCH --array=0-{n_array}
#SBATCH --job-name=embedNPMP
#SBATCH --mem=12000
#SBATCH -t 0-03:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -p olveczky,shared,cox,serial_requeue
#SBATCH --exclude=holy2c18111
#SBATCH --constraint="intel&avx2"
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
source ~/.bashrc
mj_sim python -c "import experiment; experiment.npmp_embed_single_batch('{batch_file}')"
"""
)

MERGE_SCRIPT = (
    lambda out_folder, job_id: f"""#!/bin/bash
#SBATCH --job-name=mergeNPMP
#SBATCH --dependency=afterok:{job_id}
#SBATCH --mem=120000
#SBATCH -t 0-01:00
#SBATCH -N 1
#SBATCH -c 1
# # SBATCH --output=/dev/null 
# # SBATCH --error=/dev/null
#SBATCH -p olveczky,shared,cox
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx2"
source ~/.bashrc
mj_sim python -c "import merge_embed; merge_embed.merge('{out_folder}')"
"""
)


def slurm_submit(script: Text):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id


class ParallelNpmpDispatcher:

    """Dispatches jobs to embed trajectories with npmp network in parallel.

    Attributes:
        clip_end (int): Last frame of clip.
        dataset (Text): Name of dataset registered in dm_control.
        end_steps (np.ndarray): Last steps of each chunk.
        model_dir (Text): Path to rodent tracking model.
        offset_path (Text): Path to stac output with offset (.p).
        ref_path (Text): Path to .hdf5 reference trajectories.
        save_dir (Text): Folder in which to save videos and .mat files.
        stac_params (Text): Path to stac params (.yaml).
        start_steps (np.ndarray): First steps of each chunk.
        video_length (int, optional): Length of chunks to parallelize over.
    """

    def __init__(self, params: Dict):
        """Initialize ParallelNpmpDispatcher.

        Args:
            params (Dict): Parameters dictionary
        """
        # self.ref_path = ref_path
        # self.save_dir = save_dir
        # self.dataset = dataset
        # self.model_dir = model_dir
        # self.video_length = video_length
        # self.stac_params = stac_params
        # self.offset_path = offset_path
        # self.batch_file = batch_file
        self.params = params
        self.clip_end = self.get_clip_end()
        # self.test = test

        self.start_steps = np.arange(0, self.clip_end, self.params["video_length"])
        self.end_steps = self.start_steps + self.params["video_length"]
        self.end_steps[-1] = self.clip_end
        self.batch_args = []
        for start, end in zip(self.start_steps, self.end_steps):
            args = self.params.copy()
            args["start_step"] = start
            args["end_step"] = end
            args["lstm"] = self.params["lstm"]
            args["torque_actuators"] = self.params["torque_actuators"]
            self.batch_args.append(args)
        self.save_batch_args()

    def get_clip_end(self) -> int:
        """Get the final step in the dataset.

        Returns:
            int: Number of steps in the dataset.
        """
        with h5py.File(self.params["ref_path"], "r") as file:
            num_steps = 0
            for clip in file.keys():
                # num_steps += file[clip].attrs["num_steps"]
                num_steps += self.params["video_length"]
        return num_steps

    def _get_unfinished(self):
        out_folder = os.path.join(self.params["save_dir"], "logs")
        if os.path.isdir(out_folder):
            files = [f for f in os.listdir(out_folder) if ".mat" in f]
            files = [f for f in files if "data" not in f]
            inds = [int(f.split(".")[0]) for f in files]
            inds = [start not in inds for start in self.start_steps]
        else:
            inds = [True for _ in self.start_steps]
        return inds

    def dispatch(self):
        """Submit the job to the cluster."""
        if len(self.batch_args) >= 1:
            if not self.params["test"]:
                script = EMBED_SCRIPT(
                    len(self.batch_args) - 1, self.params["batch_file"]
                )
            else:
                script = EMBED_SCRIPT(1, self.params["batch_file"])

            job_id = slurm_submit(script)
            script2 = MERGE_SCRIPT(
                os.path.join(self.params["save_dir"], "logs"), job_id
            )
            job_id = slurm_submit(script2)
        else:
            script = MERGE_SCRIPT(os.path.join(self.params["save_dir"], "logs"), 1)
            script.replace(
                "#SBATCH --dependency=afterok:1", "# #SBATCH --dependency=afterok:1"
            )
            job_id = slurm_submit(script)
            print("No unfinished chunks", flush=True)

    def save_batch_args(self):
        """Save the batch arguments."""
        if self.params["unfinished_only"]:
            unfinished = self._get_unfinished()
            self.batch_args = [
                args for i, args in enumerate(self.batch_args) if unfinished[i]
            ]
        with open(self.params["batch_file"], "wb") as f:
            pickle.dump(self.batch_args, f)


def dispatch_npmp_embed():
    """CLI Entrypoint to dispatch npmp embedding in a multi processing system.

    Args:
        model_dir (Text): Path to rodent tracking model.
        stac_params (Text): Path to stac params (.yaml).
        offset_path (Text): Path to stac output with offset(.p).

    Optional Args:
        ref_path (Text): Path to .hdf5 reference trajectories.
        save_dir (Text): Folder in which to save .mat files.
        dataset (Text): Name of dataset registered in dm_control.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_path",
        help="Path to .hdf5 reference trajectories.",
    )
    parser.add_argument(
        "save_dir",
        help="Folder in which to save videos and .mat files.",
    )
    parser.add_argument(
        "dataset",
        help="Name of dataset registered in dm_control.",
    )
    parser.add_argument(
        "--import-dir",
        dest="model_dir",
        default="/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_3_no_noise",
        help="path to rodent tracking model.",
    )
    # "/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/__npmp_embedding/rodent_tracking_model_16212280_3_no_noise",
    parser.add_argument(
        "--stac-params",
        dest="stac_params",
        help="Path to stac params (.yaml).",
    )
    parser.add_argument(
        "--offset-path",
        dest="offset_path",
        help="Path to stac output with offset(.p).",
    )
    parser.add_argument(
        "--lstm",
        dest="lstm",
        default=False,
        help="Set to true if model is lstm.",
    )
    parser.add_argument(
        "--torque-actuators",
        dest="lstm",
        default=False,
        help="Set to true if model is lstm.",
    )

    args = parser.parse_args()
    dispatcher = ParallelNpmpDispatcher(**args.__dict__)
    dispatcher.dispatch()


def load_params(param_path: Text) -> Dict:
    """Load dispatch parameters.

    Args:
        param_path (Text): Path to parameters .yaml file.

    Returns:
        Dict: Dispatch parameters dictionary
    """
    with open(param_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def build_params(param_path: Text) -> List[Dict]:
    """Build list of parameters for a batch array job.

    Args:
        param_path (Text): Path to parameters .yaml file.

    Returns:
        List[Dict]: List containing a parameters
            (dicts) for each job for each project folder.
    """
    params = load_params(param_path)
    total_params = setup_job_params(params)
    return total_params


def build_params_for_session(param_path: Text, project_folders: List) -> List[Dict]:
    """Build list of parameters for a batch array job.

    Args:
        param_path (Text): Path to parameters .yaml file.

    Returns:
        List[Dict]: List containing a parameters
            (dicts) for each job for each project folder.
    """
    params = load_params(param_path)
    params["project_folders"] = project_folders
    total_params = setup_job_params(params)
    import pdb

    return total_params


def setup_job_params(params):
    total_params = []

    # Cycle through project folders and models.
    for project_folder in params["project_folders"]:
        for n_model, (model, lstm, torque) in enumerate(
            zip(
                params["model_dirs"],
                params["is_lstm"],
                params["is_torque_actuators"],
            )
        ):
            job_params = {
                "model_dir": model,
                "lstm": lstm,
                "torque_actuators": torque,
            }
            for field in ["ref_path", "stac_params", "offset_path"]:
                job_params[field] = os.path.join(project_folder, params[field])
            job_params["test"] = params["test"]
            job_params["batch_file"] = os.path.join(
                project_folder, "_batch_args%d.p" % (n_model)
            )
            job_params["save_dir"] = os.path.join(
                project_folder, params["save_dir"], os.path.basename(model)
            )
            job_params["dataset"] = (
                project_folder.split("/")[-2] + "_" + os.path.basename(project_folder)
            )
            job_params["latent_noise"] = params["latent_noise"]
            job_params["noise_gain"] = params["noise_gain"]
            job_params["action_noise"] = params["action_noise"]
            job_params["variability_clamp"] = params["variability_clamp"]
            job_params["unfinished_only"] = params["unfinished_only"]
            job_params["video_length"] = params["video_length"]
            job_params["loop"] = params["loop"]
            total_params.append(job_params.copy())
    return total_params


def parse() -> Dict:
    """Return the parameters specified in the command line.

    Returns:
        Text: Parameters path
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "params",
        help="Path to .yaml file specifying npmp embedding parameters.",
    )
    args = parser.parse_args()
    return args.params


def main():
    """Dispatch npmp jobs for multiple project folders.

    Only submits the jobs one project folder at a time.
    """
    param_path = parse()
    params = build_params(param_path)
    script = f"""#!/bin/bash
#SBATCH --job-name=submit_projects
# Job name
#SBATCH --mem=2000
# Job memory request
#SBATCH -t 1-00:00
# Time limit hrs:min:sec
#SBATCH --array=0-{len(params) - 1}%%20
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared
setup_miniconda
source activate dispatch_embed
python -c "import dispatch_embed; dispatch_embed.submit_project('{param_path}')"
wait
"""
    slurm_submit(script)


def submit_project(param_path: Text):
    """Submit jobs for a single run.

    Args:
        param_path (Text): Path to the parameters .yaml file.
    """
    params = build_params(param_path)

    # Only use params for the project folder specified by the task_id
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    params = params[task_id]
    dispatcher = ParallelNpmpDispatcher(params)
    dispatcher.dispatch()


def submit_session(param_path: Text, project_folder: Text):
    """Submit jobs for a single run.

    Args:
        param_path (Text): Path to the parameters .yaml file.
    """
    params = build_params_for_session(param_path, [project_folder])
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    params = params[task_id]
    # Only use params for the project folder specified by the task_id
    dispatcher = ParallelNpmpDispatcher(params)
    dispatcher.dispatch()


def main_session(param_path: Text, project_folder: Text):
    """Dispatch npmp jobs for multiple project folders.

    Only submits the jobs one project folder at a time.
    """
    params = build_params_for_session(param_path, [project_folder])
    cmd = "sbatch --wait --array=0-%d%%20 submit_session.sh %s %s" % (
        len(params) - 1,
        param_path,
        project_folder,
    )
    os.system(cmd)


if __name__ == "__main__":
    main()
