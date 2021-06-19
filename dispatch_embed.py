"""Dispatch jobs for npmp/comic embedding."""
import os
import pickle
import h5py
import numpy as np
import yaml
import argparse
from typing import Text, List, Dict, Tuple, Union
import time


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

    def __init__(
        self,
        ref_path: Text,
        save_dir: Text,
        dataset: Text,
        model_dir: Text,
        stac_params: Text,
        offset_path: Text,
        video_length: int = 2500,
        lstm: bool = False,
        torque_actuators: bool = False,
        batch_file="_batch_args.p",
        test: bool = False,
    ):
        """Initialize ParallelNpmpDispatcher.

        Args:
            ref_path (Text): Path to .hdf5 reference trajectories.
            save_dir (Text): Folder in which to save videos and .mat files.
            dataset (Text): Name of dataset registered in dm_control.
            model_dir (Text): Path to rodent tracking model.
            stac_params (Text): Path to stac params (.yaml).
            offset_path (Text): Path to stac output with offset (.p).
            video_length (int, optional): Length of chunks to parallelize over.
            test (bool): If True, only submit a small test job
        """
        self.ref_path = ref_path
        self.save_dir = save_dir
        self.dataset = dataset
        self.model_dir = model_dir
        self.video_length = video_length
        self.stac_params = stac_params
        self.offset_path = offset_path
        self.batch_file = batch_file
        self.clip_end = self.get_clip_end()
        self.test = test

        self.start_steps = np.arange(0, self.clip_end, self.video_length)
        self.end_steps = self.start_steps + self.video_length
        self.end_steps[-1] = self.clip_end
        batch_args = []
        for start, end in zip(self.start_steps, self.end_steps):
            args = {
                "start_step": start,
                "end_step": end,
                "lstm": lstm,
                "torque_actuators": torque_actuators,
            }
            batch_args.append(args)
        self.save_batch_args(batch_args)

    def get_clip_end(self) -> int:
        """Get the final step in the dataset.

        Returns:
            int: Number of steps in the dataset.
        """
        with h5py.File(self.ref_path, "r") as file:
            num_steps = 0
            for clip in file.keys():
                # num_steps += file[clip].attrs["num_steps"]
                num_steps += self.video_length
        return num_steps

    def dispatch(self):
        """Submit the job to the cluster."""
        if not self.test:
            cmd1 = '"sbatch --wait --array=0-%d multi_job_embed.sh %s %s %s %s --stac-params=%s --offset-path=%s --batch-file=%s"' % (
                len(self.start_steps) - 1,
                self.ref_path,
                self.save_dir,
                self.dataset,
                self.model_dir,
                self.stac_params,
                self.offset_path,
                self.batch_file,
            )
        else:
            cmd1 = '"sbatch --wait --array=0 multi_job_embed.sh %s %s %s %s --stac-params=%s --offset-path=%s --batch-file=%s"' % (
                self.ref_path,
                self.save_dir,
                self.dataset,
                self.model_dir,
                self.stac_params,
                self.offset_path,
                self.batch_file,
            )

        out_folder = os.path.join(self.save_dir, "logs")
        cmd2 = '"merge-embed %s"' % (out_folder)
        cmd = "sbatch embed.sh " + cmd1 + " " + cmd2
        print(cmd)
        os.system(cmd)

    def save_batch_args(self, batch_args: Dict):
        """Save the batch arguments.

        Args:
                batch_args (Dict): Arguments for each of the jobs in the batch array.
        """
        with open(self.batch_file, "wb") as f:
            # print(batch_args)
            pickle.dump(batch_args, f)


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


def dispatch(params: Dict):
    dispatcher = ParallelNpmpDispatcher(
        params["ref_path"],
        params["save_dir"],
        params["dataset"],
        params["model_dir"],
        params["stac_params"],
        params["offset_path"],
        lstm=params["lstm"],
        torque_actuators=params["torque_actuators"],
        batch_file=params["batch_file"],
        test=params["test"],
    )
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
        List[Dict]: Parameters for each batch job.
    """
    in_params = load_params(param_path)
    out_params = []

    # Cycle through project folders and models.
    for project_folder in in_params["project_folders"]:
        for n_model, (model, lstm, torque) in enumerate(
            zip(
                in_params["model_dirs"],
                in_params["is_lstm"],
                in_params["is_torque_actuators"],
            )
        ):
            batch_params = {
                "model_dir": model,
                "lstm": lstm,
                "torque_actuators": torque,
            }
            for field in ["ref_path", "stac_params", "offset_path"]:
                batch_params[field] = os.path.join(project_folder, in_params[field])
            batch_params["test"] = in_params["test"]
            batch_params["batch_file"] = os.path.join(
                project_folder, "_batch_args%d.p" % (n_model)
            )
            batch_params["save_dir"] = os.path.join(
                project_folder, in_params["save_dir"], os.path.basename(model)
            )
            batch_params["dataset"] = "dannce_ephys_" + os.path.basename(project_folder)
            out_params.append(batch_params.copy())
    return out_params


def parse() -> Dict:
    """Return the parameters specified in the command line.

    Returns:
        Dict: Parameters dictionary
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "params",
        help="Path to .yaml file specifying npmp embedding parameters.",
    )
    args = parser.parse_args()
    return build_params(args.params)


def main():
    params = parse()
    for batch_params in params:
        time.sleep(1)
        dispatch(batch_params)


if __name__ == "__main__":
    main()