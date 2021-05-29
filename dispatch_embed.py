"""Dispatch jobs for npmp/comic embedding."""
import os
import pickle
import h5py
import numpy as np
import argparse
from typing import Text, List, Dict, Tuple, Union
import time


class ParallelNpmpDispatcher:

    """Dispatches jobs to embed trajectories with npmp network in parallel.

    Attributes:
        clip_end (int): Last frame of clip.
        dataset (Text): Name of dataset registered in dm_control.
        end_steps (np.ndarray): Last steps of each chunk.
        import_dir (Text): Path to rodent tracking model.
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
        import_dir: Text,
        stac_params: Text,
        offset_path: Text,
        video_length: int = 2500,
        lstm: bool = False,
        torque_actuators: bool = False,
        batch_file="_batch_args.p",
    ):
        """Initialize ParallelNpmpDispatcher.

        Args:
            ref_path (Text): Path to .hdf5 reference trajectories.
            save_dir (Text): Folder in which to save videos and .mat files.
            dataset (Text): Name of dataset registered in dm_control.
            import_dir (Text): Path to rodent tracking model.
            stac_params (Text): Path to stac params (.yaml).
            offset_path (Text): Path to stac output with offset (.p).
            video_length (int, optional): Length of chunks to parallelize over.
        """
        self.ref_path = ref_path
        self.save_dir = save_dir
        self.dataset = dataset
        self.import_dir = import_dir
        self.video_length = video_length
        self.stac_params = stac_params
        self.offset_path = offset_path
        self.batch_file = batch_file
        self.clip_end = self.get_clip_end()

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
        cmd1 = (
            '"sbatch --wait --array=0-%d multi_job_embed.sh %s %s %s %s --stac-params=%s --offset-path=%s --batch-file=%s"'
            % (
                len(self.start_steps) - 1,
                self.ref_path,
                self.save_dir,
                self.dataset,
                self.import_dir,
                self.stac_params,
                self.offset_path,
                self.batch_file,
            )
        )
        # cmd1 = '"sbatch --wait --array=0-1 multi_job_embed.sh %s %s %s %s --stac-params=%s --offset-path=%s --batch-file=%s"' % (
        #     self.ref_path,
        #     self.save_dir,
        #     self.dataset,
        #     self.import_dir,
        #     self.stac_params,
        #     self.offset_path,
        #     self.batch_file,
        # )

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
        import_dir (Text): Path to rodent tracking model.
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
        dest="import_dir",
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


def dispatch(
    ref_path,
    save_dir,
    dataset,
    import_dir,
    stac_params,
    offset_path,
    lstm,
    torque_actuators,
    batch_file,
):
    dispatcher = ParallelNpmpDispatcher(
        ref_path,
        save_dir,
        dataset,
        import_dir,
        stac_params,
        offset_path,
        lstm=lstm,
        torque_actuators=torque_actuators,
        batch_file=batch_file,
    )
    dispatcher.dispatch()


if __name__ == "__main__":

    ref_path = "./npmp_preprocessing/total.hdf5"
    stac_params = "./stac_params/params.yaml"
    offset_path = "./stac/offset.p"
    dataset = "dannce_ephys_" + os.path.basename(os.getcwd())
    is_lstm = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    is_torque_actuators = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    model_dir = [
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_1_final",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_2_final",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_3",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_3_final",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_3_no_noise",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_4_final",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24184166_1",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24184166_2",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24184166_3",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24184166_4",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24184166_5",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_1",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_2",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_3",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_4",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_5",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24189285_2",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24189285_3",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24189285_4",
        "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24189285_5",
    ]

    save_dir = [os.path.join("npmp", os.path.basename(m)) for m in model_dir]
    for i, (s_dir, model, lstm, torque_actuators) in enumerate(
        zip(save_dir, model_dir, is_lstm, is_torque_actuators)
    ):
        time.sleep(1)
        batch_file = "_batch_args%d.p" % (i)
        dispatch(
            ref_path,
            s_dir,
            dataset,
            model,
            stac_params,
            offset_path,
            lstm,
            torque_actuators,
            batch_file,
        )
