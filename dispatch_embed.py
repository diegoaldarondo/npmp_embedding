"""Dispatch jobs for npmp/comic embedding."""
import os
import pickle
import h5py
import numpy as np
import argparse
from typing import Text, List, Dict, Tuple, Union


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
        self.clip_end = self.get_clip_end()

        self.start_steps = np.arange(0, self.clip_end, self.video_length)
        self.end_steps = self.start_steps + self.video_length
        self.end_steps[-1] = self.clip_end
        batch_args = []
        for start, end in zip(self.start_steps, self.end_steps):
            batch_args.append({"start_step": start, "end_step": end})
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
        cmd = (
            "sbatch --array=0-%d multi_job_embed.sh %s %s %s %s --stac-params=%s --offset-path=%s"
            % (
                len(self.start_steps) - 1,
                self.ref_path,
                self.save_dir,
                self.dataset,
                self.import_dir,
                self.stac_params,
                self.offset_path,
            )
        )
        # cmd = "sbatch --array=0-0 multi_job_embed.sh %s %s %s %s" % (
        #     self.ref_path,
        #     self.save_dir,
        #     self.dataset,
        #     self.import_dir,
        # )
        print(cmd)
        os.system(cmd)

    def save_batch_args(self, batch_args: Dict):
        """Save the batch arguments.
        
        Args:
            batch_args (Dict): Arguments for each of the jobs in the batch array.
        """
        with open("_batch_args.p", "wb") as f:
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_path", help="Path to .hdf5 reference trajectories.",
    )
    parser.add_argument(
        "save_dir", help="Folder in which to save videos and .mat files.",
    )
    parser.add_argument(
        "dataset", help="Name of dataset registered in dm_control.",
    )
    parser.add_argument(
        "--import-dir",
        dest="import_dir",
        default="/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/__npmp_embedding/rodent_tracking_model_16212280_3_no_noise",
        help="path to rodent tracking model.",
    )
    parser.add_argument(
        "--stac-params", dest="stac_params", help="Path to stac params (.yaml).",
    )
    parser.add_argument(
        "--offset-path",
        dest="offset_path",
        help="Path to stac output with offset(.p).",
    )

    args = parser.parse_args()
    dispatcher = ParallelNpmpDispatcher(**args.__dict__)
    dispatcher.dispatch()
