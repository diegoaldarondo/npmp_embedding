import os
import numpy as np
import argparse
from typing import Text, List, Dict
from scipy.io import loadmat
import h5py
import dispatch_embed


def get_files(folder: Text) -> List:
    """Get files from video or log folders in order.

    Args:
        folder (Text): Folder to parse for files

    Returns:
        List: Sorted file names.
    """
    files = [f for f in os.listdir(folder) if ".mat" in f]
    ids = [int(f.split(".")[0]) for f in files]
    order = np.argsort(ids)
    return [os.path.join(folder, files[i]) for i in order]


def merge(folder, delete_chunks=False):
    """Merge embedding files.

    Args:
        folder (str): Path to npmp output folder.
        delete_chunks (bool, optional): Whether or not to delete chunks. Defaults to False.
    """
    log_files = get_files(folder)
    data = load_files(log_files)
    save_merge_file(folder, data)
    if delete_chunks:
        for f in log_files:
            print("deleting", f)
            os.remove(f)


def load_files(log_files: List) -> Dict:
    """Load and merge data from all batch files.

    Args:
        log_files (List): List of files to load

    Returns:
        Dict: Merged data
    """
    # Get the first file for the keys
    chunk = loadmat(log_files[0])
    fields = [field for field in chunk.keys() if "__" not in field]

    # Load the data
    data = {field: [] for field in fields}
    for f in log_files:
        chunk = loadmat(f)
        for field in fields:
            val = chunk[field][:]
            if val.dtype == "float" or val.dtype == "float32":
                val = val.astype("float16").squeeze()
            data[field].append(val)

    # Concatenate the data.
    for field in data.keys():
        data[field] = np.concatenate(data[field])
    return data


def save_merge_file(folder: Text, data: Dict):
    """Save merged data to file.

    Args:
        folder (Text): Folder in which to save data.
        data (Dict): Merged data.
    """
    with h5py.File(os.path.join(folder, "data.hdf5"), "w") as save_file:
        for field, v in data.items():
            if field != "action_names":
                save_file.create_dataset(field, data=v, compression="gzip")
            else:
                names = v[: data["action_mean"].shape[0]]
                names = [name.split("walker/")[-1].strip() for name in names]
                save_file.create_dataset(
                    "action_names",
                    data=np.array(names, dtype=h5py.string_dtype("utf-8", 30)),
                )


def merge_all(params_path: Text):
    """Merge all of the chunks in the save folders specified by the params dict.

    Args:
        params_path (Text): Path to embedding configuration .yaml file.
    """
    params = dispatch_embed.build_params(params_path)
    for job_params in params:
        save_folder = os.path.join(job_params["save_dir"], "logs")
        merge(save_folder)
