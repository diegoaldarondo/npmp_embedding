import os
import numpy as np
import argparse
from typing import Text, List, Dict
from scipy.io import loadmat
import h5py


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


def merge_files():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        dest="folder",
        help="Path to npmp output folder.",
    )
    parser.add_argument(
        "--delete-chunks",
        dest="delete_chunks",
        default=True,
        help="Whether or not to delete chunks",
    )
    args = parser.parse_args()

    # Get the files
    log_files = get_files(args.folder)

    # Get the first file for the keys
    chunk = loadmat(log_files[0])
    fields = [field for field in chunk.keys() if '__' not in field]

    # Load the data
    data = {field: [] for field in fields}
    for f in log_files:
        chunk = loadmat(f)
        for field in fields:
            data[field].append(chunk[field][:])

    # Concatenate all the chunks and make float16, if possible.
    data = {field: np.concatenate(data[field]) for field in fields}
    for field in fields:
        if (
            data[field].dtype == "float"
            or data[field].dtype == "float32"
        ):
            data[field] = data[field].astype("float16").squeeze()

    with h5py.File(os.path.join(args.folder, "data.hdf5"), "w") as save_file:
        for field, v in data.items():
            if field != "action_names":
                save_file.create_dataset(field, data=v)
            else:
                names = v[: data["action_mean"].shape[0]]
                names = [name.split("walker/")[-1].strip() for name in names]
                save_file.create_dataset("action_names", data=np.array(names, dtype=h5py.string_dtype('utf-8', 30)))

    if args.delete_chunks:
        for f in log_files:
            print("deleting", f)
            os.remove(f)
