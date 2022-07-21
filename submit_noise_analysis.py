import os
import yaml
import pickle
import argparse
from typing import List, Dict, Text
import dispatch_embed as de


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

    return total_params


def setup_job_params(params: Dict) -> List[Dict]:
    """Setup parameters for a batch array job.

    Args:
        params (Dict): Dispatch parameters dictionary.

    Returns:
        List[Dict]: List containing the parameters
    """
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
            # Setup parameters for each noise and gain
            for noise_type in params["latent_noise"]:
                for noise_gain in params["noise_gain"]:
                    job_params = {
                        "model_dir": model,
                        "lstm": lstm,
                        "torque_actuators": torque,
                    }

                    job_params["batch_file"] = os.path.join(
                        project_folder,
                        "_batch_args%d_%s_%f.p" % (n_model, noise_type, noise_gain),
                    )
                    job_params["save_dir"] = os.path.join(
                        project_folder,
                        params["save_dir"],
                        "noise_analysis",
                        os.path.basename(model),
                        noise_type + str(noise_gain),
                    )
                    job_params["dataset"] = (
                        project_folder.split("/")[-2]
                        + "_"
                        + os.path.basename(project_folder)
                    )
                    job_params["latent_noise"] = noise_type
                    job_params["noise_gain"] = noise_gain

                    for field in ["ref_path", "stac_params", "offset_path"]:
                        job_params[field] = os.path.join(project_folder, params[field])
                    for field in [
                        "unfinished_only",
                        "video_length",
                        "loop",
                        "test",
                        "action_noise",
                        "variability_clamp",
                    ]:
                        job_params[field] = params[field]
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


def submit(param_path: Text):
    """Submit jobs for a single run.

    Args:
        param_path (Text): Path to the parameters .yaml file.
    """
    params = build_params(param_path)

    # Only use params for the project folder specified by the task_id
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    params = params[task_id]
    dispatcher = de.ParallelNpmpDispatcher(params)
    dispatcher.dispatch()


def main():
    """Dispatch npmp jobs for multiple project folders.

    Only submits the jobs one project folder at a time.
    """
    param_path = parse()

    params = build_params(param_path)
    # import pdb

    # pdb.set_trace()
    cmd = "sbatch --array=0-%d%%20 submit_noise_analysis.sh %s" % (
        len(params) - 1,
        param_path,
    )
    # cmd = "sbatch --array=0-%d%%%d submit_projects.sh %s" % (len(params) - 1, N_PROJECT_FOLDERS_SIMULTANEOUSLY, param_path)
    os.system(cmd)


if __name__ == "__main__":
    main()
