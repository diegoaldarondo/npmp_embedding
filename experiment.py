"""Run an experiment using comic architecture."""
import pickle
import argparse
import ast
import os
import pickle
import tensorflow.compat.v1 as tf
from observer import Observer, MlpObserver, LstmObserver
from feeder import MlpFeeder, LstmFeeder
from system import System
from loop import Loop, OpenLoop, ClosedLoop

tf.compat.v1.disable_eager_execution()


class Experiment:
    def __init__(
        self,
        system: System,
        observer: Observer,
        loop: Loop,
    ):
        """Initialize experiment

        Args:
            system (System): Experimental System
            observer (Observer): Experimental Observer
            loop (Loop): Experimental Loop
        """
        self.system = system
        self.observer = observer
        self.loop = loop

    def run(self):
        """Run the environment through the loop using system model."""
        with tf.Session() as sess:
            self.system.load_model(sess)
            graph = tf.get_default_graph()
            timestep, feed_dict, action_output = self.loop.initialize(sess)
            self.loop.loop(sess, action_output, timestep, feed_dict, self.observer)


def parse():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace of command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref-path",
        dest="ref_path",
        default="./JDM31_rat7m/data/total.hdf5",
        help="Path to dataset containing reference trajectories.",
    )
    parser.add_argument(
        "--import-dir",
        dest="model_dir",
        default="rodent_tracking_model_16212280_3_no_noise",
        help="Path to model import directory.",
    )
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
        "--save-dir",
        dest="save_dir",
        default=".",
        help="Path to save directory.",
    )
    parser.add_argument(
        "--ref-steps",
        dest="ref_steps",
        type=ast.literal_eval,
        default=(1, 2, 3, 4, 5),
        help="Number of steps to look ahead in the reference trajectory. ",
    )
    parser.add_argument(
        "--termination-error-threshold",
        dest="termination_error_threshold",
        type=float,
        default=0.25,
        help="Termination error threshold.",
    )
    parser.add_argument(
        "--min-steps",
        dest="min_steps",
        type=int,
        default=10,
        help="Minimum number of steps to take in environment.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default="jdm31_rat7m",
        help="Name of trajectory dataset to use. Must be registered in locomotion/tasks/reference_pose/datasets.py.",
    )
    parser.add_argument(
        "--reward-type",
        dest="reward_type",
        default="rat_mimic_force",
        help="Reward function name. See locomotion/tasks/reference_pose/rewards.py",
    )
    parser.add_argument(
        "--physics-timestep",
        dest="physics_timestep",
        type=float,
        default=0.001,
        help="Physics timestep.",
    )
    parser.add_argument(
        "--body-error-multiplier",
        dest="body_error_multiplier",
        type=int,
        default=10,
        help="Body error multiplier.",
    )
    parser.add_argument(
        "--video-length",
        dest="video_length",
        type=int,
        default=2500,
        help="Timesteps to include per video. Also sets checkpoint frequency",
    )
    parser.add_argument(
        "--start-step",
        dest="start_step",
        type=int,
        default=0,
        help="Time step in trajectory to start rollout.",
    )
    parser.add_argument(
        "--end-step",
        dest="end_step",
        type=int,
        default=2500,
        help="Time step in trajectory to finish rollout.",
    )
    return parser.parse_args()


def npmp_embed_single_batch():
    """CLI Entrypoint to embed a single batch in a multi processing system.

    Embeds a single batch in a batch array. Reads batch options from _batch_args.p.

    Optional Args:
        stac_params (Text): Path to stac params (.yaml).
        offset_path (Text): Path to stac output with offset(.p).

    Deleted Parameters:
        ref_path (Text): Path to .hdf5 reference trajectories.
        save_dir (Text): Folder in which to save .mat files.
        dataset (Text): Name of dataset registered in dm_control.
        model_dir (Text): Path to rodent tracking model.
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
        help="Folder in which to save .mat files.",
    )
    parser.add_argument(
        "dataset",
        help="Name of dataset registered in dm_control.",
    )
    parser.add_argument(
        "model_dir",
        help="path to rodent tracking model.",
    )
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
        "--batch-file",
        dest="batch_file",
        default="_batch_args.p",
        help="Path to stac output with offset(.p).",
    )
    parser.add_argument(
        "--use-open-loop",
        dest="use_open_loop",
        default=False,
        help="If True, use open loop.",
    )
    args = parser.parse_args()

    # Load in parameters to modify
    with open(args.batch_file, "rb") as file:
        batch_args = pickle.load(file)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    batch_args = batch_args[task_id]
    print(batch_args)
    # args = args.__dict__
    # del args["batch_file"]

    system = System(
        ref_path=args.ref_path,
        model_dir=args.model_dir,
        dataset=args.dataset,
        stac_params=args.stac_params,
        offset_path=args.offset_path,
        start_step=batch_args["start_step"],
        torque_actuators=batch_args["torque_actuators"],
    )
    if batch_args["lstm"]:
        observer = LstmObserver(system.environment, args.save_dir)
        feeder = LstmFeeder()
    else:
        observer = MlpObserver(system.environment, args.save_dir)
        feeder = MlpFeeder()

    if args.use_open_loop:
        loop = OpenLoop(
            system.environment,
            feeder,
            batch_args["start_step"],
            batch_args["video_length"],
        )
    else:
        loop = ClosedLoop(
            system.environment,
            feeder,
            batch_args["start_step"],
            batch_args["video_length"],
        )

    exp = Experiment(system, observer, loop)
    exp.run()
