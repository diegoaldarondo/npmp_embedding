import dm_control
import h5py
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.arenas import floors
from dm_control import composer
from dm_control.utils import transformations as tr
from dm_control import mjcf
import pickle
import mocap_preprocess
import numpy as np
import sys
import os
import argparse


class NpmpPreprocessor:
    def __init__(
        self,
        stac_path,
        save_file,
        start_step=0,
        clip_length=2500,
        max_qvel=20,
        dt=0.02,
        adjust_z_offset=0.0,
        verbatim=False,
        ref_steps=(1, 2, 3, 4, 5),
    ):
        self.stac_path = stac_path
        self.save_file = save_file
        self.start_step = start_step
        self.max_qvel = max_qvel
        self.dt = dt
        self.adjust_z_offset = adjust_z_offset
        self.verbatim = verbatim
        self.clip_length = clip_length
        self.ref_steps = ref_steps

        with open(self.stac_path, "rb") as f:
            in_dict = pickle.load(f)
            self.qpos = in_dict["qpos"]
            # if end_step is not None:
            #     self.end_step = np.min([n_samples, self.end_step])
            #     self.qpos = in_dict["qpos"][self.start_step:self.end_step, :]
            # else:
            #     self.qpos = in_dict["qpos"][self.start_step:, :]

        self.walker = rodent.Rat(torque_actuators=True, foot_mods=True)
        self.arena = floors.Floor(size=(10.0, 10.0))
        self.walker.create_root_joints(self.arena.attach(self.walker))
        self.task = composer.NullTask(self.arena)
        self.env = composer.Environment(self.task)

    def extract_features(self):
        n_steps = self.qpos.shape[0]
        max_reference_index = np.max(self.ref_steps) + 1
        with h5py.File(self.save_file, "w") as file:
            for start_step in range(0, n_steps, self.clip_length):
                print(start_step, flush=True)
                end_step = np.min([start_step + self.clip_length + max_reference_index, n_steps])
                mocap_features = get_mocap_features(
                    self.qpos[start_step:end_step, :],
                    self.walker,
                    self.env.physics,
                    self.max_qvel,
                    self.dt,
                    self.adjust_z_offset,
                    self.verbatim,
                )

                mocap_features["scaling"] = []
                mocap_features["markers"] = []
                self.save_features(file, mocap_features, "clip_%d" % (start_step))

    def save_features(self, file, mocap_features, clip_name):
        clip_group = file.create_group(clip_name)
        n_steps = len(mocap_features["center_of_mass"])
        clip_group.attrs["num_steps"] = n_steps
        clip_group.attrs["dt"] = 0.02
        file.create_group("/" + clip_name + "/walkers")
        file.create_group("/" + clip_name + "/props")
        walker_group = file.create_group("/" + clip_name + "/walkers/walker_0")
        for k, v in mocap_features.items():
            if len(np.array(v).shape) == 3:
                v = np.transpose(v, (1, 2, 0))
                print(v.shape)
                walker_group[k] = np.reshape(np.array(v), (-1, n_steps))
            elif len(np.array(v).shape) == 2:
                v = np.swapaxes(v, 0, 1)
                walker_group[k] = v
            else:
                walker_group[k] = v


class ParallelNpmpPreprocessor(NpmpPreprocessor):
    def __init__(self, *args, **kwargs):
        super(ParallelNpmpPreprocessor, self).__init__(*args, **kwargs)

    def extract_features(self):
        n_steps = self.qpos.shape[0]
        max_reference_index = np.max(self.ref_steps) + 1
        with h5py.File(self.save_file, "w") as file:
            end_step = np.min([self.start_step + self.clip_length + max_reference_index, n_steps])
            mocap_features = get_mocap_features(
                self.qpos[self.start_step:end_step, :],
                self.walker,
                self.env.physics,
                self.max_qvel,
                self.dt,
                self.adjust_z_offset,
                self.verbatim,
            )

            mocap_features["scaling"] = []
            mocap_features["markers"] = []
            self.save_features(file, mocap_features, "clip_0")

def parallel_submit():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stac_path",
        help="Path to stac data containing reference trajectories.",
    )
    parser.add_argument(
        "save_folder",
        help="Path to .h5 file in which to save data.",
    )
    args = parser.parse_args()
    dispatcher = NpmpPreprocessingDispatcher(**args.__dict__)
    dispatcher.dispatch()

def submit():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stac_path",
        help="Path to stac data containing reference trajectories.",
    )
    parser.add_argument(
        "save_file",
        help="Path to .h5 file in which to save data.",
    )
    args = parser.parse_args()
    npmp_preprocessor = NpmpPreprocessor(**args.__dict__)
    npmp_preprocessor.extract_features()


def npmp_embed_preprocessing_single_batch():
    # Load in parameters to modify
    with open("_batch_preprocessing_args.p", "rb") as file:
        batch_args = pickle.load(file)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id = 0
    batch_args = batch_args[task_id]
    npmp_preprocessor = ParallelNpmpPreprocessor(**batch_args)
    npmp_preprocessor.extract_features()


class NpmpPreprocessingDispatcher:
    def __init__(
        self, stac_path, save_folder, clip_length=2500,
    ):
        self.stac_path = stac_path
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.clip_length = clip_length
        self.clip_end = self.get_clip_end()

        self.start_steps = np.arange(0, self.clip_end, self.clip_length)
        self.end_steps = self.start_steps + self.clip_length
        self.end_steps[-1] = self.clip_end
        batch_args = []
        for start, end in zip(self.start_steps, self.end_steps):
            batch_args.append(
                {
                    "stac_path": self.stac_path,
                    "save_file": os.path.join(self.save_folder, "%d.hdf5" % (start)),
                    "start_step": start,
                }
            )
            # print(batch_args)
        self.save_batch_args(batch_args)

    def get_clip_end(self):
        with open(self.stac_path, "rb") as f:
            in_dict = pickle.load(f)
            n_samples = in_dict["qpos"].shape[0]
        return n_samples

    def dispatch(self):
        cmd = "sbatch --wait --array=0-%d multi_job_embed_preprocessing.sh" % (
            len(self.start_steps) - 1
        )
        # cmd = "sbatch --array=0-9 multi_job_embed_preprocessing.sh"
        print(cmd)
        sys.exit(os.WEXITSTATUS(os.system(cmd)))

    def save_batch_args(self, batch_args):
        with open("_batch_preprocessing_args.p", "wb") as f:
            pickle.dump(batch_args, f)


def get_mocap_features(
    mocap_qpos,
    walker,
    physics,
    max_qvel,
    dt,
    adjust_z_offset,
    verbatim,
    null_xyr=False,
    shift_position=None,
    shift_rotation=None,
):
    """Convert mocap_qpos to valid reference features."""
    # Clip the angles.
    joint_names = [b.name for b in walker.mocap_joints]
    joint_ranges = physics.bind(walker.mocap_joints).range
    min_angles = joint_ranges[:, 0]
    max_angles = joint_ranges[:, 1]
    angles = mocap_qpos[:, 7:]
    clipped_angles = np.clip(angles, min_angles, max_angles)
    indexes = np.where(angles != clipped_angles)
    if verbatim and indexes[0].size != 0:
        for i, j in zip(*indexes):
            if np.abs(angles[i, j] - clipped_angles[i, j]) >= 0.1:
                print(
                    "Step {} angle of {} clipped from {} to {}.".format(
                        i, joint_names[j], angles[i, j], clipped_angles[i, j]
                    )
                )
    mocap_qpos[:, 7:] = clipped_angles
    # Generate the mocap_features.
    mocap_features = {}
    mocap_features["position"] = []
    mocap_features["quaternion"] = []
    mocap_features["joints"] = []
    mocap_features["center_of_mass"] = []
    mocap_features["end_effectors"] = []
    mocap_features["velocity"] = []
    mocap_features["angular_velocity"] = []
    mocap_features["joints_velocity"] = []
    mocap_features["appendages"] = []
    mocap_features["body_positions"] = []
    mocap_features["body_quaternions"] = []
    feet_height = []
    walker_bodies = walker.mocap_tracking_bodies
    body_names = [b.name for b in walker_bodies]
    if adjust_z_offset:
        left_foot_index = body_names.index("foot_L")
        right_foot_index = body_names.index("foot_R")

    # Padding for velocity corner case.
    mocap_qpos = np.concatenate([mocap_qpos, mocap_qpos[-1, np.newaxis,:]], axis=0)
    print(mocap_qpos.shape)
    qvel = np.zeros(len(mocap_qpos[0]) - 1)

    for n_frame, qpos in enumerate(mocap_qpos[:-1]):
        set_walker(
            physics,
            walker,
            qpos,
            qvel,
            null_xyr=null_xyr,
            position_shift=shift_position,
            rotation_shift=shift_rotation,
        )
        freejoint = mjcf.get_attachment_frame(walker.mjcf_model).freejoint
        root_pos = physics.bind(freejoint).qpos[:3].copy()
        mocap_features["position"].append(root_pos)
        root_quat = physics.bind(freejoint).qpos[3:].copy()
        mocap_features["quaternion"].append(root_quat)
        joints = np.array(physics.bind(walker.mocap_joints).qpos)
        mocap_features["joints"].append(joints)
        freejoint_frame = mjcf.get_attachment_frame(walker.mjcf_model)
        com = np.array(physics.bind(freejoint_frame).subtree_com)
        mocap_features["center_of_mass"].append(com)
        end_effectors = np.copy(
            walker.observables.end_effectors_pos(physics)[:]
        ).reshape(-1, 3)
        mocap_features["end_effectors"].append(end_effectors)
        if hasattr(walker.observables, "appendages_pos"):
            appendages = np.copy(walker.observables.appendages_pos(physics)[:]).reshape(
                -1, 3
            )
        else:
            appendages = np.copy(end_effectors)
        mocap_features["appendages"].append(appendages)
        xpos = physics.bind(walker_bodies).xpos.copy()
        mocap_features["body_positions"].append(xpos)
        xquat = physics.bind(walker_bodies).xquat.copy()
        mocap_features["body_quaternions"].append(xquat)
        if adjust_z_offset:
            feet_height += [xpos[left_foot_index][2], xpos[right_foot_index][2]]


    # Array
    mocap_features["position"] = np.array(mocap_features["position"])
    mocap_features["quaternion"] = np.array(mocap_features["quaternion"])
    mocap_features["joints"] = np.array(mocap_features["joints"])
    mocap_features["center_of_mass"] = np.array(mocap_features["center_of_mass"])
    mocap_features["end_effectors"] = np.array(mocap_features["end_effectors"])
    mocap_features["appendages"] = np.array(mocap_features["appendages"])
    mocap_features["body_positions"] = np.array(mocap_features["body_positions"])
    mocap_features["body_quaternions"] = np.array(mocap_features["body_quaternions"])

    # Offset vertically the qpos and xpos to ensure that the clip is aligned
    # with the floor. The heuristic uses the 10 lowest feet heights and
    # compensates for the thickness of the geoms.
    feet_height = np.sort(feet_height)
    if adjust_z_offset:
        z_offset = feet_height[:10].mean() - 0.006
    else:
        z_offset = 0
    mocap_qpos[:, 2] -= z_offset
    mocap_features["position"][:, 2] -= z_offset
    mocap_features["center_of_mass"][:, 2] -= z_offset
    mocap_features["body_positions"][:, :, 2] -= z_offset

    # Calculate qvel, clip.
    mocap_qvel = compute_velocity_from_kinematics(mocap_qpos, dt)
    vels = mocap_qvel[:, 6:]
    clipped_vels = np.clip(vels, -max_qvel, max_qvel)
    indexes = np.where(vels != clipped_vels)
    if verbatim and indexes[0].size != 0:
        for i, j in zip(*indexes):
            if np.abs(vels[i, j] - clipped_vels[i, j]) >= 0.1:
                print(
                    "Step {} velocity of {} clipped from {} to {}.".format(
                        i, joint_names[j], vels[i, j], clipped_vels[i, j]
                    )
                )
    mocap_qvel[:, 6:] = clipped_vels
    mocap_features["velocity"] = mocap_qvel[:, :3]
    mocap_features["angular_velocity"] = mocap_qvel[:, 3:6]
    mocap_features["joints_velocity"] = mocap_qvel[:, 6:]
    return mocap_features


def set_walker(
    physics,
    walker,
    qpos,
    qvel,
    offset=0,
    null_xyr=False,
    position_shift=None,
    rotation_shift=None,
):
    """Set the freejoint and walker's joints angles and velocities."""
    qpos = qpos.copy()
    if null_xyr:
        qpos[:3] = 0.0
        euler = tr.quat_to_euler(qpos[3:7], ordering="ZYX")
        euler[0] = 0.0
        quat = tr.euler_to_quat(euler, ordering="ZYX")
        qpos[3:7] = quat
    qpos[:3] += offset
    freejoint = mjcf.get_attachment_frame(walker.mjcf_model).freejoint
    physics.bind(freejoint).qpos = qpos[:7]
    physics.bind(freejoint).qvel = qvel[:6]
    physics.bind(walker.mocap_joints).qpos = qpos[7:]
    physics.bind(walker.mocap_joints).qvel = qvel[6:]
    if position_shift is not None or rotation_shift is not None:
        walker.shift_pose(
            physics,
            position=position_shift,
            quaternion=rotation_shift,
            rotate_velocity=True,
        )


def compute_velocity_from_kinematics(qpos_trajectory, dt):
    """Computes velocity trajectory from position trajectory.
  Args:
    qpos_trajectory: trajectory of qpos values T x ?
      Note assumes has freejoint as the first 7 dimensions
    dt: timestep between qpos entries
  Returns:
    Trajectory of velocities.
  """
    qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
    qvel_gyro = []
    for t in range(qpos_trajectory.shape[0] - 1):
        normed_diff = tr.quat_diff(qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7])
        normed_diff /= np.linalg.norm(normed_diff)
        qvel_gyro.append(tr.quat_to_axisangle(normed_diff) / dt)
    qvel_gyro = np.stack(qvel_gyro)
    qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
    return np.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)

def merge_preprocessed_files():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "data_folder",
        help="Path to folder with .hdf5 data.",
    )
    args = parser.parse_args()

    files = [f for f in os.listdir(args.data_folder) if (".hdf5" in f and "total" not in f)]
    file_ids = np.argsort([int(f.split('.')[0]) for f in files])
    files = [files[i] for i in file_ids]
    print(files)
    with h5py.File(os.path.join(args.data_folder, "total.hdf5"), 'w') as save_file:
        for file in files:
            print(file)
            with h5py.File(os.path.join(args.data_folder, file), 'r') as chunk:
                clip_name = 'clip_' + file.split('.')[0]
                # fd = save_file.create_group(clip_name)
                # fd = save_file.create_group('clip_0')
                chunk.copy('clip_0', save_file['/'], name=clip_name)

                
                

