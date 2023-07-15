import numpy as np
import tensorflow.compat.v1 as tf
from typing import Dict, List

OBSERVATIONS = [
    "walker/actuator_activation",
    "walker/appendages_pos",
    "walker/body_height",
    "walker/end_effectors_pos",
    "walker/joints_pos",
    "walker/joints_vel",
    "walker/sensors_accelerometer",
    "walker/sensors_force",
    "walker/sensors_gyro",
    "walker/sensors_torque",
    "walker/sensors_touch",
    "walker/sensors_velocimeter",
    "walker/tendons_pos",
    "walker/tendons_vel",
    "walker/world_zaxis",
    "walker/reference_rel_joints",
    "walker/reference_rel_bodies_pos_global",
    "walker/reference_rel_bodies_quats",
    "walker/reference_rel_bodies_pos_local",
    "walker/reference_ego_bodies_quats",
    "walker/reference_rel_root_quat",
    "walker/reference_rel_root_pos_local",
    "walker/reference_appendages_pos",
    "walker/clip_id",
    "walker/velocimeter_control",
    "walker/gyro_control",
    "walker/joints_vel_control",
    "walker/time_in_clip",
]

MLP_INPUTS = {
    "step_type": "step_type_2:0",
    "reward": "reward_2:0",
    "discount": "discount_1:0",
    "walker/actuator_activation": "walker/actuator_activation_1:0",
    "walker/appendages_pos": "walker/appendages_pos_1:0",
    "walker/body_height": "walker/body_height_1:0",
    "walker/end_effectors_pos": "walker/end_effectors_pos_1:0",
    "walker/joints_pos": "walker/joints_pos_1:0",
    "walker/joints_vel": "walker/joints_vel_1:0",
    "walker/sensors_accelerometer": "walker/sensors_accelerometer_1:0",
    "walker/sensors_force": "walker/sensors_force_1:0",
    "walker/sensors_gyro": "walker/sensors_gyro_1:0",
    "walker/sensors_torque": "walker/sensors_torque_1:0",
    "walker/sensors_touch": "walker/sensors_touch_1:0",
    "walker/sensors_velocimeter": "walker/sensors_velocimeter_1:0",
    "walker/tendons_pos": "walker/tendons_pos_1:0",
    "walker/tendons_vel": "walker/tendons_vel_1:0",
    "walker/world_zaxis": "walker/world_zaxis_1:0",
    "walker/reference_rel_joints": "walker/reference_rel_joints_1:0",
    "walker/reference_rel_bodies_pos_global": "walker/reference_rel_bodies_pos_global_1:0",
    "walker/reference_rel_bodies_quats": "walker/reference_rel_bodies_quats_1:0",
    "walker/reference_rel_bodies_pos_local": "walker/reference_rel_bodies_pos_local_1:0",
    "walker/reference_ego_bodies_quats": "walker/reference_ego_bodies_quats_1:0",
    "walker/reference_rel_root_quat": "walker/reference_rel_root_quat_1:0",
    "walker/reference_rel_root_pos_local": "walker/reference_rel_root_pos_local_1:0",
    "walker/reference_appendages_pos": "walker/reference_appendages_pos_1:0",
    "walker/clip_id": "walker/clip_id_1:0",
    "walker/velocimeter_control": "walker/velocimeter_control_1:0",
    "walker/gyro_control": "walker/gyro_control_1:0",
    "walker/joints_vel_control": "walker/joints_vel_control_1:0",
    "walker/time_in_clip": "walker/time_in_clip_1:0",
    "dummy_core_state": "state_9:0",
    "dummy_target_core_state": "state_10:0",
    "dummy_policy_state_level_1": "state_11:0",
    "dummy_policy_state_level_2": "state_12:0",
    "dummy_target_policy_state_level_1": "state_14:0",
    "dummy_target_policy_state_level_2": "state_15:0",
    "latent": "state_13:0",
    "target_latent": "state_16:0",
    "action": "state_17:0",
}

LSTM_INPUTS = {
    "step_type": "step_type_2:0",
    "reward": "reward_2:0",
    "discount": "discount_1:0",
    "walker/actuator_activation": "walker/actuator_activation_1:0",
    "walker/appendages_pos": "walker/appendages_pos_1:0",
    "walker/body_height": "walker/body_height_1:0",
    "walker/end_effectors_pos": "walker/end_effectors_pos_1:0",
    "walker/joints_pos": "walker/joints_pos_1:0",
    "walker/joints_vel": "walker/joints_vel_1:0",
    "walker/sensors_accelerometer": "walker/sensors_accelerometer_1:0",
    "walker/sensors_force": "walker/sensors_force_1:0",
    "walker/sensors_gyro": "walker/sensors_gyro_1:0",
    "walker/sensors_torque": "walker/sensors_torque_1:0",
    "walker/sensors_touch": "walker/sensors_touch_1:0",
    "walker/sensors_velocimeter": "walker/sensors_velocimeter_1:0",
    "walker/tendons_pos": "walker/tendons_pos_1:0",
    "walker/tendons_vel": "walker/tendons_vel_1:0",
    "walker/world_zaxis": "walker/world_zaxis_1:0",
    "walker/reference_rel_joints": "walker/reference_rel_joints_1:0",
    "walker/reference_rel_bodies_pos_global": "walker/reference_rel_bodies_pos_global_1:0",
    "walker/reference_rel_bodies_quats": "walker/reference_rel_bodies_quats_1:0",
    "walker/reference_rel_bodies_pos_local": "walker/reference_rel_bodies_pos_local_1:0",
    "walker/reference_ego_bodies_quats": "walker/reference_ego_bodies_quats_1:0",
    "walker/reference_rel_root_quat": "walker/reference_rel_root_quat_1:0",
    "walker/reference_rel_root_pos_local": "walker/reference_rel_root_pos_local_1:0",
    "walker/reference_appendages_pos": "walker/reference_appendages_pos_1:0",
    "walker/clip_id": "walker/clip_id_1:0",
    "walker/velocimeter_control": "walker/velocimeter_control_1:0",
    "walker/gyro_control": "walker/gyro_control_1:0",
    "walker/joints_vel_control": "walker/joints_vel_control_1:0",
    "walker/time_in_clip": "walker/time_in_clip_1:0",
    # "action": "state_17:0",
    "lstm_policy_hidden_level_1": "state_22:0",
    "lstm_policy_cell_level_1": "state_23:0",
    "lstm_policy_hidden_level_2": "state_24:0",
    "lstm_policy_cell_level_2": "state_25:0",
    "latent": "state_26:0",
    "target_latent": "state_32:0",
}

MLP_ACTIONS = {
    "action": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift/forward/add:0",
    "dummy_core_state": "agent_0/step_1/reset_core/Select:0",
    "dummy_target_core_state": "agent_0/step_1/reset_core_2/Select:0",
    "dummy_policy_state_level_1": "agent_0/step_1/reset_core_1/Select:0",
    "dummy_policy_state_level_2": "agent_0/step_1/reset_core_1/Select_1:0",
    "dummy_target_policy_state_level_1": "agent_0/step_1/reset_core_1_1/Select:0",
    "dummy_target_policy_state_level_2": "agent_0/step_1/reset_core_1_1/Select_1:0",
    "encoder_0": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso_encoder/model/mlp/Tanh:0",
    "encoder_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso_encoder/model/mlp/Tanh_1:0",
    "decoder_0": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/model/mlp/Tanh:0",
    "decoder_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/model/mlp/Tanh_1:0",
    "latent": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "latent_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "target_latent": "agent_0/step_1/reset_core_1_1/MultiLevelSamplerWithARPrior/add_2:0",
    "prior_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/mul_1:0",
    "level_1_scale": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0",
    "level_1_loc": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "latent_sample": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "action_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0",
}

LSTM_ACTIONS = {
    "action": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift/forward/add:0",
    "action_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0",
    "encoder_0": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso_encoder/model/mlp/Tanh:0",
    "encoder_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso_encoder/model/mlp/Tanh_1:0",
    "lstm_policy_hidden_level_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm/mul_2:0",
    "lstm_policy_cell_level_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm/add_2:0",
    "lstm_policy_hidden_level_2": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm_1/mul_2:0",
    "lstm_policy_cell_level_2": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm_1/add_2:0",
    "latent": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "latent_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "latent_sample": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "level_1_scale": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0",
    "level_1_loc": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "target_latent": "agent_0/step_1/reset_core_1_1/MultiLevelSamplerWithARPrior/add_2:0",
}

MLP_STATES = [
    "latent",
    "target_latent",
    "dummy_core_state",
    "dummy_target_core_state",
    "dummy_policy_state_level_1",
    "dummy_policy_state_level_2",
    "dummy_target_policy_state_level_1",
    "dummy_target_policy_state_level_2",
    "action",
]

LSTM_STATES = [
    "latent",
    "target_latent",
    "lstm_policy_hidden_level_1",
    "lstm_policy_cell_level_1",
    "lstm_policy_hidden_level_2",
    "lstm_policy_cell_level_2",
]


class Feeder:
    def __init__(
        self,
        inputs: Dict,
        actions: Dict,
        states: List,
        observations: List = OBSERVATIONS,
        tag: str = "",
    ):
        self.inputs = inputs
        self.actions = actions
        self.states = states
        self.observations = observations
        self.tag = tag

    def feed(self, timestep, action_output_np: np.ndarray = None):
        feed_dict = {}
        for obs in self.observations:
            feed_dict[self.graph_inputs[obs]] = timestep.observation[obs]
        for state in self.states:
            # TODO Check if there is a bug with setting the obs to 0 in the first step
            if action_output_np is None:
                feed_dict[self.graph_inputs[state]] = np.zeros(
                    self.graph_inputs[state].shape
                )
            else:
                feed_dict[self.graph_inputs[state]] = action_output_np[state].flatten()
        feed_dict[self.graph_inputs["step_type"]] = timestep.step_type
        feed_dict[self.graph_inputs["reward"]] = timestep.reward
        feed_dict[self.graph_inputs["discount"]] = timestep.discount
        return feed_dict

    def get_inputs(self, sess: tf.Session) -> Dict:
        """Setup graph_inputs for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: full input dict
        """
        self.graph_inputs = {
            self.tag + k: sess.graph.get_tensor_by_name(self.tag + v)
            for k, v in self.inputs.items()
        }
        return self.graph_inputs

    def get_outputs(self, sess: tf.Session) -> Dict:
        """Setup action output for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: Action output dict
        """
        try:
            action_output = {
                self.tag + k: sess.graph.get_tensor_by_name(self.tag + v)
                for k, v in self.actions.items()
            }
        except KeyError:
            # Use the alternate action tensor for the new LSTM experiments.
            self.actions[
                "action"
            ] = "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/MultivariateNormalDiag_CONSTRUCTED_AT_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head/sample/chain_of_shift_of_scale_matvec_linear_operator/forward/shift/forward/add:0"
            action_output = {
                self.tag + k: sess.graph.get_tensor_by_name(self.tag + v)
                for k, v in self.actions.items()
            }
        return action_output


class MlpFeeder(Feeder):
    def __init__(self, **kwargs):
        super().__init__(MLP_INPUTS, MLP_ACTIONS, MLP_STATES, **kwargs)


class LstmFeeder(Feeder):
    def __init__(self, **kwargs):
        super().__init__(LSTM_INPUTS, LSTM_ACTIONS, LSTM_STATES, **kwargs)
