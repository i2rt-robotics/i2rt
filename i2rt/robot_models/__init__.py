import os

_ROBOT_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))

# Arm XML paths
ARM_YAM_XML_PATH = os.path.join(_ROBOT_MODELS_ROOT, "arm/yam/yam.xml")
ARM_YAM_PRO_XML_PATH = os.path.join(_ROBOT_MODELS_ROOT, "arm/yam_pro/yam_pro.xml")
ARM_YAM_ULTRA_XML_PATH = os.path.join(_ROBOT_MODELS_ROOT, "arm/yam_ultra/yam_ultra.xml")
ARM_BIG_YAM_XML_PATH = os.path.join(_ROBOT_MODELS_ROOT, "arm/big_yam/big_yam.xml")

# Gripper XML paths
GRIPPER_CRANK_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/crank_4310/crank_4310.xml")
GRIPPER_LINEAR_3507_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/linear_3507/linear_3507.xml")
GRIPPER_LINEAR_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/linear_4310/linear_4310.xml")
GRIPPER_TEACHING_HANDLE_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/yam_teaching_handle/yam_teaching_handle.xml")
GRIPPER_FLEXIBLE_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/flexible_4310/flexible_4310.xml")
GRIPPER_NO_GRIPPER_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/no_gripper/no_gripper.xml")
