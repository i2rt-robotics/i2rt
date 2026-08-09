import os
import re
from typing import List

_ROBOT_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
_ARM_ROOT = os.path.join(_ROBOT_MODELS_ROOT, "arm")
_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def _arm_xml_path(arm: str, version: int) -> str:
    """Build the MJCF path for ``arm`` at model ``version``. Does not check existence."""
    return os.path.join(_ARM_ROOT, arm, f"v{version}", f"{arm}.xml")


def available_arm_versions(arm: str) -> List[int]:
    """Model versions shipped for ``arm``, ascending. Empty if the arm is unknown.

    A ``v<N>`` dir only counts once it actually holds ``<arm>.xml``. An in-progress import --
    e.g. a raw CAD export dropped in before the alignment pipeline has been run -- is a directory,
    not a shipped version, and must not be reported as one.
    """
    arm_dir = os.path.join(_ARM_ROOT, arm)
    if not os.path.isdir(arm_dir):
        return []
    return sorted(
        int(m.group(1))
        for name in os.listdir(arm_dir)
        if (m := _VERSION_DIR_RE.match(name)) and os.path.isfile(os.path.join(arm_dir, name, f"{arm}.xml"))
    )


def available_arm_families() -> List[str]:
    """Arm families with at least one shipped model, ascending.

    Same valve as ``available_arm_versions``, one level up: a family dir counts only once some
    ``v<N>/`` under it holds ``<arm>.xml``, so a raw CAD export staged ahead of the alignment
    pipeline is not yet a family. Scans the filesystem rather than any in-code registry, so a
    brand-new family is visible to callers that keep the registry and the models in sync.
    """
    if not os.path.isdir(_ARM_ROOT):
        return []
    return sorted(name for name in os.listdir(_ARM_ROOT) if available_arm_versions(name))


def get_arm_xml_path(arm: str, version: int = 1) -> str:
    """Return the arm-only MJCF path for ``arm`` at model ``version``.

    Layout is ``arm/<arm>/v<version>/<arm>.xml``; the URDF and the ``assets/`` mesh dir sit
    beside it in the same version dir.
    """
    path = _arm_xml_path(arm, version)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No model for arm {arm!r} version {version}: {path} does not exist. "
            f"Available versions for {arm!r}: {available_arm_versions(arm)}"
        )
    return path


# Arm XML paths — deprecated v1 aliases, kept so downstream users of the public package keep
# working. Prefer get_arm_xml_path(arm, version). Built without a filesystem check so importing
# this package never raises, matching the previous plain-os.path.join behavior.
ARM_YAM_XML_PATH = _arm_xml_path("yam", 1)
ARM_YAM_PRO_XML_PATH = _arm_xml_path("yam_pro", 1)
ARM_YAM_ULTRA_XML_PATH = _arm_xml_path("yam_ultra", 1)
ARM_BIG_YAM_XML_PATH = _arm_xml_path("big_yam", 1)

# Gripper XML paths
GRIPPER_CRANK_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/crank_4310/crank_4310.xml")
GRIPPER_LINEAR_3507_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/linear_3507/linear_3507.xml")
GRIPPER_LINEAR_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/linear_4310/linear_4310.xml")
GRIPPER_TEACHING_HANDLE_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/yam_teaching_handle/yam_teaching_handle.xml")
GRIPPER_FLEXIBLE_4310_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/flexible_4310/flexible_4310.xml")
GRIPPER_NO_GRIPPER_PATH = os.path.join(_ROBOT_MODELS_ROOT, "gripper/no_gripper/no_gripper.xml")
