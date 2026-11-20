"""Write a URDF next to each gripper MJCF, converted from the MJCF.

The gripper MJCFs are the source of truth and their URDFs are generated, so edit the MJCF and re-run this
instead of editing a gripper URDF by hand. `flexible_4310` is the exception: its URDF is an ONShape export.

Usage:
    python i2rt/robot_models/scripts/mjcf_to_urdf.py i2rt/robot_models/gripper/linear_4310/linear_4310.xml ...

The URDF's root link `gripper` is the MJCF's world frame, which is the arm's terminal `gripper` link (the
last joint's child in every arm URDF), where `combine_arm_and_gripper_xml` mounts the gripper. The MJCF
root body's own pos/quat is folded into everything under it, so the links sit on an arm's `gripper` link as is.

Mapping:
- body -> link, with the inertia as a full tensor in the link frame
- slide / hinge joint -> prismatic / revolute, with the axis in the child frame as in MJCF
- `<equality><joint>` (joint1 = c0 + c1 * joint2) -> `<mimic>` on joint2
- site -> a fixed child link with the site's name
- mesh geom -> `<visual>`, plus `<collision>` when the geom collides in MJCF
URDF has no contact exclusions, so `<contact>` is dropped. URDF requires effort and velocity limits, and the
MJCFs carry none, so both are 1, as in the arm URDFs.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import tyro

_JOINT_TYPES = {"slide": "prismatic", "hinge": "revolute"}


def _vec(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rpy(rot: np.ndarray) -> np.ndarray:
    """URDF rpy of ``rot``, which is Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pitch = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
    if abs(rot[2, 0]) < 1 - 1e-12:
        return np.array([np.arctan2(rot[2, 1], rot[2, 2]), pitch, np.arctan2(rot[1, 0], rot[0, 0])])
    return np.array([0.0, pitch, np.arctan2(-rot[0, 1], rot[1, 1])])  # gimbal lock: put it all in yaw


def _local(el: ET.Element) -> np.ndarray:
    """``el``'s 4x4 pose in its body, from pos/quat (the only orientation the gripper MJCFs use)."""
    unsupported = {"euler", "axisangle", "xyaxes", "zaxis", "fullinertia"} & el.attrib.keys()
    assert not unsupported, f"<{el.tag} name={el.get('name')}>: {sorted(unsupported)} is not supported"
    pose = np.eye(4)
    pose[:3, :3] = _quat_to_mat(_vec(el.get("quat"), "1 0 0 0"))
    pose[:3, 3] = _vec(el.get("pos"), "0 0 0")
    return pose


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{0.0 if abs(v) < 1e-10 else v:.9g}" for v in values)


def _origin(parent: ET.Element, pose: np.ndarray) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt(pose[:3, 3]), rpy=_fmt(_rpy(pose[:3, :3])))


def _fixed_joint(robot: ET.Element, name: str, parent: str, child: str, pose: np.ndarray) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    _origin(joint, pose)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)


def convert(mjcf_path: str) -> str:
    """The URDF text for the gripper MJCF at ``mjcf_path``."""
    mjcf = ET.parse(mjcf_path).getroot()
    assert mjcf.find("default") is None, "<default> classes are not supported"
    compiler = mjcf.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_files = {m.get("name"): os.path.join(meshdir, m.get("file")) for m in mjcf.iter("mesh")}

    mimics = {}  # joint2 -> (joint1, multiplier, offset)
    for eq in mjcf.iterfind("equality/joint"):
        c0, c1, *rest = _vec(eq.get("polycoef"), "0 1 0 0 0")
        assert eq.get("joint2") and c1 != 0 and not any(rest), f"unsupported joint equality {eq.attrib}"
        mimics[eq.get("joint2")] = (eq.get("joint1"), 1 / c1, -c0 / c1)

    robot = ET.Element("robot", name=mjcf.get("model"))

    def add_link(body: ET.Element, link: str, pose: np.ndarray) -> None:
        """Add ``body`` as ``link``; ``pose`` is the body's frame in the link's frame."""
        link_el = ET.SubElement(robot, "link", name=link)
        inertial = body.find("inertial")
        if inertial is not None:
            com = pose @ _local(inertial)
            tensor = com[:3, :3] @ np.diag(_vec(inertial.get("diaginertia"), "")) @ com[:3, :3].T
            inertial_el = ET.SubElement(link_el, "inertial")
            ET.SubElement(inertial_el, "origin", xyz=_fmt(com[:3, 3]), rpy="0 0 0")
            ET.SubElement(inertial_el, "mass", value=inertial.get("mass"))
            (ixx, ixy, ixz), (_, iyy, iyz), (_, _, izz) = tensor
            ET.SubElement(
                inertial_el,
                "inertia",
                **dict(
                    zip(
                        ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"],
                        _fmt([ixx, ixy, ixz, iyy, iyz, izz]).split(),
                        strict=True,
                    )
                ),
            )
        for geom in body.findall("geom"):
            assert geom.get("type") == "mesh", f"{link}: only mesh geoms are supported"
            collides = geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0"
            for tag in ("visual", "collision") if collides else ("visual",):
                el = ET.SubElement(link_el, tag)
                _origin(el, pose @ _local(geom))
                ET.SubElement(ET.SubElement(el, "geometry"), "mesh", filename=mesh_files[geom.get("mesh")])
                if tag == "visual":
                    rgba = geom.get("rgba", "0.5 0.5 0.5 1")
                    material = ET.SubElement(el, "material", name="rgba_" + rgba.replace(" ", "_"))
                    ET.SubElement(material, "color", rgba=rgba)

        for site in body.findall("site"):
            ET.SubElement(robot, "link", name=site.get("name"))
            _fixed_joint(robot, f"{site.get('name')}_joint", link, site.get("name"), pose @ _local(site))

        for child in body.findall("body"):
            joint_pose = pose @ _local(child)
            joints = child.findall("joint")
            assert len(joints) <= 1, f"{child.get('name')}: more than one joint"
            if not joints:
                _fixed_joint(robot, f"{child.get('name')}_joint", link, child.get("name"), joint_pose)
            else:
                (joint,) = joints
                assert np.allclose(_vec(joint.get("pos"), "0 0 0"), 0), f"{joint.get('name')}: pos must be 0"
                assert joint.get("range"), f"{joint.get('name')}: needs a range"
                joint_el = ET.SubElement(
                    robot, "joint", name=joint.get("name"), type=_JOINT_TYPES[joint.get("type", "hinge")]
                )
                _origin(joint_el, joint_pose)
                ET.SubElement(joint_el, "parent", link=link)
                ET.SubElement(joint_el, "child", link=child.get("name"))
                ET.SubElement(joint_el, "axis", xyz=_fmt(_vec(joint.get("axis"), "0 0 1")))
                lower, upper = _vec(joint.get("range"), "")
                ET.SubElement(joint_el, "limit", effort="1", velocity="1", lower=_fmt([lower]), upper=_fmt([upper]))
                if joint.get("name") in mimics:
                    other, multiplier, offset = mimics[joint.get("name")]
                    ET.SubElement(joint_el, "mimic", joint=other, multiplier=_fmt([multiplier]), offset=_fmt([offset]))
            add_link(child, child.get("name"), np.eye(4))

    (root,) = mjcf.findall("worldbody/body")
    add_link(root, root.get("name"), _local(root))

    ET.indent(robot, space="    ")
    header = (
        f"<!-- Generated from {os.path.basename(mjcf_path)} by i2rt/robot_models/scripts/mjcf_to_urdf.py. Edit the\n"
        "     MJCF and re-run the script instead of editing this file. The root link `gripper` is the arm's\n"
        "     `gripper` mount link, so these links sit on an arm URDF's `gripper` link as is. -->\n"
    )
    return '<?xml version="1.0"?>\n' + header + ET.tostring(robot, encoding="unicode") + "\n"


def main(mjcf: tyro.conf.Positional[list[str]]) -> None:
    """Write a URDF next to each gripper MJCF in ``mjcf``."""
    for mjcf_path in mjcf:
        urdf_path = os.path.splitext(mjcf_path)[0] + ".urdf"
        with open(urdf_path, "w") as f:
            f.write(convert(mjcf_path))
        print(f"wrote {urdf_path}")


if __name__ == "__main__":
    tyro.cli(main)
