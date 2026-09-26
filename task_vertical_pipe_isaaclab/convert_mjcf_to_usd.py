"""
==============================================================================
convert_mjcf_to_usd.py - MJCF -> USD for the vertical-pipe task (step 3 of the port)
==============================================================================
    D:\\Isaacsim\\env_isaaclab\\Scripts\\python.exe convert_mjcf_to_usd.py [--mesh-dir DIR] [--no-visual]

1. Builds the MJCF that the MuJoCo env actually simulates. vertical_pipe_env.py edits
   urdf/ContinuumRobot_Native.xml in code (MjSpec): the collision copies of the meshes
   are removed, one collision cylinder per segment is added, 31 position servos are
   added and armature / damping are set on every DOF. The same edits are written into
   assets/generated/mjcf/continuum_{physics,visual}.xml (a jointless "base_link" body
   is added as the fixed articulation root, which does not change the MuJoCo model).
2. Converts both files with the Isaac Sim MJCF importer (isaaclab.sim.converters.MjcfConverter):
     continuum_physics.usd  - colliders only (training, like the MuJoCo env with visual=False)
     continuum_visual.usd   - + the CAD meshes as visuals (play / video)
3. Checks the USD against the MJCF (bodies, inertia, joints, limits, colliders) and prints
   what the importer does not carry over; the env sets those in Isaac Lab (see README).
4. Writes the pipe assets pipe_rig.usd / pipe_random.usd (24 box staves on one kinematic
   rigid body, like the mocap "pipe" body of the MuJoCo env).
==============================================================================
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)  # Kit exits with os._exit: keep prints when stdout is a file
import xml.etree.ElementTree as ET

import numpy as np

# Windows: load torch / tensordict DLLs before Kit starts (avoids access violations)
import torch  # noqa: F401

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

import constants as C  # noqa: E402

GEN_MJCF_DIR = os.path.join(C.GENERATED_DIR, "mjcf")
GEN_USD_DIR = os.path.join(C.GENERATED_DIR, "usd")


# ---------------------------------------------------------------------------
# 1. MJCF of the simulated model
# ---------------------------------------------------------------------------
def _fmt(values):
    return " ".join(f"{v:.10g}" for v in values)


def build_import_mjcf(src, dst, visual, mesh_dir):
    tree = ET.parse(src)
    root = tree.getroot()
    root.set("model", "continuum_visual" if visual else "continuum_physics")

    # <option> and <default> = VerticalPipeEnv._configure_model
    compiler = root.find("compiler")
    idx = list(root).index(compiler) + 1
    root.insert(idx, ET.Element("option", timestep=f"{C.SIM_DT}", gravity="0 0 0", integrator="implicitfast"))
    default = ET.Element("default")
    ET.SubElement(default, "joint", armature=f"{C.JOINT_ARMATURE}", damping=f"{C.JOINT_DAMPING}")
    root.insert(idx + 1, default)

    # meshes: training drops them, visual keeps the contype=0 copies (the others collide in the XML).
    # The importer wants mesh paths relative to the MJCF and writes temporary files next to the
    # meshes, so they are hard-linked (copied if that fails) into assets/generated/mjcf/meshes.
    asset = root.find("asset")
    if visual:
        local_dir = os.path.join(os.path.dirname(dst), "meshes")
        os.makedirs(local_dir, exist_ok=True)
        for mesh in asset.findall("mesh"):
            name = mesh.get("file")
            target = os.path.join(local_dir, name)
            if not os.path.exists(target):
                try:
                    os.link(os.path.join(mesh_dir, name), target)
                except OSError:
                    import shutil
                    shutil.copyfile(os.path.join(mesh_dir, name), target)
            mesh.set("file", f"meshes/{name}")
    else:
        root.remove(asset)
    for parent in root.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("type") == "mesh" and (not visual or geom.get("contype") != "0"):
                parent.remove(geom)

    # jointless root body (welded to the world) holding the frame geoms and the elevator
    worldbody = root.find("worldbody")
    base = ET.Element("body", name="base_link")
    ET.SubElement(base, "inertial", pos="0 0 0.95", mass="1", diaginertia="0.01 0.01 0.01")
    for child in list(worldbody):
        worldbody.remove(child)
        base.append(child)
    worldbody.append(base)

    # one collision cylinder per segment (VerticalPipeEnv._prepare_robot)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for i in range(1, 16):
        radius, z0, z1 = C.SEG_COLLIDERS[i]
        ET.SubElement(bodies[f"Seg{i}"], "geom", name=f"Seg{i}_col", type="cylinder",
                      size=_fmt([radius, 0.5 * (z1 - z0)]), pos=_fmt([0.0, 0.0, 0.5 * (z0 + z1)]),
                      contype="1", conaffinity="0", group="3", density="0", friction="0.3 0.005 0.0001",
                      rgba="1 0.55 0.1 0.35")

    # 30 bending servos + elevator servo: gain kp, bias (0, -kp, -kv) == <position kp kv>
    actuator = ET.SubElement(root, "actuator")
    for i in range(1, 16):
        for axis in ("x", "y"):
            ET.SubElement(actuator, "position", name=f"Seg{i}_{axis}", joint=f"Seg{i}_{axis}",
                          kp=f"{C.BEND_KP:g}", kv=f"{C.BEND_KV:g}")
    ET.SubElement(actuator, "position", name="Elevator", joint="Elevator_Joint",
                  kp=f"{C.ELEV_KP:g}", kv=f"{C.ELEV_KV:g}")

    ET.indent(tree, space="  ")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    return dst


# ---------------------------------------------------------------------------
# 2b. Fixes on top of the importer output (authored in the root layer)
# ---------------------------------------------------------------------------
def postprocess_robot_usd(path, visual):
    """
    - the importer adds an empty /worldBody prim carrying a second ArticulationRootAPI; Isaac Lab
      needs exactly one articulation root -> deactivate it (the fixed joint to the world is kept)
    - visuals / collisions are instanceable; instance proxies cannot be edited, so the collider
      properties set by the env (contact / rest offset) would be ignored -> de-instance them
    - the importer also makes a visual copy of every collision cylinder (MuJoCo hides them in
      geom group 3); in the CAD-mesh USD they are hidden
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    world_body = stage.GetPrimAtPath(root.GetPath().AppendChild("worldBody"))
    if world_body.IsValid():
        world_body.SetActive(False)
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    if visual:
        for prim in stage.Traverse():
            if prim.GetName().endswith("_col") and "/visuals/" in str(prim.GetPath()):
                UsdGeom.Imageable(prim).MakeInvisible()
    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------------
# 3. USD report
# ---------------------------------------------------------------------------
def inspect_robot_usd(path, report):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = Usd.Stage.Open(path)
    bodies, joints, colliders, roots, filtered = [], [], [], [], 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            bodies.append((prim.GetPath().pathString, mass.GetMassAttr().Get(), mass.GetDiagonalInertiaAttr().Get()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath().pathString)
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            info = {"path": prim.GetPath().pathString, "type": prim.GetTypeName(),
                    "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                    "body1": [str(t) for t in j.GetBody1Rel().GetTargets()]}
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                typed = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(prim)
                info.update(axis=typed.GetAxisAttr().Get(), lower=typed.GetLowerLimitAttr().Get(),
                            upper=typed.GetUpperLimitAttr().Get())
                kind = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
                drive = UsdPhysics.DriveAPI.Get(prim, kind)
                if drive:
                    info.update(stiffness=drive.GetStiffnessAttr().Get(), damping=drive.GetDampingAttr().Get(),
                                max_force=drive.GetMaxForceAttr().Get())
                pj = PhysxSchema.PhysxJointAPI(prim)
                if pj:
                    info.update(armature=pj.GetArmatureAttr().Get())
            joints.append(info)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            desc = prim.GetTypeName()
            if prim.IsA(UsdGeom.Cylinder):
                cyl = UsdGeom.Cylinder(prim)
                centre = UsdGeom.Xformable(prim.GetParent()).GetLocalTransformation().ExtractTranslation()
                desc += (f" r={cyl.GetRadiusAttr().Get():.4f} h={cyl.GetHeightAttr().Get():.4f} "
                         f"axis={cyl.GetAxisAttr().Get()} centre_z={centre[2]:.4f}")
            colliders.append((prim.GetPath().pathString, desc))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            filtered += len(UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets())

    report.append(f"USD: {path}")
    report.append(f"  articulation roots: {roots}")
    report.append(f"  rigid bodies: {len(bodies)}   joints: {len(joints)}   colliders: {len(colliders)}   "
                  f"filtered collision pairs: {filtered}")
    for b in bodies[:4]:
        report.append(f"    body {b[0]}  mass={b[1]}  inertia={b[2]}")
    for j in joints:
        keys = ("type", "axis", "lower", "upper", "stiffness", "damping", "max_force", "armature")
        report.append("    joint " + j["path"].rsplit("/", 1)[-1] + ": " +
                      ", ".join(f"{k}={j[k]}" for k in keys if k in j) +
                      f"  [{','.join(p.rsplit('/', 1)[-1] for p in j['body0'])} -> "
                      f"{','.join(p.rsplit('/', 1)[-1] for p in j['body1'])}]")
    for c in colliders:
        report.append(f"    collider {c[0]}: {c[1]}")
    return bodies, joints, colliders


# ---------------------------------------------------------------------------
# 4. Pipe assets
# ---------------------------------------------------------------------------
def tube_mesh(r_in, r_out, z_bot, z_top, n=72):
    """Closed tube mesh (same as vertical_pipe_env.tube_mesh), visual only."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)

    def ring(r, z):
        return np.stack([r * c, r * s, np.full(n, z)], axis=1)

    verts, faces = [], []

    def add_band(ring_a, ring_b, flip):
        base = sum(len(v) for v in verts)
        verts.extend([ring_a, ring_b])
        for k in range(n):
            k2 = (k + 1) % n
            a, b, cc, d = base + k, base + k2, base + n + k2, base + n + k
            faces.extend([[a, cc, b], [a, d, cc]] if flip else [[a, b, cc], [a, cc, d]])

    add_band(ring(r_out, z_bot), ring(r_out, z_top), flip=False)
    add_band(ring(r_in, z_bot), ring(r_in, z_top), flip=True)
    add_band(ring(r_in, z_top), ring(r_out, z_top), flip=False)
    add_band(ring(r_in, z_bot), ring(r_out, z_bot), flip=True)
    return np.concatenate(verts), np.array(faces, dtype=np.int32)


def write_pipe_usd(path, pipe_mode):
    """Kinematic rigid body whose origin is the centre of the pipe's top opening (VerticalPipeEnv._add_pipe)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt, PhysxSchema

    bore = C.bore_length(pipe_mode)
    r_in, wall = C.PLATE_HOLE_RADIUS, C.PIPE_WALL
    half_width = (r_in + wall) * np.tan(np.pi / C.N_STAVES)

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    pipe = UsdGeom.Xform.Define(stage, "/Pipe")
    stage.SetDefaultPrim(pipe.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(pipe.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(pipe.GetPrim()).CreateMassAttr(1.0)

    # friction 0.3 (MuJoCo friction[0]); no restitution (MuJoCo soft contacts do not bounce)
    material = UsdShade.Material.Define(stage, "/Pipe/PhysicsMaterial")
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr(C.CONTACT_FRICTION)
    mat_api.CreateDynamicFrictionAttr(C.CONTACT_FRICTION)
    mat_api.CreateRestitutionAttr(0.0)

    UsdGeom.Scope.Define(stage, "/Pipe/staves")
    for i in range(C.N_STAVES):
        angle = 2.0 * np.pi * i / C.N_STAVES
        cube = UsdGeom.Cube.Define(stage, f"/Pipe/staves/stave_{i:02d}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d((r_in + 0.5 * wall) * np.cos(angle), (r_in + 0.5 * wall) * np.sin(angle), -0.5 * bore))
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, float(np.rad2deg(angle))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        xf.SetScale(Gf.Vec3f(wall, 2.0 * half_width, bore))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)       # colliders are not drawn
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.6, 0.9)])

    # visuals (no collision): tube + collars, as in the MuJoCo scene
    UsdGeom.Scope.Define(stage, "/Pipe/visuals")

    def add_tube(name, r0, r1, z0, z1, rgba):
        verts, faces = tube_mesh(r0, r1, z0, z1)
        mesh = UsdGeom.Mesh.Define(stage, f"/Pipe/visuals/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
        mesh.CreateDisplayOpacityAttr([rgba[3]])

    h = C.PIPE_HEIGHT
    add_tube("pipe_tube", r_in, r_in + wall, -h, 0.0, [0.55, 0.78, 1.0, 0.35])
    add_tube("pipe_collar_top", r_in + 0.0005, r_in + wall + 0.004, -0.006, 0.0006, [1.0, 0.78, 0.12, 1.0])
    add_tube("pipe_collar_bot", r_in + 0.0005, r_in + wall + 0.004, -h - 0.0006, -h + 0.006, [0.2, 0.85, 0.35, 1.0])
    stage.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MJCF -> USD for the continuum robot vertical-pipe task")
    parser.add_argument("--mjcf", default=C.SOURCE_MJCF, help="source MJCF (ContinuumRobot_Native.xml)")
    parser.add_argument("--mesh-dir", default=r"D:\mujoco\Continuum_MuJoCo\urdf",
                        help="folder with the .obj meshes referenced by the MJCF (visual USD only)")
    parser.add_argument("--no-visual", action="store_true", help="only build the collider-only USD")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    if not any(a.startswith("--/app/vulkan") for a in sys.argv):
        sys.argv.append("--/app/vulkan=false")
    args, _ = parser.parse_known_args()
    args.headless = True

    variants = {"continuum_physics": False}
    if not args.no_visual:
        if os.path.isdir(args.mesh_dir):
            variants["continuum_visual"] = True
        else:
            print(f"[WARN] mesh folder {args.mesh_dir} not found - skipping the visual USD")
    xmls = {}
    for name, visual in variants.items():
        xmls[name] = build_import_mjcf(args.mjcf, os.path.join(GEN_MJCF_DIR, f"{name}.xml"), visual, args.mesh_dir)
        print(f"[MJCF] {xmls[name]}")

    app = AppLauncher(args).app
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    report = []
    for name, xml in xmls.items():
        cfg = MjcfConverterCfg(asset_path=xml, usd_dir=os.path.join(GEN_USD_DIR, name), usd_file_name=f"{name}.usd",
                               fix_base=True, import_sites=False, force_usd_conversion=True, make_instanceable=False,
                               self_collision=False)
        usd_path = MjcfConverter(cfg).usd_path
        print(f"[USD] {usd_path}")
        postprocess_robot_usd(usd_path, variants[name])
        inspect_robot_usd(usd_path, report)

    for mode in C.PIPE_MODES:
        path = write_pipe_usd(os.path.join(GEN_USD_DIR, f"pipe_{mode}.usd"), mode)
        report.append(f"pipe ({mode}): {path}  bore {C.bore_length(mode) * 1000:.0f} mm, {C.N_STAVES} box staves")

    text = "\n".join(report)
    with open(os.path.join(C.GENERATED_DIR, "conversion_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    app.close()


if __name__ == "__main__":
    main()
