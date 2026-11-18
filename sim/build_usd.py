"""Build the robot USD from the vendored URDFs (sim/robot/duo_<variant>.urdf) with Isaac Sim's URDF importer.

    python sim/build_usd.py                  # both variants -> sim/robot/usd/{actual,upright}/ (git-ignored)

Run once per machine: the generated USD stores absolute mesh paths, which is why it is not committed.
  actual  = arms mounted tilted +-45 deg about x (the real rig)     upright = arms not tilted
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=["actual", "upright"], choices=["actual", "upright"])
a = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

ROBOT = Path(__file__).resolve().parent / "robot"
for v in a.variants:
    cfg = URDFImporterConfig(urdf_path=str(ROBOT / f"duo_{v}.urdf"), usd_path=str(ROBOT / "usd" / v), merge_fixed_joints=True,
                             fix_base=True, collision_from_visuals=False, allow_self_collision=False, joint_target_type="position")
    print("IMPORT", v, URDFImporter(cfg).import_urdf(), flush=True)
app.close()
