# Depth-Aware Image-to-Video: Occlusion & Temporal Consistency (ELEC4542)

Read **CONVENTIONS.md** before writing any code.

## Layout
```
dvid/
  contracts.py    # resolution constants + shape/dtype checks (shared)
  types.py        # Scene dataclass (shared)
  geometry.py     # Member 1: scene_from_rgbd, render, add_to_scene   (stubs)
  inpainting.py   # Member 2: inpaint (stub), selective_composite (done)
  pipeline.py     # run_sequence(scene, poses, condition): conditions A-D
tests/
  test_interfaces.py   # contract tests: must pass before merging to main
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Workflow
1. `git checkout -b m1-geometry` (Member 1) or `git checkout -b m2-inpainting` (Member 2).
2. Replace a stub's body with the real implementation. **Don't change its signature or remove its `check_*` calls.**
3. Run `pytest -q`. Add tests for your own module as you go.
4. Open a pull request to `main` and have your teammate review it.

Some tests check behaviour the stubs only fake (e.g. no holes at the identity pose). Once real code replaces a stub, those tests start checking real behaviour.
