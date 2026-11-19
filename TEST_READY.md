# E2E Test Suite Ready

## Test Runner
- Command: `python -m pytest tests/`
- Diagnostics Runner: `python tests/test_fault_diagnostics.py`
- Crowd Stress Runner: `python tests/test_crowd_stress.py`
- Expected: All tests pass with exit code 0 (54/54 passed in repository).

## Coverage Summary
| Tier | Count | Description |
|---|---:|---|
| Tier 1: Feature & Crowd Density Stress | 24 | Evaluates 15, 30, 60 faces simultaneously, Rows 1-8 (1m-10m), 16-25px bboxes, occlusions (10%-40%), lighting (backlight, underexposure, overexposure), and batched ArcFace throughput |
| Tier 2: Fault Diagnostics (V01-V15) | 26 | Systematically asserts baseline fault conditions and verifies production remediations across detector, recognizer, attendance logic, anti-spoofing, and config |
| Tier 3: Enterprise Security & Isolation | 4 | Database HMAC-SHA256 integrity, approach A isolation, and administrative portal verification |
| **Total Test Cases** | **54** | **100% automated pass rate** |

## Feature Checklist
| Feature | Stress Test | Fault Diagnostic | Unit Regression |
|---|:---:|:---:|:---:|
| F01: Crowd Density (15, 30, 60) | ✓ | - | ✓ |
| F02: Distance Degradation Rows 1-8 (1m-10m) | ✓ | ✓ | ✓ |
| F03: Mutual Occlusion (10%, 25%, 40%) | ✓ | - | - |
| F04: Invariant Yaw Ratio (V01) | - | ✓ | ✓ |
| F05: Multi-Scale SCRFD det_size (V02) | ✓ | ✓ | ✓ |
| F06: Affine Transform Fallback (V03) | - | ✓ | ✓ |
| F07: Enrolled Roster Candidate Filtering (V04) | - | ✓ | ✓ |
| F08: Distinct Shot Voting Quorum (V05) | ✓ | ✓ | ✓ |
| F09: True 4D Batch Tensor Inference (V06) | ✓ | ✓ | ✓ |
| F10: Config Defaults Alignment (V07) | - | ✓ | ✓ |
| F11: Clean Frame Warping (V08) | - | ✓ | ✓ |
| F12: Absent Records Persistence (V09) | - | ✓ | ✓ |
| F13: Spectacle Glare & Chroma Bias (V10, V11) | - | ✓ | ✓ |
| F14: Scale-Adaptive Anti-Spoof (V12) | ✓ | ✓ | ✓ |
| F15: Windows DirectShow Capture (V13) | - | ✓ | ✓ |
