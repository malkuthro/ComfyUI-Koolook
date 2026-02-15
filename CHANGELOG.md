# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog and SemVer.

## [Unreleased]

### Added
- Initial release/process scaffolding:
  - `docs/RELEASE_WORKFLOW.md`
  - `docs/REPO_AUDIT_2026-02-14.md`
  - `third_party/THIRD_PARTY.md`
  - `third_party/forks_manifest.yaml`
  - `tools/report_mods.py`
- Contribution/release templates:
  - `.github/PULL_REQUEST_TEMPLATE.md`
  - `.github/ISSUE_TEMPLATE/release_checklist.md`
  - `docs/RELEASE_NOTES_TEMPLATE.md`
- RunPod_Comfy implementation scaffold:
  - `Runpod_Comfy/README.md`
  - `Runpod_Comfy/setup/USER_README.md`
  - `Runpod_Comfy/core/docker/Dockerfile`
  - `Runpod_Comfy/core/docker/start.sh`
  - `Runpod_Comfy/core/scripts/bootstrap.sh`
  - `Runpod_Comfy/core/scripts/install_custom_nodes.sh`
  - `Runpod_Comfy/core/scripts/download_models.sh`
  - `Runpod_Comfy/image/active/comfyui.lock`
  - `Runpod_Comfy/image/active/custom_nodes.lock.json`
  - `Runpod_Comfy/image/active/models.json`
  - `Runpod_Comfy/image/DEPLOY_RUNPOD.md`

## [0.1.0] - 2026-02-14

### Added
- Project-level release/versioning strategy and third-party tracking scaffold.
