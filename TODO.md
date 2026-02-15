# TODO

## NOW
- [ ] Fill real values for RunPod profile files (`Runpod_Comfy/profiles/*.example.yaml` copies)
- [ ] Compile selected profile and verify generated lock files
- [ ] Build and push first RunPod image

## NEXT
- [ ] Deploy image on RunPod and run smoke test workflow
- [ ] Add helper script: `Runpod_Comfy/tools/build_from_lock.sh`
- [ ] Add profile validation checks (schema-lite)

## BLOCKED
- [ ] Final production model URLs and checksums (waiting on selection)

## DONE
- [x] Add profile compiler scaffold (`compile_profile.sh`)
- [x] Add example profile files (runpod/nodes/models + urls format)
- [x] Wire Dockerfile to configurable base image via build-arg
- [x] Generate lock files from sample profile and sanity-check JSON
