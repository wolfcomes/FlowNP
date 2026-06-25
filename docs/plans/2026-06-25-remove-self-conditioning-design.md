# Remove Self-Conditioning Design

**Date:** 2026-06-25

## Goal

Remove the self-conditioning feature from the source tree and first-upload configuration surface before publishing `FlowNP` to GitHub.

## Scope

This cleanup removes:
- the `SelfConditioningResidualLayer` implementation
- the self-conditioning import and constructor option in `CTMCVectorField`
- the conditional forward-pass logic that performs self-conditioning
- the `prev_dst_dict` plumbing that only supported self-conditioning
- the `--self_conditioning` command-line override
- `self_conditioning` keys from the checked-in config files

This cleanup does not modify historical checkpoint directories, because those local artifacts should remain ignored and outside the GitHub upload.

## Approach

Use a full source-level removal rather than keeping a disabled compatibility switch. The public repository should not advertise a configuration option or code path that is no longer part of the project direction.

## Expected Result

After the cleanup:
- `src/models/self_conditioning.py` no longer exists
- `src/models/vector_field.py` has no `self_conditioning` or `prev_dst_dict` references
- `src/model_utils/sweep_config.py` no longer registers or merges `--self_conditioning`
- `configs/*.yaml` no longer contain `self_conditioning`
- targeted tests and syntax checks pass

## Verification

Use a focused repository cleanup test to scan tracked source/config paths for removed tokens, then run Python syntax checks on the directly edited Python files.
