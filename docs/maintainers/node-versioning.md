# Node versioning — how to change a node without breaking saved workflows

> **⚠️ Scope — OPT-IN for Koolook nodes, MANDATORY for forks.**
> Backward compatibility is **not** a default constraint for Koolook-created
> nodes (the root `k_*.py` custom nodes); for those, **ignore everything below**
> and make the cleanest change — rename/drop/reorder freely — *unless* the
> maintainer says `check backward compatibility`. **For anything under
> `forks/`** (upstream ports/wrappers, including the Koolook-original nodes
> exposed via `SKIP_VERSION_SUFFIX`) **these rules still apply by default** —
> forks keep full back-compat discipline with no request needed. See
> [`CLAUDE.md`](../../CLAUDE.md) → *Change management*.

This is a hard-won list. ComfyUI's workflow format silently breaks in
several non-obvious ways when you change a node's inputs, outputs, or
class name. The rules below are what to follow whenever you touch a
**fork** node's `INPUT_TYPES` / `RETURN_TYPES` / `NODE_CLASS_MAPPINGS`,
or a **Koolook** node's once `check backward compatibility` has been
requested.

## How ComfyUI loads a saved workflow

When the frontend opens a workflow JSON, it reconstructs each node by:

1. **Looking up the node by its registered ID** (the key in
   `NODE_CLASS_MAPPINGS`). If the lookup fails, the node renders as a
   red "missing" stub and the workflow won't run.
2. **Mapping connections by input/output name.** Most cases.
3. **Mapping widget values by *position* in the widget array** — this is
   the source of most breakage.
4. **Filling missing inputs with their default**, if they're declared in
   `INPUT_TYPES["optional"]`. Missing *required* inputs → validation
   error.

## What breaks vs what doesn't

| Change | Breaks saved workflows? | Why |
|---|---|---|
| Add a new input to `INPUT_TYPES["required"]` | **YES** | Old workflow has no value for it → validation fails |
| Add a new input to `INPUT_TYPES["optional"]` *at the end* | **safe** | Missing inputs use defaults |
| Add a new widget *between* existing widgets | **YES** | Widget array indices shift → values land in wrong widgets |
| Add a new widget *at the end* | mostly safe | Old workflows just don't have that index → uses default |
| Rename an existing input | **YES** | Saved value can't map by name |
| Rename the class / node ID | **YES** | Lookup fails entirely |
| Reorder existing combo options | **YES** for indexed serialization | Some workflows store combo as index, not string |
| Add new combo option *at the end* | **safe** | Existing string values still match |
| Remove an input | soft break | Saved value silently dropped, may lose semantic meaning |
| Change `RETURN_TYPES` order or types | **YES** | Downstream connections targeting `output[1]` now hit wrong type |
| Change a default value | "stealth break" | Old workflows have the old default baked in; new workflows get the new default — silent behavior divergence |

> "*Field gets messed up*" almost always = **widget position shift**.
> ComfyUI serializes widgets as `widgets_values: [val0, val1, val2]`.
> Insert a new widget at index 1 → every subsequent value shifts one
> slot → wrong values land in wrong widgets.

## The five rules

1. **Never rename a registered node ID.** When you really must, register
   both old and new IDs pointing to the same class for a few releases
   (deprecation alias), then remove the old one with a CHANGELOG note.
2. **Append new inputs/widgets to the END.** Never insert in the middle.
   Order in `INPUT_TYPES` dict matters because Python preserves insertion
   order and the frontend uses it.
3. **New inputs go in `INPUT_TYPES["optional"]` with a default**, not in
   `required`. Old workflows that don't have the field will use the
   default; new workflows can wire it up.
4. **Never reorder combo options.** Append new options. If you need to
   remove one, leave it as a hidden alias or deprecation stub for a
   release first.
5. **Treat `RETURN_TYPES` as immutable.** If you need to change outputs,
   make a new node ID — `MyNodeV2`. The Comfy ecosystem (KJ Nodes, Was
   Node Suite, Crystools) does this constantly.

## When you can't avoid a breaking change

| Pattern | Example |
|---|---|
| **Suffix-version a new node** | `EasyAIPipeline` → `EasyAIPipelineV2`. Both register, both work. |
| **Mark old as deprecated in display name** | Old: `Easy AI Pipeline (deprecated, use V2)` — display name change is fine, ID is what matters |
| **Document migration in CHANGELOG** | "Old `EasyAIPipeline` retained; new pipeline work should use `EasyAIPipelineV2` which adds X/Y/Z" |
| **Keep deprecated nodes for ≥2 minor versions** before removing |

This is exactly the pattern used for `EasyResize` → `EasyResize_Koolook`
in v0.1.6 (see [`../../CHANGELOG.md`](../../CHANGELOG.md)). Both IDs
register, both load, the bare-name one's display says "deprecated", and
the bare-name one will be removed in a future major release once the
deprecation has had time to propagate.

## Concrete pattern — the rename + alias

```python
# In whichever k_*.py file holds the class:

class MyNode:
    """ ... existing implementation, unchanged ... """
    # ...

# Register BOTH IDs against the same class:
NODE_CLASS_MAPPINGS = {
    "MyNode_Koolook": MyNode,           # new canonical ID
    "MyNode": MyNode,                    # legacy alias (deprecated)
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode_Koolook": "My Node (Koolook)",
    "MyNode": "My Node (deprecated, use 'My Node (Koolook)')",
}
```

That's it. Saved workflows that reference `MyNode` keep loading. New
workflows pick `MyNode_Koolook` from the search. The bare-name display
nudges users to switch.

## Concrete pattern — adding a new optional input

Suppose `EasyAIPipeline` currently has these inputs:

```python
"required": {
    "shot_duration": ("INT", {...}),
    "seed": ("INT", {...}),
    "base_path": ("STRING", {...}),
    "shot_name": ("STRING", {...}),
    "ai_method": ("STRING", {...}),
    "version": ("INT", {...}),
},
```

You want to add a new `pass_name` field. **Do not** add it to `required`
between existing fields — that breaks every saved workflow. Instead:

```python
"required": { ... unchanged ... },
"optional": {
    "pass_name": ("STRING", {"default": "", "tooltip": "..."}),
},
```

- Old workflows that don't have `pass_name` use the default `""`.
- New workflows that do have it use the value the user sets.
- Output behavior must remain backward-compatible when `pass_name=""`
  (i.e. equivalent to the pre-change behavior). If `pass_name=""` would
  change the output filename, you need a `MyNodeV2` instead.

## Concrete pattern — output change

Suppose `EasyAIPipeline` currently returns `(output_path,)` and you want
to add a `(output_path, output_dir)` tuple. That's a `RETURN_TYPES`
change → downstream connections targeting `output[1]` would hit the new
output unexpectedly (or just exist where they didn't before).

Don't modify the existing class. Make a new one:

```python
class EasyAIPipeline_V2(EasyAIPipeline):
    """Same as EasyAIPipeline, plus emits the resolved output directory."""
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "output_dir")
    # ... reuse or override the execute method as needed ...

NODE_CLASS_MAPPINGS = {
    "EasyAIPipeline": EasyAIPipeline,           # unchanged
    "EasyAIPipeline_V2": EasyAIPipeline_V2,     # new
}
```

## Anti-patterns to refuse — *for forks, or after `check backward compatibility`*

These are anti-patterns for **any fork node**, and for a **Koolook node only
once the maintainer has asked** to preserve backward compatibility. For a
Koolook node with no such request, every one of them is the *correct* clean
change — append-vs-insert is the lone exception below.

- ❌ "I'll just add the field in the middle for clarity, the dict order
  doesn't matter." It does. Append. *(This one is worth following even
  without a back-compat request — a mid-list widget insertion silently
  scrambles the `widgets_values` of any workflow already on disk, including
  your own dev graphs, for zero benefit. Appending is free.)*
- ❌ "I'll rename the input to be clearer." For a fork node, or any node
  under a back-compat request: add a new optional input with the clearer
  name and deprecate the old one in display only. For a Koolook node with
  no request, **just rename it.**
- ❌ "It's a small change, no need to bump the node version." For a fork
  node, or under a back-compat request, a surface change that could break a
  saved workflow → new node ID. For a Koolook node with no request, change
  in place and note it in the CHANGELOG.
- ❌ "Nobody's using the old node yet." For a fork node, or under a
  back-compat request, you can't assume that. For a Koolook node with no
  request you *do* assume it — clean change, move on.

## Tooling — what we have, what's still missing

- ✅ **Workflow-fixture smoke test** — `tools/preflight_release.py`
  check `workflows` walks `tests/workflows/*.json`, extracts node IDs
  by both heuristic and AST-extracted set, and reports any reference
  that no longer resolves. The `preflight-release` skill in
  `.claude/skills/` runs all four checks before a release.
- ⏳ **`node-api-change` agent skill** — a check that would run before
  any edit to `INPUT_TYPES` / `RETURN_TYPES` / class names, diff
  against the rules above, and refuse unsafe changes. Same shape as
  `license-pre-check`. Not built yet; scoped for the next time someone
  changes a node's input/output surface.
