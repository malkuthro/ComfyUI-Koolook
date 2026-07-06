# Koolook Publish contract nodes

| Aspect | Value |
|---|---|
| Display names | `Koolook Publish Input`, `Koolook Publish Output`, `Koolook Publish Router`, `Koolook Publish Result` |
| Node IDs | `Koolook_PublishInput`, `Koolook_PublishOutput`, `Koolook_PublishRouter`, `Koolook_PublishResult` |
| ComfyUI category | `Koolook/Publish` |
| Source | Koolook-native |
| Source file | [`k_publish_contract.py`](../../../../k_publish_contract.py) |

## What they do

These nodes mark which parts of a saved workflow should become controls in a
published setup. Use them when a workflow should be callable from another app
through the Koolook published-setup API instead of being edited directly on the
ComfyUI canvas.

The nodes are not image processors. They are contract markers:

- `Koolook_PublishInput` declares source inputs such as an EXR sequence folder,
  QuickTime file, single image/file, or prompt.
- `Koolook_PublishOutput` declares output folder/name/version controls, plus an
  optional independent **output type** (so an EXR source can write a QT movie).
- `Koolook_PublishRouter` connects the selected mode to matching writer
  branches so only the chosen branch runs.
- `Koolook_PublishResult` optionally reports the final resolved path or status
  string back to the external app.

## Required group layout

Put the nodes inside named ComfyUI groups before publishing the saved workflow:

| Group name | Put inside |
|---|---|
| `Koolook Input` | One `Koolook_PublishInput` node. Nearby source/helper nodes may overlap the group for review. |
| `Koolook Output` | One `Koolook_PublishOutput` node. Add `Koolook_PublishRouter` when the setup has mode-specific writer branches. Add `Koolook_PublishResult` when the app should display a custom resolved result. |

The publisher records nearby nodes for review, but only recognized
`Koolook_Publish*` nodes define the app-facing form.

## Node roles

### `Koolook_PublishInput`

| Field/output | Meaning |
|---|---|
| `mode` | Source mode: `EXR`, `QT`, `Img`, or `Prompt`. |
| `sequence_folder` | Directory path for image-sequence sources. |
| `qt_file` | Full path to one QuickTime/video file. |
| `single_file` | Full path to one still image or single source file. |
| `prompt` | Prompt text for setups that expose a prompt field. |
| `switch` | Integer mode index used by the workflow to select matching branches. |

### `Koolook_PublishOutput`

| Field/output | Meaning |
|---|---|
| `folder` | Destination folder for generated outputs. |
| `name` | Base output name. |
| `version` | Version token or number. |
| `output_mode` | Output type: `Same as input`, `EXR`, `QT`, or `Img`. `Same as input` (default) follows the input type; a concrete choice writes a different type than the source. `Prompt` is not an output format. |
| `input_switch` (input) | Wire `Koolook_PublishInput.switch` here. Used only when `output_mode` is `Same as input`, so the output type mirrors the chosen input type. |
| `switch` (output) | Resolved output-type index. Wire into `Koolook_PublishRouter.selector` to drive the writer branch. |

### `Koolook_PublishRouter`

Wire the workflow's main payload into `payload`, and wire the `selector` from
**one of two switches** depending on whether output type should follow input
type or be chosen independently:

```text
# Output type follows input type (simplest):
Koolook_PublishInput.switch  -> Koolook_PublishRouter.selector

# Output type chosen independently of the source (EXR in, QT out):
Koolook_PublishInput.switch  -> Koolook_PublishOutput.input_switch
Koolook_PublishOutput.switch -> Koolook_PublishRouter.selector

workflow payload -> Koolook_PublishRouter.payload

Koolook_PublishRouter.EXR -> EXR writer branch
Koolook_PublishRouter.QT -> video writer branch
Koolook_PublishRouter.Img -> image writer branch
Koolook_PublishRouter.Prompt -> prompt/no-op branch if needed
```

When the setup runs from the external app, Koolook keeps the selected writer
branch and prunes the unselected branches from the queued prompt. When the
router is driven by `Koolook_PublishOutput.switch`, that pruning follows the
independent **Output type** control instead of the input type.

### `Koolook_PublishResult`

Use this when the external app should show a specific resolved string after the
workflow runs, such as the selected output path. If omitted, the runner still
uses available writer/history output items.

## Minimal recipe

1. Save the workflow in the Kforge Labs Workflows tree.
2. Add a group named `Koolook Input` around `Koolook_PublishInput`.
3. Add a group named `Koolook Output` around `Koolook_PublishOutput` and any
   router/result nodes.
4. Wire the publish input/output values into the real loader, path, and writer
   nodes used by the workflow.
5. Right-click the saved workflow and choose **Publish setup...**.
6. Review the publish dialog before confirming.

For API and maintainer-level details, see
[`published-setups.md`](../../../maintainers/published-setups.md) and
[`published-setup-external-ui-contract.md`](../../../maintainers/published-setup-external-ui-contract.md).
