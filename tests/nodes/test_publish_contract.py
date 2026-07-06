"""Koolook publish contract nodes."""

from __future__ import annotations

from k_publish_contract import (
    Koolook_PublishInput,
    Koolook_PublishOutput,
    Koolook_PublishResult,
    Koolook_PublishRouter,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resolve_output_switch,
)


def test_publish_contract_nodes_register() -> None:
    assert NODE_CLASS_MAPPINGS["Koolook_PublishInput"] is Koolook_PublishInput
    assert NODE_CLASS_MAPPINGS["Koolook_PublishOutput"] is Koolook_PublishOutput
    assert NODE_CLASS_MAPPINGS["Koolook_PublishResult"] is Koolook_PublishResult
    assert NODE_CLASS_MAPPINGS["Koolook_PublishRouter"] is Koolook_PublishRouter
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishInput"] == "Koolook Publish Input"
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishOutput"] == "Koolook Publish Output"
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishResult"] == "Koolook Publish Result"
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishRouter"] == "Koolook Publish Router"


def test_publish_input_exposes_stable_fields_and_switch_output() -> None:
    spec = Koolook_PublishInput.INPUT_TYPES()["required"]

    assert list(spec) == ["mode", "sequence_folder", "qt_file", "single_file", "prompt"]
    assert spec["mode"][0] == ["EXR", "QT", "Img", "Prompt"]
    assert Koolook_PublishInput.RETURN_NAMES == (
        "sequence_folder",
        "qt_file",
        "single_file",
        "prompt",
        "switch",
    )

    result = Koolook_PublishInput().run(
        mode="Img",
        sequence_folder="/seq",
        qt_file="/movie.mov",
        single_file="/image.png",
        prompt="hidden",
    )

    assert result == ("/seq", "/movie.mov", "/image.png", "hidden", 2)


def test_publish_input_accepts_published_numeric_switch_values() -> None:
    node = Koolook_PublishInput()

    assert node.run(0, "/seq", "/movie.mov", "/image.png", "hidden")[-1] == 0
    assert node.run("1", "/seq", "/movie.mov", "/image.png", "hidden")[-1] == 1
    assert node.run(2, "/seq", "/movie.mov", "/image.png", "hidden")[-1] == 2
    assert node.run("3", "/seq", "/movie.mov", "/image.png", "hidden")[-1] == 3


def test_publish_output_exposes_stable_fields() -> None:
    spec = Koolook_PublishOutput.INPUT_TYPES()
    required = spec["required"]
    optional = spec.get("optional", {})

    assert list(required) == ["folder", "name", "version", "output_mode"]
    assert required["output_mode"][0] == ["Same as input", "EXR", "QT", "Img"]
    assert list(optional) == ["input_switch"]
    assert Koolook_PublishOutput.RETURN_NAMES == ("folder", "name", "version", "switch")

    # Default output_mode ("Same as input") passes the wired input switch through.
    assert Koolook_PublishOutput().run(
        folder="/out",
        name="mask",
        version="1",
        input_switch=1,
    ) == ("/out", "mask", "1", 1)


def test_publish_output_switch_resolution() -> None:
    """output_mode picks the writer branch; 'Same as input' follows input_switch."""
    # Concrete override by label -> that branch index (EXR=0, QT=1, Img=2).
    assert Koolook_PublishOutput().run("/o", "n", "1", "QT", 0)[-1] == 1
    assert Koolook_PublishOutput().run("/o", "n", "1", "EXR", 2)[-1] == 0
    # Concrete override by numeric index (external runner injects an int).
    assert Koolook_PublishOutput().run("/o", "n", "1", 2, 0)[-1] == 2
    # "Same as input" (and the Prompt index 3, which is not an output type)
    # both fall back to the input switch.
    assert Koolook_PublishOutput().run("/o", "n", "1", "Same as input", 1)[-1] == 1
    assert _resolve_output_switch(3, 1) == 1
    # Unwired input_switch defaults to Img (2).
    assert _resolve_output_switch("Same as input", None) == 2


def test_publish_result_exposes_resolved_result() -> None:
    spec = Koolook_PublishResult.INPUT_TYPES()["required"]

    assert list(spec) == ["result"]
    assert Koolook_PublishResult.RETURN_NAMES == ("result",)
    assert Koolook_PublishResult.OUTPUT_NODE is True
    assert Koolook_PublishResult().run(result="/out/mask_v001.png") == {
        "ui": {"text": ["/out/mask_v001.png"]},
        "result": ("/out/mask_v001.png",),
    }


def test_publish_router_exposes_switch_aligned_payload_outputs() -> None:
    spec = Koolook_PublishRouter.INPUT_TYPES()["required"]

    assert list(spec) == ["selector", "payload"]
    assert Koolook_PublishRouter.RETURN_NAMES == ("EXR", "QT", "Img", "Prompt")
    assert Koolook_PublishRouter().route(selector=2, payload="pixels") == (
        "pixels",
        "pixels",
        "pixels",
        "pixels",
    )
