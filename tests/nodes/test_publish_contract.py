"""Koolook publish contract nodes."""

from __future__ import annotations

from k_publish_contract import (
    Koolook_PublishInput,
    Koolook_PublishOutput,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


def test_publish_contract_nodes_register() -> None:
    assert NODE_CLASS_MAPPINGS["Koolook_PublishInput"] is Koolook_PublishInput
    assert NODE_CLASS_MAPPINGS["Koolook_PublishOutput"] is Koolook_PublishOutput
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishInput"] == "Koolook Publish Input"
    assert NODE_DISPLAY_NAME_MAPPINGS["Koolook_PublishOutput"] == "Koolook Publish Output"


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


def test_publish_output_exposes_stable_fields() -> None:
    spec = Koolook_PublishOutput.INPUT_TYPES()["required"]

    assert list(spec) == ["folder", "name", "version", "result"]
    assert Koolook_PublishOutput.RETURN_NAMES == ("folder", "name", "version", "result")
    assert Koolook_PublishOutput().run(
        folder="/out",
        name="mask",
        version="1",
        result="/out/mask_v001.png",
    ) == ("/out", "mask", "1", "/out/mask_v001.png")
