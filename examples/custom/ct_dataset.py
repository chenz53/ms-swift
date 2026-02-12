"""Custom Gemma3-vision template with built-in CT DICOM processing.

Each dataset row has structured content with ``r2://`` image URLs::

    {"messages": [
        {"role": "user",
         "content": [
             {"type": "image", "image": "r2://bucket/series.tar"},
             {"type": "text", "text": "Write a radiology report."}
         ]},
        {"role": "assistant",
         "content": [{"type": "text", "text": "Findings: ..."}]}
    ]}

The custom template overrides ``encode()`` so that CT processing
(R2 download, DICOM extraction, windowing, base64 encoding) happens
**inside** the template — i.e.  lazily, one row at a time, during
streaming.  The standard ``EncodePreprocessor`` drives everything;
no custom dataset preprocessor is required.

Usage::

    swift sft \\
        --external_plugins examples/custom/ct_dataset.py \\
        --template_type ct_gemma3_vision \\
        --model google/medgemma-1.5-4b-it \\
        --dataset gradient_report_generation \\
        --streaming true ...
"""

from copy import deepcopy
from typing import Any, Dict, List, Union

from swift.dataset import DatasetMeta, MessagesPreprocessor, register_dataset
from swift.template.register import register_template
from swift.template.template_inputs import TemplateInputs
from swift.template.templates.gemma import GemmaTemplateMeta, Gemma3VisionTemplate
from swift.utils import get_logger

from smb_utils import process_ct_slice_info

logger = get_logger()


# ------------------------------------------------------------------
# Custom template
# ------------------------------------------------------------------

class CTGemma3VisionTemplate(Gemma3VisionTemplate):
    """Gemma-3 vision template that transparently expands CT DICOM tars.

    Before the normal Gemma-3 encoding pipeline runs, this template:

    1. Detects ``r2://`` URLs in structured message content.
    2. Injects ``max_slices`` into each image element.
    3. Calls ``process_ct_slice_info`` which downloads the tar, reads
       the DICOM series, windows the Hounsfield values, selects
       representative slices, and returns base64-encoded JPEG images.
    4. Passes the result (structured content with ``data:image/…``
       URIs) to the standard Gemma-3 ``encode()`` — which converts
       structured content to flat ``<image>`` tags, loads PIL images
       from the data URIs, tokenises, and builds ``pixel_values``.

    No separate dataset preprocessor is needed.
    """

    # ---- class-level knobs (set before training) --------------------
    ct_max_slices: int = 40
    ct_encode_format: str = "jpeg"

    # -----------------------------------------------------------------

    def encode(
        self,
        inputs: Union[TemplateInputs, Dict[str, Any]],
        return_template_inputs: bool = False,
        return_length: bool = False,
    ) -> Dict[str, Any]:
        # Only intercept raw dict inputs (from EncodePreprocessor).
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            if self._has_r2_images(messages):
                inputs = self._process_ct(inputs, messages)

        return super().encode(
            inputs,
            return_template_inputs=return_template_inputs,
            return_length=return_length,
        )

    # ---- internal helpers -------------------------------------------

    @staticmethod
    def _has_r2_images(messages: List[Dict[str, Any]]) -> bool:
        """Return True if any message contains an ``r2://`` image."""
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image"
                    and isinstance(item.get("image"), str)
                    and item["image"].startswith("r2://")
                ):
                    return True
        return False

    def _process_ct(
        self,
        inputs: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Download, extract, window CT slices and update *inputs*."""
        # Work on a copy so the underlying dataset row is not mutated.
        messages = deepcopy(messages)

        # Inject max_slices
        if self.ct_max_slices is not None:
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        item.setdefault("max_slices", self.ct_max_slices)

        # Call CT processing (may raise — EncodePreprocessor will skip)
        messages = process_ct_slice_info(messages, self.ct_encode_format)

        # Flatten text-only structured content back to plain strings.
        # Gemma3Template._swift_encode concatenates system + user content
        # and expects them to be strings; structured content that has no
        # images (system, assistant) must be converted.
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list) and all(
                isinstance(c, dict) and c.get("type") == "text" for c in content
            ):
                msg["content"] = "".join(c.get("text", "") for c in content)

        # Update the inputs dict.  remove_messages_media (called inside
        # super().encode → from_dict) will extract images from the
        # structured content, so we must NOT leave stale top-level images.
        inputs = dict(inputs)  # shallow copy
        inputs["messages"] = messages
        inputs.pop("images", None)
        return inputs


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

register_template(
    GemmaTemplateMeta(
        "ct_gemma3_vision",
        template_cls=CTGemma3VisionTemplate,
    ),
)

register_dataset(
    DatasetMeta(
        dataset_path="/workspace/data/xxx.jsonl",
        dataset_name="report_generation",
        preprocess_func=MessagesPreprocessor(),
    ),
)
