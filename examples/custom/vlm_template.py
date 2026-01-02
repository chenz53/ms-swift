from typing import Any, Dict, List, Literal

import torch
from PIL import Image
from smb_biopan_utils.imaging_process import fetch_medical_volume

from swift.llm import Template, register_template
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.utils import get_logger

logger = get_logger()

# Define a new template type
MLLMTemplateType.qwen3_vl_custom = "qwen3-vl-custom"


def process_imaging_info(
    images: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    volume_inputs: list[torch.Tensor] = []
    grid_thws: list[torch.Tensor] = []
    for img in images:
        if "path" in img:
            img["image"] = img.pop("path")
        volume, grid_thw = fetch_medical_volume(img)
        volume_inputs.append(volume)
        grid_thws.append(grid_thw)
    if len(volume_inputs) == 0:
        return None
    return torch.cat(volume_inputs, dim=0), torch.stack(grid_thws)


class CustomQwen3VLTemplate(Qwen3VLTemplate):
    """
    Custom Qwen3VL Template that supports passing pre-processed 3D tensor images directly.
    """

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        """
        Override _preprocess_inputs to assign custom pixel values and replace paths with dummies.
        """
        if inputs.images:
            # We assume if strings are passed, they are our custom paths
            if any(isinstance(img, dict) and "path" in img for img in inputs.images):
                try:
                    flattened_patches, grid_thws = process_imaging_info(inputs.images)
                    print(flattened_patches.shape)

                    # Reshape flattened_patches if necessary: [num_images, num_patches, dim] -> [total_patches, dim]
                    if flattened_patches is not None and flattened_patches.dim() == 3:
                        total_patches = (
                            flattened_patches.shape[0] * flattened_patches.shape[1]
                        )
                        pixel_values = flattened_patches.reshape(total_patches, -1)
                    else:
                        pixel_values = flattened_patches

                    # Store these in extra_kwargs so we can retrieve them in _encode
                    inputs.extra_kwargs["custom_pixel_values"] = pixel_values
                    inputs.extra_kwargs["custom_image_grid_thw"] = grid_thws

                    # Replace with dummies to satisfy base loader
                    inputs.images = [Image.new("RGB", (1, 1)) for _ in inputs.images]

                except Exception as e:
                    logger.error(f"Failed to process custom imaging info: {e}")
                    raise e

        super()._preprocess_inputs(inputs)

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ) -> List[Context]:
        """
        Override replace_tag to avoid importing qwen_vl_utils and return standard placeholders.
        """
        if media_type == "image":
            # Return standard placeholders for Qwen2-VL / Qwen3-VL
            return ["<|vision_start|><|image_pad|><|vision_end|>"]
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        Override _encode to use custom pixel values for token expansion.
        """
        # Call Template._encode which handles text tokenization (but not image expansion yet)
        encoded = Template._encode(self, inputs)

        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)

        # Retrieve custom values
        pixel_values = inputs.extra_kwargs.get("custom_pixel_values")
        image_grid_thw = inputs.extra_kwargs.get("custom_image_grid_thw")

        if pixel_values is not None:
            media_token = self.image_token_id
            idx_list = findall(input_ids, media_token)

            # Standard QwenVL merge length
            merge_length = self.processor.image_processor.merge_size**2

            def _get_new_tokens(i):
                # Calculate number of tokens based on our custom grid
                token_len = image_grid_thw[i].prod() // merge_length
                return [media_token] * token_len

            # Expand tokens
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens
            )

            # Add pixel values to result
            encoded["pixel_values"] = pixel_values
            encoded["image_grid_thw"] = image_grid_thw

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        return encoded


# Register the new template
register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_vl_custom,
        template_cls=CustomQwen3VLTemplate,
        default_system=None,
        thinking_prefix="<think>\n",
    )
)
