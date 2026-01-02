import torch
from transformers import (
    AutoConfig,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)


def load_qwen_with_merged_vision(
    config_dir: str, base_model_id: str, vision_source_id: str
):
    """
    Loads a Qwen VL model and replaces its vision encoder weights with those from another model.
    """
    config = AutoConfig.from_pretrained(config_dir)
    # Load the primary model (e.g., "Qwen/Qwen2-VL-7B-Instruct")
    # model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    #     base_model_id,
    #     config=config,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     ignore_mismatched_sizes=True,
    # )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )

    # Load the source model containing the desired vision weights
    vision_source = AutoModel.from_pretrained(
        vision_source_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("copying vision weights")
    for key, params in vision_source.named_parameters():
        if "encoder" in key:
            if key == "encoder.patch_embed.proj_c1.weight":
                model.state_dict()["model.visual.patch_embed.proj.weight"].copy_(params)
            elif key == "encoder.patch_embed.proj_c1.bias":
                model.state_dict()["model.visual.patch_embed.proj.bias"].copy_(params)
            elif "proj" in key:
                pass
            else:
                model.state_dict()[key.replace("encoder.", "model.visual.")].copy_(
                    params
                )
    print("done copying vision weights")

    return model


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    model = load_qwen_with_merged_vision(
        args[0],
        args[1],
        args[2],
    )
    # Save modified model
    # accelerator = Accelerator()
    # accelerator.save_model(
    #     model=model,
    #     save_directory="/home/user/checkpoints/qwen3-vl-30b-a3b-instruct-merged",
    #     # max_shard_size="4GB",
    #     # safe_serialization=True,
    # )
    model.save_pretrained(args[0])

    del model
    torch.cuda.empty_cache()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args[0],
        dtype=torch.bfloat16,
        device_map="auto",
    )
