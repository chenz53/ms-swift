import torch
from swift.llm import get_template_meta
from swift.utils import get_logger
from swift.llm.template.template_inputs import StdTemplateInputs
import sys
import os
from transformers import Qwen3VLProcessor

sys.path.append(os.getcwd())

# Import our custom template registration to ensure it's registered
from examples.custom.vlm_template import CustomQwen3VLTemplate

from dotenv import load_dotenv

load_dotenv()

logger = get_logger()


if __name__ == "__main__":
    from swift.llm import load_dataset
    import torch

    # Test Verification
    print("--- Verifying Custom VLM Dataset & Template ---")

    # Debug helper
    import sys

    print("Loaded modules containing 'vlm_template':")
    for k in sys.modules.keys():
        if "vlm_template" in k:
            print(f" - {k}")

    # 2. Load the dataset
    dataset_path = "examples/custom/dummy_data/"
    print(f"Loading dataset from {dataset_path}...")
    try:
        train_dataset, val_dataset = load_dataset([dataset_path])
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not train_dataset:
        print("No training dataset loaded!")
        sys.exit(1)

    print(f"Loaded {len(train_dataset)} training examples.")

    # 3. Processor & Template
    template_type = "qwen3-vl-custom"
    template_meta = get_template_meta(template_type)
    model_id = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    print(f"Loading processor: {model_id}")
    processor = Qwen3VLProcessor.from_pretrained(model_id, trust_remote_code=True)

    if not hasattr(processor, "model_info"):
        from types import SimpleNamespace

        processor.model_info = SimpleNamespace(
            max_model_len=8192,
            config=SimpleNamespace(problem_type="causal_lm"),
            task_type="causal_lm",
            torch_dtype=torch.float16,
        )
        processor.model_meta = SimpleNamespace(is_multimodal=True)

    # Force re-register or ensuring we use the logic
    # We manually instantiated CustomQwen3VLTemplate
    template = CustomQwen3VLTemplate(processor=processor, template_meta=template_meta)
    template.set_mode("train")

    # 4. Process Loop
    print("\nProcessing examples...")
    for i, item in enumerate(train_dataset):
        print(f"\n--- Example {i} ---")
        print(item)

        messages = item.get("messages", [])
        images = item.get("images", [])

        print(f"Images type: {type(images)}")
        if images:
            print(f"Image 0 type: {type(images[0])}")
            print(f"Image 0 val: {images[0]}")

        inputs = StdTemplateInputs(messages=messages, images=images)

        # Encode
        try:
            from swift.llm.template.template_inputs import TemplateInputs

            template_inputs = TemplateInputs(chosen=inputs)
            encoded = template.encode(template_inputs)

            print("Encoding Successful.")
            print(f"Input IDs shape: {len(encoded['input_ids'])}")
            decoded_inputs = processor.tokenizer.decode(encoded["input_ids"])
            print(f"Decoded Inputs: {decoded_inputs}")

            if "pixel_values" in encoded:
                print(f"Pixel Values: {encoded['pixel_values'].shape}")

            if "labels" in encoded:
                decoded_labels = processor.tokenizer.decode(
                    [l for l in encoded["labels"] if l != -100]
                )
                print(f"Decoded Labels: {decoded_labels}")

        except Exception as e:
            print(f"Failed to encode example {i}: {e}")
            import traceback

            traceback.print_exc()

        if i >= 2:
            break
