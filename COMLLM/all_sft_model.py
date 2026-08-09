"""
Merge a trained LoRA adapter into a base causal language model.
Use this script when you want to merge an SFT or GRPO LoRA adapter into the base model.
All paths are provided via command-line arguments, allowing users to plug in their own model checkpoints without modifying the source code.
将一个训练好的LoRA适配器合并到基础因果语言模型中。
当你想要将SFT或GRPO的LoRA适配器合并到原模型时，可使用此脚本。
所有路径均通过命令行参数提供，以便用户无需修改源代码即可接入自己的模型检查点。
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model-path", required=True, help="Base model or merged SFT model path.")
    parser.add_argument("--lora-adapter-path", required=True, help="LoRA adapter directory to merge.")
    parser.add_argument("--output-dir", required=True, help="Directory used to save the merged model.")
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Dtype used when loading the base model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16

    print("--- Step 1: Loading the base model and tokenizer ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"--- Step 2: Loading the LoRA adapter from: {args.lora_adapter_path} ---")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    print("--- Step 3: Merging the LoRA adapter into the base model ---")
    merged_model = model.merge_and_unload()
    print("Merging complete!")

    print(f"--- Step 4: Saving the merged model to: {args.output_dir} ---")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n--- All Done! ---")
    print(f"The merged model has been saved successfully to: {args.output_dir}")


if __name__ == "__main__":
    main()
