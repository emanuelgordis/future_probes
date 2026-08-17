import argparse
import json
import os
import torch
import vllm
from tqdm import tqdm
import re
import openai_harmony
import pdb

from src.reasoning_model_utils import extract_response_part, extract_thinking_content_from_response

# Fix for VLLM CUDA multiprocessing issue
# Set VLLM's multiprocessing method to spawn before any VLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


from src.model_utils import (
    get_vllm_model_and_tokenizer,
    delete_vllm_model_and_free_memory,
)
from src.custom_datasets import get_dataset
from plotting_scripts.behavior_stability import plot_behavioral_stability

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def run_behavioral_stability(
    model_name,
    dataset_name,
    subset,
    results_file,
    num_samples,
    max_new_tokens,
    decoder,
    seed,
    sampling_seed,
    temperature,
    multi_gpu,
    disable_reasoning,
    ensure_thinking,
    model_path,
    dataset_path=None,
    skip_behavior_scoring=False,
    max_model_len=None,
):
    dataset = get_dataset(
        dataset_name,
        subset=subset,
        icl_examples=None,
        seed=seed,
        disable_reasoning=disable_reasoning,
        dataset_path=dataset_path,
    )

    dataset_supports_behavior_scoring = getattr(
        dataset, "supports_behavior_scoring", True
    )
    behavior_scoring_enabled = (
        dataset_supports_behavior_scoring and not skip_behavior_scoring
    )
    if not behavior_scoring_enabled:
        reason = (
            "the dataset has no binary behavior labels"
            if not dataset_supports_behavior_scoring
            else "--skip_behavior_scoring was set"
        )
        print(f"Behavior scoring disabled because {reason}.")

    # Existing short, single-turn datasets retain their memory-saving 2k prompt
    # allowance.  Persona-drift histories are much longer, so let vLLM use the
    # model's configured context length unless the caller supplies an explicit cap.
    effective_max_model_len = max_model_len
    if effective_max_model_len is None and not getattr(
        dataset, "requires_long_context", False
    ):
        effective_max_model_len = max_new_tokens + 2000

    model, tokenizer, lora_request = get_vllm_model_and_tokenizer(
        model_name, multi_gpu=multi_gpu, max_model_len=effective_max_model_len
    )

    sampling_params = vllm.SamplingParams(
        n=num_samples,
        temperature=0.0 if decoder == "greedy" else temperature,
        max_tokens=max_new_tokens,
        seed=sampling_seed,
        skip_special_tokens=False,
    )

    # only needed for gpt-oss models
    encoding = openai_harmony.load_harmony_encoding(openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS)

    # Gather all input prompts
    input_prompts = []
    for idx, example in enumerate(dataset):
        input_ids, input_prompt = dataset.get_model_input_from_an_example(
            example, tokenizer
        )

        input_prompts.append(input_prompt)

    # Generate outputs
    model_outputs = model.generate(
        input_prompts, sampling_params, lora_request=lora_request
    )

    # After this step we don't need the model anymore, delete and clean it.
    delete_vllm_model_and_free_memory(model)

    # Print cuda allocated memory in GB
    print(
        f"Cuda allocated memory: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB"
    )

    average_behaviors = []
    first_behaviors = []
    worst_behaviors = []

    outputs_to_save = []
    for example_output, example in tqdm(
        zip(model_outputs, dataset), total=len(dataset)
    ):
        input_ids, input_prompt = dataset.get_model_input_from_an_example(
            example, tokenizer
        )

        responses = [response.text for response in example_output.outputs]
        response_token_ids = [
            [int(token_id) for token_id in response.token_ids]
            for response in example_output.outputs
        ]
        if isinstance(input_ids, torch.Tensor):
            prompt_token_ids = input_ids.detach().cpu().reshape(-1).tolist()
        else:
            prompt_token_ids = list(input_ids)
        prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]

        if "gpt-oss" in model_name:
            # GPT-oss uses harmony format
            thinking_contents = []
            answer_contents = []
            responses_tokens = [response.token_ids for response in example_output.outputs]
            for one_response_tokens in responses_tokens:
                entries = encoding.parse_messages_from_completion_tokens(one_response_tokens, openai_harmony.Role.ASSISTANT)

                analysis_messages = [entry for entry in entries if entry.channel == "analysis"]
                final_messages = [entry for entry in entries if entry.channel == "final"]

                if len(analysis_messages) > 0 :
                    analysis_msg = analysis_messages[0]
                    thinking_content = analysis_msg.content[0].text
                else:
                    thinking_content = ""
                
                if len(final_messages) > 0:
                    final_msg = final_messages[0]
                    answer_content = final_msg.content[0].text
                else:
                    answer_content = ""

                thinking_contents.append(thinking_content)
                answer_contents.append(answer_content)
        elif "gemma-4" in model_name:
            # Gemma 4 uses <|channel>thought...<channel|> for thinking.
            # skip_special_tokens=False is set above so markers appear in response.text.
            thinking_contents = [
                extract_thinking_content_from_response(response)
                for response in responses
            ]
            answer_contents = [
                extract_response_part(response)
                for response in responses
            ]
        elif "ministral" in model_name.lower():
            thinking_contents = [
                extract_thinking_content_from_response(response)
                for response in responses
            ]
            answer_contents = [
                extract_response_part(response)
                for response in responses
            ]
        else:
            thinking_contents = [
                response.split("</think>")[0] if "</think>" in response else ""
                for response in responses
            ]
            answer_contents = [
                response.split("</think>")[1] if "</think>" in response else response
                for response in responses
            ]

        if ensure_thinking:
            no_thinking_examples = [
                i
                for i, thinking_content in enumerate(thinking_contents)
                if thinking_content == ""
            ]
            all_have_thinking = len(no_thinking_examples) == 0
            if not all_have_thinking:
                print("WARNING:")
                print(
                    f"Thinking content is empty for {len(no_thinking_examples)} examples. Ensure ensure_thinking is False if you want to ignore this."
                )
                print("--------------------------------")

        if behavior_scoring_enabled:
            # Behavior is evaluated only on the public final answer; thinking is private.
            examples = [example] * len(answer_contents)
            behaviors = dataset.detect_behavior_batched(examples, answer_contents)
            behaviors = [
                int(behavior) if behavior is not None else behavior
                for behavior in behaviors
            ]
            int_behaviors = [behavior for behavior in behaviors if behavior is not None]
            average_behavior = sum(int_behaviors) / (len(int_behaviors) or 1)
            first_behavior = behaviors[0]
            worst_behavior = max(int_behaviors) if int_behaviors else None
        else:
            behaviors = [None] * len(answer_contents)
            average_behavior = None
            first_behavior = None
            worst_behavior = None

        average_behaviors.append(average_behavior)
        first_behaviors.append(first_behavior)
        worst_behaviors.append(worst_behavior)

        output_record = {
            "input_prompt": input_prompt,
            # Preserve the exact prompt/completion tokenization used for generation.
            # Activation gathering can then replay the rollout without silently
            # changing BPE boundaries by re-tokenizing concatenated text.
            "prompt_token_ids": prompt_token_ids,
            "responses": [
                {
                    "response": response,
                    "token_ids": token_ids,
                    "behavior": behavior,
                    "thinking_content": thinking_content,
                    "answer_content": answer_content,
                }
                for response, token_ids, behavior, thinking_content, answer_content in zip(
                    responses,
                    response_token_ids,
                    behaviors,
                    thinking_contents,
                    answer_contents,
                )
            ],
            "average_behavior": average_behavior,
            "worst_behavior": worst_behavior,
            "first_behavior": first_behavior,
        }
        if "messages" in example:
            output_record["messages"] = example["messages"]
        metadata_keys = getattr(dataset, "output_metadata_keys", ())
        metadata = {key: example[key] for key in metadata_keys if key in example}
        if metadata:
            output_record["metadata"] = metadata
        outputs_to_save.append(output_record)

    total_response_count = sum(
        len(example_output["responses"]) for example_output in outputs_to_save
    )
    if behavior_scoring_enabled:
        print(f"Average behaviors: {average_behaviors}")
        total_avg_behavior = sum(average_behaviors) / (len(average_behaviors) or 1)
        print(f"Total average behavior: {total_avg_behavior}")

        all_behaviors_flat = [
            response["behavior"]
            for example_output in outputs_to_save
            for response in example_output["responses"]
        ]
        none_behavior_count = sum(
            1 for behavior in all_behaviors_flat if behavior is None
        )
        none_behavior_fraction = (
            none_behavior_count / total_response_count
            if total_response_count > 0
            else 0.0
        )
        print(
            f"None behavior fraction: {none_behavior_fraction:.4f} "
            f"({none_behavior_count}/{total_response_count})"
        )
    else:
        total_avg_behavior = None
        none_behavior_count = None
        none_behavior_fraction = None

    results = {
        "model_name": model_name,
        "disable_reasoning": disable_reasoning,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "subset": subset,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "decoder": decoder,
        "seed": seed,
        "sampling_seed": sampling_seed,
        "temperature": temperature,
        "multi_gpu": multi_gpu,
        "max_model_len": effective_max_model_len,
        "behavior_scoring_enabled": behavior_scoring_enabled,
        "average_behaviors": repr(average_behaviors),
        "worst_behaviors": repr(worst_behaviors),
        "first_behaviors": repr(first_behaviors),
        "total_avg_behavior": total_avg_behavior,
        "none_behavior_count": none_behavior_count,
        "total_response_count": total_response_count,
        "none_behavior_fraction": none_behavior_fraction,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

    outputs_file = results_file.replace("results.json", "outputs.json")
    with open(outputs_file, "w") as f:
        json.dump(outputs_to_save, f, indent=4)
    print(f"Saved outputs to {outputs_file}")

    if behavior_scoring_enabled:
        plot_behavioral_stability(results_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run behavioral stability analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Behavioral stability args
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model or path to the model directory.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Name of the dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Optional local dataset file or directory. Persona-drift accepts either "
            "a transcript JSON/JSONL file, its containing directory, or an Assistant "
            "Axis repository checkout."
        ),
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Subset of the dataset to use."
    )
    parser.add_argument(
        "--seed", type=int, default=43, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help=(
            "vLLM context-length cap, including prompt and generation. Long-context "
            "datasets use the model default when this is omitted."
        ),
    )
    parser.add_argument(
        "--skip_behavior_scoring",
        action="store_true",
        help=(
            "Skip dataset behavior detection and behavior-stability plotting. This is "
            "automatic for persona-drift datasets."
        ),
    )
    parser.add_argument(
        "--disable_reasoning",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Disable reasoning by the model.",
    )
    parser.add_argument(
        "--ensure_thinking",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Throw an error if the thinking content is empty.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Random seed for sampling when generating the outputs.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate for each example in the dataset.",
    )
    parser.add_argument(
        "--multi_gpu",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use multiple GPUs for model inference. Use --multi_gpu=True or --multi_gpu=False",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the model weights or configuration for ablated/multiplied models."
        " If not provided, the base model is used.",
    )

    args = parser.parse_args()

    if args.disable_reasoning and args.ensure_thinking:
        raise ValueError(
            "It does not make sense to use --disable_reasoning and --ensure_thinking at the same time."
        )

    experiment_config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "subset": args.subset,
        "seed": args.seed,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "multi_gpu": args.multi_gpu,
        "max_model_len": args.max_model_len,
        "skip_behavior_scoring": args.skip_behavior_scoring,
    }

    if args.dataset is None:
        raise ValueError("Provide a dataset name with --dataset.")
    
    decoder = "greedy" if args.temperature == 0.0 else "gumbel"

    if decoder == "greedy" and args.num_samples > 1:
        print(
            "WARNING: Temperature == 0.0 cannot be used with multiple samples, setting num_samples to 1."
        )
        args.num_samples = 1

    short_model_name = args.model_name.split("/")
    # make sure the correct model name is extracted when a path is given as model name
    if len(short_model_name) <= 2:
        short_model_name = short_model_name[-1]
    else:
        short_model_name = short_model_name[1]

    # Create output directory and file name
    if args.model_path is None:  # in this case we use the base model
        output_dir = f"results/behavioral_stability/{short_model_name}/base_model/{args.dataset}"
    else:
        # extract topk, m, pos/neg from the model path
        match = re.search(
            r"topk_(\d+).*?_b_(True|False).*?_m_([\d.]+|None)", args.model_path
        )
        output_dir = f"results/behavioral_stability/{short_model_name}/modified_model/top_{match.group(1)}/m{match.group(3)}/{args.dataset}/{match.group(2)}_heads"

    os.makedirs(output_dir, exist_ok=True)
    filename = f"n{args.subset if args.subset else 'full'}_nsamp{args.num_samples}_l{args.max_new_tokens}_{decoder}_s{args.seed}_ss{args.sampling_seed}_t{args.temperature}{'_no_reasoning' if args.disable_reasoning else ''}_results.json"
    output_file = os.path.join(output_dir, filename)

    run_behavioral_stability(
        args.model_name,
        args.dataset,
        args.subset,
        output_file,
        args.num_samples,
        args.max_new_tokens,
        decoder,
        args.seed,
        args.sampling_seed,
        args.temperature,
        args.multi_gpu,
        args.disable_reasoning,
        args.ensure_thinking,
        args.model_path,
        args.dataset_path,
        args.skip_behavior_scoring,
        args.max_model_len,
    )
