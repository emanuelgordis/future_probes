from src.custom_datasets import (
    conflicting_qa,
    eli5_alce,
    gandalf,
    purple,
    tensor_trust,
    harmbench,
    mmlu_pro,
    sep,
    rules,
    struq,
    elephant,
    sycophancy,
    sorrybench,
    strongreject,
    winogenerated,
    wealth_seeking_inclination,
    survival_instinct,
    myopic_reward,
    persona_drift,
)


def get_dataset(
    dataset_name: str,
    subset: int = None,
    icl_examples: int = None,
    seed: int = None,
    disable_reasoning: bool = False,
    dataset_path: str = None,
):
    if dataset_name.startswith("persona_drift"):
        dataset = persona_drift.PersonaDriftDataset(
            dataset_name,
            subset=subset,
            seed=seed,
            disable_reasoning=disable_reasoning,
            data_path=dataset_path,
        )
    elif dataset_name.startswith("sep"):
        dataset = sep.SEPDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("purple"):
        dataset = purple.PurpleDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif "tensor_trust" in dataset_name:
        dataset = tensor_trust.TensorTrustDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("gandalf"):
        dataset = gandalf.GandalfDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("eli5_alce"):
        dataset = eli5_alce.ELI5ALCEDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name == "conflicting_qa_double":
        dataset = conflicting_qa.ConflictingQADataset(
            dataset_name,
            double_context=True,
            subset=subset,
            seed=seed,
            disable_reasoning=disable_reasoning,
        )
    elif dataset_name.startswith("conflicting_qa"):
        dataset = conflicting_qa.ConflictingQADataset(
            dataset_name,
            double_context=False,
            subset=subset,
            seed=seed,
            disable_reasoning=disable_reasoning,
        )
    elif dataset_name.startswith("harmbench_standard"):
        dataset = harmbench.HarmBenchDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("mmlu_pro"):
        dataset = mmlu_pro.MMLUProDataset(
            dataset_name,
            subset=subset,
            icl_examples=icl_examples,
            seed=seed,
            disable_reasoning=disable_reasoning,
        )
    elif dataset_name.startswith("rules"):
        dataset = rules.RulesDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("struq"):
        dataset = struq.StruQDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("elephant_aita"):
        dataset = elephant.ElephantAITADataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("sycophancy"):
        dataset = sycophancy.SycophancyDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("sorrybench"):
        dataset = sorrybench.SorryBenchDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("strongreject"):
        dataset = strongreject.StrongRejectDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("winogenerated"):
        dataset = winogenerated.WinogeneratedDataset(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("wealth_seeking"):
        dataset = wealth_seeking_inclination.WealthSeeking(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("survival_instinct"):
        dataset = survival_instinct.SurvivalInstinct(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    elif dataset_name.startswith("myopic_reward"):
        dataset = myopic_reward.MyopicReward(
            dataset_name, subset=subset, seed=seed, disable_reasoning=disable_reasoning
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset
