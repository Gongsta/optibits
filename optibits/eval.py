import torch
import math
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader


def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss = 0
    total_tokens = 0
    dataloader = DataLoader(texts, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity


def score_batch(model, tokenizer, questions, choices_list, format_prompt):
    """
    Computes log-probabilities of each answer choice for a batch of questions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [
        format_prompt(question, choices) for question, choices in zip(questions, choices_list)
    ]

    # Tokenize prompts once
    prompt_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    prompt_len = prompt_inputs.input_ids.shape[1]

    scores = []
    for i in range(len(choices_list[0])):  # Iterate over answer choices
        full_inputs = tokenizer(
            [p + choices[i] for p, choices in zip(prompts, choices_list)],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**full_inputs)

        # Get log-probabilities for each choice
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        token_ids = full_inputs.input_ids[:, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        choice_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(choice_log_probs.sum(dim=-1))  # Sum log probabilities per choice

    return (
        torch.stack(scores, dim=1).argmax(dim=1).tolist()
    )  # Get index of max log-prob for each question


def format_prompt_mmlu(question, choices):
    return (
        f"{question}\n"
        + "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        + "\nAnswer: "
    )


def evaluate_multiple_choice_dataset(
    model,
    tokenizer,
    dataset_name="cais/mmlu",
    subject="all",
    split="test",
    num_samples=500,
    batch_size=16,
    question_key="question",
    choices_key="choices",
    answer_key="answer",
    format_prompt=None,
):
    """
    Evaluates a preloaded Hugging Face model on a multiple-choice dataset.

    Parameters:
    - model (torch.nn.Module): Preloaded Hugging Face model.
    - tokenizer (AutoTokenizer): Preloaded tokenizer.
    - dataset_name (str): Hugging Face dataset name.
    - subject (str): Subject within the dataset (if applicable).
    - split (str): Dataset split to evaluate (e.g., 'test', 'validation').
    - num_samples (int): Number of questions to evaluate.
    - batch_size (int): Number of questions to process in parallel.
    - question_key (str): Key for accessing the question text.
    - choices_key (str): Key for accessing answer choices.
    - answer_key (str): Key for accessing the correct answer.
    - format_prompt (function): A function that formats the prompt based on dataset requirements.

    Returns:
    - accuracy (float): Accuracy of the model on the dataset.
    """

    # Load dataset
    dataset = load_dataset(dataset_name, subject, split=split).select(range(num_samples))

    # Ensure format_prompt is defined
    if format_prompt is None:
        raise ValueError(
            "A format_prompt function must be provided to specify how prompts should be structured."
        )

    # Evaluate dataset in batches
    correct = 0
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Evaluating {dataset_name}"):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        batch_questions = [sample[question_key] for sample in batch]
        batch_choices = [sample[choices_key] for sample in batch]
        batch_answers = [sample[answer_key] for sample in batch]

        predicted = score_batch(model, tokenizer, batch_questions, batch_choices, format_prompt)
        correct += sum([1 for p, a in zip(predicted, batch_answers) if p == a])

    # Final Accuracy Score
    final_accuracy = correct / num_samples
    print(f"\nFinal Accuracy on '{dataset_name} ({split})': {final_accuracy:.2%}")

    return final_accuracy
