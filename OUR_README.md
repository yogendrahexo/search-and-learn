## Replicating our test-time compute results

### System Setup for Experiment

- **Instance:** g4dn.2xlarge
- **GPUs:** 1 × 16GB T4
- **Search Batch Size:** 2
- **PRM Batch Size:** 2
- **Max Tokens:** 2048
- **Model Length:** 2048
- **N:** 8
- **Seed:** 0

## Setup

```
!git clone https://github.com/yogendrahexo/search-and-learn.git
%cd search-and-learn
!git checkout t4
!pip install -e .[dev]
```

To get started quickly, we recommend using the provided YAML files in the `recipes/` directory. These files contain all the necessary parameters for running the different search algorithms (Best-of-N, beam search, DVTS).

**Basic Usage:**

1.  **Choose a YAML file:** Select the appropriate YAML file for your desired model and search method (e.g., `recipes/Llama-3.2-1B-Instruct/best_of_n.yaml`).
2.  **Run the script:** Execute the `test_time_compute.py` script, providing the path to your chosen YAML file:

    ```shell
    export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
    python scripts/test_time_compute.py $CONFIG
    ```

**Configuration Options:**

The YAML files provide a wide range of configuration options. You can modify these directly in the YAML file, or override them with command-line arguments. Key parameters include:

- `approach`: The search algorithm to use (`best_of_n`, `beam_search`, or `dvts`).
- `n`: The number of generations.
- `search_batch_size`: The batch size for parallel generation.
- `prm_batch_size`:
- `num_samples`: The number of problems to solve from the dataset.
- `seed`: The random seed for reproducibility.
- `gpu_memory_utilization`: Fraction of GPU memory to allocate to vLLM.
- `prm_path`: Path to the Preference Reward Model (PRM). Available options:
  - `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` (recommended for 16GB GPUs)
  - `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
  - `peiyi9979/math-shepherd-mistral-7b-prm`
  - `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- And many more (see `src/sal/config.py` for the complete list).

**Pushing Results to the Hugging Face Hub (Optional):**

By default, results are saved locally in a jsonl file. To push your results to a dataset on the Hugging Face Hub, you can use the following _optional_ parameters (usually configured within the YAML file):

- `push_to_hub`: Set to `true` to enable pushing to the Hub.
- `hub_dataset_id`: The ID of the dataset repository where results will be stored (e.g., `your_username/your_dataset_name`).
- `hub_dataset_private`: Set to `true` to make the dataset private.
- `overwrite_hub_revision`: Set to `true` to overwrite an existing revision on the Hub. If set to `false` and the revision exists, the script will exit.

To use these, uncomment them in your YAML file and set the desired values. For example:

```yaml
# In your YAML file (e.g., recipes/Llama-3.2-1B-Instruct/best_of_n.yaml)
push_to_hub: true
hub_dataset_id: your_username/your_dataset_name
hub_dataset_private: true
overwrite_hub_revision: false
```

## Evaluating Results

### My Way

I created a simple evaluate script in the `eval/combined.py`. which verify the results from answers and plot a final graph and save it in a png format. Just change the file paths.

### Official Way

After generating completions and (optionally) pushing them to the Hugging Face Hub, you can evaluate the results using the `evaluation/evaluate_hf.py` script. This script calculates the accuracy of the completions at different values of `n` (the number of completions used for voting).

**Prerequisites:**

1.  **Clone the Qwen2.5-Math repository:**
    ```bash
    git clone https://github.com/QwenLM/Qwen2.5-Math
    ```
2.  **Set up the evaluation environment:**
    ```bash
    cd Qwen2.5-Math
    cd latex2sympy
    pip install -e .
    cd ..
    pip install -r requirements.txt
    pip install vllm==0.5.1 --no-build-isolation
    pip install transformers==4.42.3
    cd .. # Ensure you are in the root of the `search-and-learn` repo
    ```

**Running the Evaluation:**

The following commands demonstrate how to evaluate a dataset on the Hugging Face Hub. Replace the example values with your actual `DATASET_ID`, `DATASET_CONFIG`, and `VOTING_N` values.

```shell
# hub dataset repo
export DATASET_ID="your_username/your_dataset_name"  # Replace with your dataset ID
# config to evaluate
export DATASET_CONFIG="your_dataset_config" # Replace with your dataset config
# preds@N to evaluate
export VOTING_N="1 2 4 8 16"

# Run the evaluation script
python evaluation/evaluate_hf.py \
    --dataset_id $DATASET_ID \
    --dataset_config $DATASET_CONFIG \
    --voting_n $VOTING_N
```

- **`DATASET_ID`**: The ID of the Hugging Face dataset containing your completions (e.g., `your_username/your_dataset_name`).
- **`DATASET_CONFIG`**: The name of the dataset configuration to evaluate (this corresponds to a branch on the Hub dataset).
- **`VOTING_N`**: A space-separated string of integers representing the values of 'n' (number of completions) to evaluate. For example, "1 2 4 8 16" will evaluate the accuracy using the single best completion, the majority vote of the top 2 completions, the top 4, and so on.

The script will output the accuracy for each value of `n` specified in `VOTING_N`.
