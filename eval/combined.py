import json
import numpy as np
import matplotlib.pyplot as plt
from latex2sympy2 import latex2sympy
from sympy import latex, simplify

def get_canonical_form(expression: str) -> str:
    try:
        expression = expression.replace('\\boxed{', '').replace('}', '')
        parsed_expr = latex2sympy(expression)
        simplified_expr = simplify(parsed_expr)
        return latex(simplified_expr)
    except:
        return expression

def calculate_accuracies(jsonl_file):
    ns = [1, 2, 4, 8, 16]
    accuracies = {n: [] for n in ns}
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            correct_answer = get_canonical_form(data['answer'])
            for n in ns:
                pred = get_canonical_form(data[f'pred_weighted@{n}'])
                is_correct = (pred == correct_answer)
                accuracies[n].append(is_correct)
    
    final_accuracies = []
    for n in ns:
        acc = np.mean(accuracies[n]) * 100
        final_accuracies.append(acc)
    
    return ns, final_accuracies

def plot_combined_accuracies(beam_file, best_of_n_file, dvts_file):
    plt.figure(figsize=(12, 8))
    
    # Calculate accuracies for each method
    ns, beam_accuracies = calculate_accuracies(beam_file)
    _, best_of_n_accuracies = calculate_accuracies(best_of_n_file)
    _, dvts_accuracies = calculate_accuracies(dvts_file)
    
    x_values = [2**i for i in range(len(ns))]
    
    # Plot all methods
    plt.plot(x_values, beam_accuracies, marker='o', label='Beam Search', linewidth=2)
    plt.plot(x_values, best_of_n_accuracies, marker='s', label='Best-of-N', linewidth=2)
    plt.plot(x_values, dvts_accuracies, marker='^', label='DVTS', linewidth=2)
    
    plt.xscale('log', base=2)
    plt.xlabel('Number of generations per problem', fontsize=12)
    plt.ylabel('MATH-500 accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.title('Comparison of Search Methods on MATH-500', fontsize=14)
    
    # Add minor gridlines
    plt.grid(True, which='minor', alpha=0.1)
    
    # Save the combined plot
    plt.savefig('combined_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# File paths
beam_search_file = "/home/ubuntu/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/beam_search_completions_n_16.jsonl"
best_of_n_file = "/home/ubuntu/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"
dvts_file = "/home/ubuntu/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/dvts_completions.jsonl"

# Create the combined plot
plot_combined_accuracies(beam_search_file, best_of_n_file, dvts_file)