from benchmark_evaluator import BenchmarkEvaluator

def main():
    # Example usage of the BenchmarkEvaluator
    
    # Initialize the evaluator with your benchmark file
    evaluator = BenchmarkEvaluator('path/to/your/benchmark.csv')
    
    # Example cleaned texts (replace with your actual cleaned texts)
    cleaned_texts = [
        "This is a cleaned text",
        "Another cleaned example",
        # Add more cleaned texts...
    ]
    
    # Evaluate the cleaned texts
    results = evaluator.evaluate_benchmark(cleaned_texts)
    
    # Print results
    print("Evaluation Results:")
    print(f"Mean Levenshtein Distance: {results['mean_distance']:.2f}")
    print(f"Mean Normalized Distance: {results['mean_normalized_distance']:.4f}")
    print(f"Standard Deviation (Distance): {results['std_distance']:.2f}")
    print(f"Standard Deviation (Normalized): {results['std_normalized_distance']:.4f}")
    
    # Save results to a file
    evaluator.save_results(results, 'evaluation_results.json')

if __name__ == "__main__":
    main() 