import numpy as np
from timegan import timegan
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
import matplotlib.pyplot as plt

def run_timegan_on_train_data(npz_path, output_path="timegan_generated_train_data.npz"):
    """
    Run TimeGAN on your train data and evaluate performance.
    """
    print("Loading train data...")
    npz = np.load(npz_path)
    sequences = npz["sequences"]  # (n_samples, seq_len, features)
    mask = npz["mask"]
    
    # Filter out sequences with no valid data
    valid_idx = mask.sum(axis=1) > 0
    sequences = sequences[valid_idx]
    mask = mask[valid_idx]
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Feature dimensions: {sequences.shape[2]}")
    print(f"Max sequence length: {sequences.shape[1]}")
    print(f"Sample valid lengths: {[int(mask[i].sum()) for i in range(min(5, len(mask)))]}")
    
    # Convert to TimeGAN format (list of variable-length sequences)
    # This is what TimeGAN expects
    ori_data = []
    for i in range(len(sequences)):
        valid_length = int(mask[i].sum())
        if valid_length > 0:
            # Extract only the valid part of the sequence
            valid_seq = sequences[i, :valid_length, :]
            ori_data.append(valid_seq)
    
    print(f"Converted to {len(ori_data)} variable-length sequences")
    print(f"Sample sequence lengths: {[seq.shape[0] for seq in ori_data[:5]]}")
    
    # TimeGAN parameters (you can tune these)
    parameters = {
        'module': 'gru',        # gru, lstm, or lstmLN
        'hidden_dim': 24,       # Hidden dimensions 
        'num_layer': 3,         # Number of layers
        'iterations': 5000,     # Training iterations (reduced for testing)
        'batch_size': 128       # Batch size
    }
    
    print("Starting TimeGAN training...")
    print(f"Parameters: {parameters}")
    
    # Generate synthetic data using TimeGAN
    generated_data = timegan(ori_data, parameters)
    
    print("TimeGAN training completed!")
    print(f"Generated {len(generated_data)} synthetic sequences")
    
    # Save generated data
    # Convert back to padded format for consistency with your pipeline
    max_len = max(seq.shape[0] for seq in generated_data)
    feature_dim = generated_data[0].shape[1]
    
    padded_generated = np.zeros((len(generated_data), max_len, feature_dim))
    generated_mask = np.zeros((len(generated_data), max_len))
    
    for i, seq in enumerate(generated_data):
        seq_len = seq.shape[0]
        padded_generated[i, :seq_len, :] = seq
        generated_mask[i, :seq_len] = 1
    
    np.savez_compressed(
        output_path,
        sequences=padded_generated,
        mask=generated_mask
    )
    print(f"Generated data saved to {output_path}")
    
    return ori_data, generated_data

def evaluate_timegan_performance(ori_data, generated_data, metric_iterations=3):
    """
    Evaluate TimeGAN performance using the original metrics.
    """
    print("\n" + "="*50)
    print("EVALUATING TIMEGAN PERFORMANCE")
    print("="*50)
    
    # 1. Discriminative Score
    print("Computing discriminative scores...")
    discriminative_scores = []
    for i in range(metric_iterations):
        score = discriminative_score_metrics(ori_data, generated_data)
        discriminative_scores.append(score)
        print(f"  Iteration {i+1}: {score:.4f}")
    
    avg_discriminative = np.mean(discriminative_scores)
    print(f"Average Discriminative Score: {avg_discriminative:.4f}")
    print("(Lower is better - closer to 0.5 means harder to distinguish)")
    
    # 2. Predictive Score  
    print("\nComputing predictive scores...")
    predictive_scores = []
    for i in range(metric_iterations):
        score = predictive_score_metrics(ori_data, generated_data)
        predictive_scores.append(score)
        print(f"  Iteration {i+1}: {score:.4f}")
    
    avg_predictive = np.mean(predictive_scores)
    print(f"Average Predictive Score: {avg_predictive:.4f}")
    print("(Lower is better - better generalization)")
    
    # 3. Visualization
    print("\nGenerating visualizations...")
    try:
        # PCA visualization
        visualization(ori_data, generated_data, 'pca')
        plt.title("PCA: Real vs Generated Train Data")
        plt.savefig("timegan_pca_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # t-SNE visualization  
        visualization(ori_data, generated_data, 'tsne')
        plt.title("t-SNE: Real vs Generated Train Data")
        plt.savefig("timegan_tsne_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as PNG files")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Summary
    results = {
        'discriminative_score': avg_discriminative,
        'predictive_score': avg_predictive
    }
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Discriminative Score: {avg_discriminative:.4f}")
    print(f"Predictive Score: {avg_predictive:.4f}")
    print("="*50)
    
    return results

def compare_sequence_statistics(ori_data, generated_data):
    """
    Compare basic statistics between original and generated data.
    """
    print("\n" + "="*50) 
    print("SEQUENCE STATISTICS COMPARISON")
    print("="*50)
    
    # Sequence lengths
    ori_lengths = [seq.shape[0] for seq in ori_data]
    gen_lengths = [seq.shape[0] for seq in generated_data]
    
    print("Sequence Lengths:")
    print(f"  Original - Mean: {np.mean(ori_lengths):.2f}, Std: {np.std(ori_lengths):.2f}")
    print(f"  Generated - Mean: {np.mean(gen_lengths):.2f}, Std: {np.std(gen_lengths):.2f}")
    
    # Feature statistics (assuming 2 features: delay, delta_time)
    ori_all = np.concatenate(ori_data, axis=0)
    gen_all = np.concatenate(generated_data, axis=0)
    
    feature_names = ['delay_in_min', 'delta_departure_min']
    
    for i, feature_name in enumerate(feature_names):
        print(f"\n{feature_name}:")
        print(f"  Original - Mean: {np.mean(ori_all[:, i]):.3f}, Std: {np.std(ori_all[:, i]):.3f}")
        print(f"  Generated - Mean: {np.mean(gen_all[:, i]):.3f}, Std: {np.std(gen_all[:, i]):.3f}")

if __name__ == "__main__":
    # Run TimeGAN on your train data
    ori_data, generated_data = run_timegan_on_train_data(
        npz_path="preprocessed_normalized_routes_03_new.npz"
    )
    
    # Compare basic statistics
    compare_sequence_statistics(ori_data, generated_data)
    
    # Evaluate performance
    results = evaluate_timegan_performance(ori_data, generated_data, metric_iterations=3)
    
    # Simple visualization of a few sequences
    print("\nVisualizing sample sequences...")
    plt.figure(figsize=(15, 8))
    
    for i in range(3):  # Show first 3 sequences
        # Original data
        plt.subplot(2, 3, i+1)
        plt.plot(ori_data[i][:, 0], 'b-', label='Delay', marker='o')
        plt.plot(ori_data[i][:, 1], 'r-', label='Delta Time', marker='s')
        plt.title(f'Original Sequence {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Generated data
        plt.subplot(2, 3, i+4)
        plt.plot(generated_data[i][:, 0], 'b-', label='Delay', marker='o')
        plt.plot(generated_data[i][:, 1], 'r-', label='Delta Time', marker='s')
        plt.title(f'Generated Sequence {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("timegan_sequence_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Sequence comparison plot saved as 'timegan_sequence_comparison.png'")