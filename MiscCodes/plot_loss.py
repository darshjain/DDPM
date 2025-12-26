import re
import matplotlib.pyplot as plt

def plot_loss(log_file_path, output_image_path):
    loss_data = {}  # Dictionary to store {iteration: loss_value}
    
    # Regex to find 'loss: <number>' and iteration number
    # Format: loss: 0.0513:  47%|████▋     | 13999/30000
    loss_pattern = re.compile(r'loss:\s*(\d+\.?\d*):.*?\|\s*(\d+)/\d+')
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Find matches with both loss and iteration number
                match = loss_pattern.search(line)
                if match:
                    loss_value = float(match.group(1))
                    iteration = int(match.group(2))
                    # Store the loss value for this iteration (will overwrite if same iteration appears multiple times)
                    loss_data[iteration] = loss_value
        
        if not loss_data:
            print("No loss values found in the file.")
            return
        
        # Sort by iteration number and extract values
        sorted_iterations = sorted(loss_data.keys())
        loss_values = [loss_data[it] for it in sorted_iterations]
        
        # Find the iteration with minimum loss
        min_loss = min(loss_values)
        min_loss_idx = loss_values.index(min_loss)
        best_iteration = sorted_iterations[min_loss_idx]

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_image_path)
        print(f"Loss plot saved to {output_image_path}")
        print(f"Found {len(loss_values)} unique data points (iterations).")
        print(f"Iteration range: {sorted_iterations[0]} to {sorted_iterations[-1]}")
        print(f"\n*** Best Iteration ***")
        print(f"Lowest loss: {min_loss:.6f} at iteration {best_iteration}")

    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    log_file = '/home/zjy6us/mlia/logs/mliav3.err'
    output_file = '/home/zjy6us/mlia/mliav3_loss_plot.png'
    plot_loss(log_file, output_file)
