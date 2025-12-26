import sys
from eval_utils import run_evaluation

if __name__ == '__main__':
    run_evaluation(
        name='Original (mliav4)', 
        folder_path='/home/zjy6us/mlia/results_combinedV4-mild-augmented-more-steps', 
        dim=64
    )

