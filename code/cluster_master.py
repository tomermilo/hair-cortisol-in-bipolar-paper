import os
import argparse
import subprocess
import pickle
import numpy as np


if __name__ == '__main__':
    
    N = 1000
    
    parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '-I',
#         dest='default_config_filepath',
#         default='./configurations/default_no_policy',
#         help='default parameters file path for all runs, if empty takes the default no policy parameters file',
#     )
    parser.add_argument('output_directory', help='output directory to store the results')
    parser.add_argument('-D', dest='decay_coef_filepath', default='', help='file path for decaying coefficeints, saved as pickle file')
    args = parser.parse_args()
    
    res_directory = os.path.join('..', 'results', 'continous_input', args.output_directory)
    if not os.path.exists(res_directory):
        os.mkdir(res_directory)
        
    sampled_alphas = [None] * N
    if args.decay_coef_filepath != '':
        with open(args.decay_coef_filepath, 'rb') as f:
            experimental_alphas = pickle.load(f)[:, 1]
        sampled_alphas = np.random.choice(experimental_alphas, size=N)
    
    for repeat, alpha in enumerate(sampled_alphas):
        queue = 'alon'
        queue = 'new-short'
        output_file_path = os.path.join(res_directory, str(repeat))
        #cluster_command = 'bsub -q infiniband -o out -R "select[mem>1GB] -R rusage[mem=1000]"'
        #cluster_command = 'bsub -q new-short -o out -R "select[mem>1GB] -R rusage[mem=1000]"'
        cluster_command = f'bsub -q {queue} -o out -R select[mem>1GB] -R rusage[mem=1000]'
        #cluster_command = 'bsub -o out -R select[mem>1GB] -R rusage[mem=1000]'
        
#        if alpha is None:
#            python_command = '"python hpa_simulation.py {}"'.format(output_file_path)
#        else:
#            python_command = '"python hpa_simulation.py {} -A {}"'.format(output_file_path, alpha)
        
        #subprocess.run(' '.join([cluster_command, python_command]).split())
        
        if alpha is None:
            subprocess.run(args=f'{cluster_command} python hpa_simulation.py {output_file_path} -R {repeat}'.split())
        else:
            subprocess.run(args=f'{cluster_command} python hpa_simulation.py {output_file_path} -A {alpha} -R {repeat}'.split())