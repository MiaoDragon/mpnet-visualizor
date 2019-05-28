python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4035 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model informedrrtstar
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4035 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model rrtconnect
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4035 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model rrtstar

python3 visualizer_mat.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4035 --line_path ../visual_input/ --n 6 --dim 2 --encoding latin1 \
--model mpnet
