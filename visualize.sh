python ompl_plan.py --env_type c2d --planner bitstar --env_idx 1 --path_idx 4176 --data_path ../data/c2d/ --model_path ../visual_input/
python ompl_plan.py --env_type c2d --planner informedrrtstar --env_idx 1 --path_idx 4176 --data_path ../data/c2d/ --model_path ../visual_input/
python ompl_plan.py --env_type c2d --planner rrtconnect --env_idx 1 --path_idx 4176 --data_path ../data/c2d/ --model_path ../visual_input/
python ompl_plan.py --env_type c2d --planner rrtstar --env_idx 1 --path_idx 4176 --data_path ../data/c2d/ --model_path ../visual_input/
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 1 --path_idx 4176 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model bitstar
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 1 --path_idx 4176 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model informedrrtstar
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 1 --path_idx 4176 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model rrtconnect
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 1 --path_idx 4176 --tree_path ../visual_input/ \
--N 1400 --dim 2 --encoding latin1 --model rrtstar

python3 visualizer_mat.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 1 --path_idx 4176 --line_path ../visual_input/ --n 6 --dim 2 --encoding latin1 \
--model mpnet
