#python ompl_plan.py --planner bitstar --env_idx 8 --path_idx 4094 --data_path ../data/c2d/
python ompl_plan.py --planner informedrrtstar --env_idx 8 --path_idx 4094 --data_path ../data/c2d/
python ompl_plan.py --planner rrtconnect --env_idx 8 --path_idx 4094 --data_path ../data/c2d/
python ompl_plan.py --planner rrtstar --env_idx 8 --path_idx 4094 --data_path ../data/c2d/
#python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
#--env_idx 8 --path_idx 4094 --tree_path ../visual/ \
#--N 1400 --dim 2 --encoding latin1 --model bitstar
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4094 --tree_path ../visual/ \
--N 1400 --dim 2 --encoding latin1 --model informedrrtstar
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4094 --tree_path ../visual/ \
--N 1400 --dim 2 --encoding latin1 --model rrtconnect
python3 visualizer_tree.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4094 --tree_path ../visual/ \
--N 1400 --dim 2 --encoding latin1 --model rrtstar

python3 visualizer_mat.py --in_path ../data/c2d/ --out_path ../visual/ \
--env_idx 8 --path_idx 4094 --line_path ../visual/ --n 6 --dim 2 --encoding latin1 \
--model mpnet
