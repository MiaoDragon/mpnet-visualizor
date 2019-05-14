python ompl_plan.py --planner bitstar --env_idx 29 --path_idx 4034
python ompl_plan.py --planner informedrrtstar --env_idx 29 --path_idx 4034
python ompl_plan.py --planner rrtconnect --env_idx 29 --path_idx 4034
python ompl_plan.py --planner rrtstar --env_idx 29 --path_idx 4034
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 29 --path_idx 4034 --tree_path ../visual/bitstar_path_env29_path4034.graphml \
--N 1400 --dim 2 --encoding latin1 --model bitstar
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 29 --path_idx 4034 --tree_path ../visual/informedrrtstar_path_env29_path4034.graphml \
--N 1400 --dim 2 --encoding latin1 --model informedrrtstar
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 29 --path_idx 4034 --tree_path ../visual/rrtconnect_path_env29_path4034.graphml \
--N 1400 --dim 2 --encoding latin1 --model rrtconnect
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 29 --path_idx 4034 --tree_path ../visual/rrtstar_path_env29_path4034.graphml \
--N 1400 --dim 2 --encoding latin1 --model rrtstar

python3 visualizer_mat.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 29 --path_idx 4034 --line_path ../visual/path_env_29_path4034.p --n 2 --dim 2 --encoding latin1 \
--model mpnet
