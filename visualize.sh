python ompl_plan.py --planner bitstar
python ompl_plan.py --planner informedrrtstar
python ompl_plan.py --planner rrtconnect
python ompl_plan.py --planner rrtstar
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 10 --path_idx 1253 --tree_path ../visual/bitstar_path_env10_path1253.graphml \
--N 1400 --dim 2 --encoding latin1 --model bitstar
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 10 --path_idx 1253 --tree_path ../visual/informedrrtstar_path_env10_path1253.graphml \
--N 1400 --dim 2 --encoding latin1 --model informedrrtstar
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 10 --path_idx 1253 --tree_path ../visual/rrtconnect_path_env10_path1253.graphml \
--N 1400 --dim 2 --encoding latin1 --model rrtconnect
python3 visualizer_tree.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 10 --path_idx 1253 --tree_path ../visual/rrtstar_path_env10_path1253.graphml \
--N 1400 --dim 2 --encoding latin1 --model rrtstar

python3 visualizer_mat.py --in_path ../data/simple/ --out_path ../visual/ \
--env_idx 10 --path_idx 1253 --line_path ../visual/mpnet_path_env10_path1253.p --n 4 --dim 2 --encoding latin1 \
--model mpnet
