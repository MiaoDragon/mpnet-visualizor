python ompl_plan.py --planner bitstar
python ompl_plan.py --planner informedrrtstar
python ompl_plan.py --planner rrtstar
python3 visualizer_mat.py --in_path ../data/simple/obs_cloud/ --out_path ../visual/ \
--env 10 --path_i 1253 --line_path ../visual/bitstar_path_env10_path1253.p --N 1400 --dim 2 --encoding latin1 \
--model bitstar
python3 visualizer_mat.py --in_path ../data/simple/obs_cloud/ --out_path ../visual/ \
--env 10 --path_i 1253 --line_path ../visual/informedrrtstar_path_env10_path1253.p --N 1400 --dim 2 --encoding latin1 \
--model informedrrtstar
python3 visualizer_mat.py --in_path ../data/simple/obs_cloud/ --out_path ../visual/ \
--env 10 --path_i 1253 --line_path ../visual/rrtstar_path_env10_path1253.p --N 1400 --dim 2 --encoding latin1 \
--model rrtstar
python3 visualizer_mat.py --in_path ../data/simple/obs_cloud/ --out_path ../visual/ \
--env 10 --path_i 1253 --line_path ../visual/mpnet_path_env10_path1253.p --N 1400 --dim 2 --encoding latin1 \
--model mpnet
