# bitstar
python ompl_plan_general.py --model_path ../ompl_res/bitstar/s2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --env_type s2d \
--planner bitstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/s2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --env_type s2d \
--planner bitstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/c2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ --env_type c2d \
--planner bitstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/c2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ --env_type c2d \
--planner bitstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/r2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --env_type r2d \
--planner bitstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/r2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --env_type r2d \
--planner bitstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/r3d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ --env_type r3d \
--planner bitstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/bitstar/r3d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ --env_type r3d \
--planner bitstar --data_type unseen
# rrt*
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/s2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --env_type s2d \
--planner rrtstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/s2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --env_type s2d \
--planner rrtstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/c2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ --env_type c2d \
--planner rrtstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/c2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ --env_type c2d \
--planner rrtstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/r2d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --env_type r2d \
--planner rrtstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/r2d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --env_type r2d \
--planner rrtstar --data_type unseen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/r3d/ --env_idx 0 --path_idx 4000 \
--N 100 --NP 200 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ --env_type r3d \
--planner rrtstar --data_type seen
python ompl_plan_general.py --model_path ../ompl_res/rrtstar/r3d/ --env_idx 100 --path_idx 0 \
--N 10 --NP 2000 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ --env_type r3d \
--planner rrtstar --data_type unseen
