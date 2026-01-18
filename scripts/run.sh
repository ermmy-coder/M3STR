# Evaluation scripts are in this format.
nohup python run_task1_evaluation_vllm.py\
    --model_type instructblip \
    --model_used all \
    --cuda 2 > log_instructblip3.txt &


nohup python run_task1_evaluation_vllm.py\
    --model_type internvl \
    --model_used all \
    --cuda 4 > log_internvl.txt &


nohup python run_task1_evaluation_vllm.py\
    --model_type minicpm \
    --model_used all \
    --cuda 2 > log_minicpm.txt &


nohup python run_task1_evaluation_qwen_vl.py\
    --model_type qwen2-vl \
    --model_used all \
    --cuda 3 > log_qwen2_vl2.txt &
