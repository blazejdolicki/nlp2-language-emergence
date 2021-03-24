import subprocess
from datetime import datetime

now = datetime.now()

date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
log_file = open(f"logs/log_{date_time}.txt", "w")

tasks = ["standard", "img_clas", "target_clas"]
n_imgs = [2]
same_class_probs = [0.0, 0.5, 1.0]
seeds = [7, 122, 809]
for task in tasks:
    for num_imgs in n_imgs:
        for same_class_prob in same_class_probs:
            for seed in seeds:
                subprocess.call(["python", "main.py",  
                                "--task", task, 
                                "--num_imgs", str(num_imgs), 
                                "--same_class_prob", str(same_class_prob),
                                "--seed",str(seed)], 
                                stdout=log_file)
            
with open(f"logs/log_{date_time}.txt", "r") as f:
    log = f.read()

log_paths = [run.split()[-1] for run in log.split("\n") if "Log path" in run]
print("Log paths:", log_paths)