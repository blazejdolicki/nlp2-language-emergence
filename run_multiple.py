import subprocess


from datetime import datetime

now = datetime.now()

date_time = now.strftime("%d_%m_%Y_%H_%M_%S")

log_file = open(f"logs/log_{date_time}.txt", "w")

tasks = ["standard", "img_clas", "target_clas"]
n_imgs = [2]
same_class_probs = [0.0, 1.0]
for task in tasks:
    for num_imgs in n_imgs:
        for same_class_prob in same_class_probs:
            subprocess.call(["python", "main.py",  
                             "--task", task, 
                             "--num_imgs", str(num_imgs), 
                             "--same_class_prob", str(same_class_prob)], 
                            stdout=log_file)