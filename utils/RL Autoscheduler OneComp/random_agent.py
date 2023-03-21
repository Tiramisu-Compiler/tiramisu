from Environment_sparse import SearchSpaceSparse
import random, os

tiramisu_path = '/scratch/ne2128/tiramisu/' # Put the path to your tiramisu installation here
os.environ['TIRAMISU_ROOT'] = tiramisu_path

env=SearchSpaceSparse("../new_dataset2.json","../Dataset")


f=open("./output_results.txt","w")

#A random agent to test the environment independently from the model
for i in range(1000):
    f.write("---------- Episode {} ------------------\n\n".format(i))
    obs=env.reset()
    done=False
    step=0
    while not done:
        f.write("---------- Step {} ------------------\n".format(step))
        raw_action=random.randint(0,6)
        print("the raw action is {}\n".format(raw_action))
        f.write("Action {}: \n".format(raw_action))
        obs, reward, done, info = env.step(raw_action)
        
        f.write("Schedule str: {}\n".format(env.schedule_str))
        f.write("Schedule: {}\n".format(env.schedule))
        f.write("Observation: {}\n".format(obs))
        f.write("Reward: {}\n".format(reward))
        f.write("Done: {}\n".format(done))
        f.write("Info: {}\n".format(info))
        step+=1
       
