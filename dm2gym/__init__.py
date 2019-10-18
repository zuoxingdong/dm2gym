from gym.envs import register
from dm_control import suite


for domain_name, task_name in suite.ALL_TASKS:
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-v0'
    register(id=ID, 
             entry_point='dm2gym.envs:DMSuiteEnv', 
             kwargs={'domain_name': domain_name, 'task_name': task_name}, 
             max_episode_steps=1000)
