from gymnax.experimental import RolloutWrapper

def create_rollout_manager(env_name, model_apply):
    manager = RolloutWrapper(model_apply, env_name=env_name)
    return manager
