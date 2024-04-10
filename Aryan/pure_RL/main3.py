import gymnax

# Define the name of the Gymnax environment
env_name = "PointRobot-misc"

# Create the Gymnax environment using gymnax.make
env, env_params = gymnax.make(env_name)

# Check the type of action space
action_space = env.action_space

# Get the type of action space
action_space_type = type(env.action_space())

# Check for different types of action spaces
print(action_space_type)
if action_space_type == gymnax.environments.spaces.Discrete:
    print("The environment has a Discrete action space.")
    num_actions = action_space.n
    print("Number of discrete actions:", num_actions)
elif action_space_type == gymnax.environments.spaces.Box:
    print("The environment has a Continuous (Box) action space.")
elif action_space_type == gymnax.environments.spaces.Tuple:
    discrete_dimensions = all(isinstance(space, gymnax.environments.spaces.Discrete) for space in action_space.spaces)
    if discrete_dimensions:
        print("The environment has a Tuple action space with discrete dimensions.")
        for i, subspace in enumerate(action_space.spaces):
            print(f"Subspace {i}: Type: {type(subspace)}, Number of Actions: {subspace.n}")
    else:
        print("The environment has a Tuple action space with mixed types.")
else:
    print("The environment has another type of action space.")
