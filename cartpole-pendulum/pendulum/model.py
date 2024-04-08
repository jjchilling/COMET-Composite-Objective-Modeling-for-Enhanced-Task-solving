import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int

    @nn.compact
    def __call__(self, x, rng):
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.num_output_units)(x)
        return x

def initialize_model(rng):
    model = MLP(48, 1, 1)
    policy_params = model.init(rng, jnp.zeros(3), None)
    return model, policy_params
