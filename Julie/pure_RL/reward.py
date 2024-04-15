#    def step_env(
#         self,
#         key: chex.PRNGKey,
#         state: EnvState,
#         action: Union[int, float, chex.Array],
#         params: EnvParams,
#     ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
#         """Sample bernoulli reward, increase counter, construct input."""
#         a = jnp.clip(action, -params.max_force, params.max_force)
#         pos = state.pos + a
#         goal_distance = jnp.linalg.norm(state.goal - state.pos)
#         goal_reached = goal_distance <= params.goal_radius
#         # Dense reward - distance to goal, sparse reward - 1 if in radius
#         reward = jax.lax.select(params.dense_reward, -goal_distance, goal_reached * 1.0)
#         sampled_pos = sample_agent_position(
#             key, params.circle_radius, params.center_init
#         )[0]  # Ensure sampled_pos is a 1-dimensional array
#         # Sample/set new initial position if goal was reached
#         new_pos = jax.lax.select(goal_reached, sampled_pos, pos)
#         state = EnvState(
#             last_action=action,
#             last_reward=reward,
#             pos=new_pos,
#             goal=state.goal,
#             goals_reached=state.goals_reached + goal_reached,
#             time=state.time + 1,
#         )

#         done = self.is_terminal(state, params)
#         return (
#             lax.stop_gradient(self.get_obs(state, params)),
#             lax.stop_gradient(state),
#             reward,
#             done,
#             {"discount": self.discount(state, params)},
#         )