from signalpilot.environment import Action, EnvironmentConfig, IntersectionEnvironment


def test_vehicles_discharge_only_from_active_movement() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(
            arrival_rates=(0.0, 0.0, 0.0, 0.0),
            initial_queues=(2, 0, 3, 0),
            saturation_flow=1,
        ),
        seed=4,
    )

    result = environment.step(Action.KEEP_PHASE)

    assert result.info["served"] == (1, 0, 0, 0)
    assert result.state.queues == (1, 0, 3, 0)
    assert result.state.total_served == 1


def test_waiting_time_counts_vehicles_remaining_after_discharge() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(
            arrival_rates=(0.0, 0.0, 0.0, 0.0),
            initial_queues=(2, 0, 0, 0),
            saturation_flow=1,
        ),
        seed=4,
    )

    result = environment.step(Action.KEEP_PHASE)

    assert result.info["waiting_increment"] == 1
    assert result.state.total_waiting_time == 1
    assert result.reward == -1.0


def test_vehicle_conservation_holds_for_seeded_arrivals() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(
            episode_horizon=10,
            arrival_rates=(0.7, 0.4, 0.5, 0.2),
            saturation_flow=1,
        ),
        seed=12,
    )

    initial_total = sum(environment.state.queues)
    for _ in range(10):
        result = environment.step(Action.KEEP_PHASE)
        state = result.state
        assert initial_total + state.total_arrivals == (
            state.total_served + sum(state.queues)
        )
