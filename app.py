"""Streamlit batch-run interface for SignalPilot.

Run with ``streamlit run app.py`` after installing the optional UI dependency.
"""

from signalpilot.agents import (
    FixedTimeController,
    QLearningController,
    RandomController,
)
from signalpilot.environment import (
    DemandScenario,
    IntersectionEnvironment,
    scenario_config,
)
from signalpilot.training import compare_controllers
from signalpilot.visualization import (
    plot_intersection,
    plot_queue_lengths,
    plot_signal_phase,
    plot_waiting_distribution,
    record_episode,
)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SignalPilot", layout="wide")
    st.title("SignalPilot traffic signal simulator")
    scenario = st.selectbox("Demand scenario", list(DemandScenario), format_func=str)
    controller_name = st.selectbox(
        "Controller", ("fixed-time", "random", "q-learning", "compare")
    )
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    horizon = st.number_input("Episode horizon", min_value=1, value=60, step=1)

    if st.button("Run episode"):
        config = scenario_config(scenario, episode_horizon=int(horizon))
        if controller_name == "compare":
            reports = compare_controllers(
                {
                    "fixed-time": FixedTimeController,
                    "random": lambda: RandomController(seed=int(seed)),
                    "q-learning": lambda: QLearningController(
                        epsilon=0.0, seed=int(seed)
                    ),
                },
                lambda selected: scenario_config(
                    selected,
                    episode_horizon=int(horizon),
                ),
                scenarios=(scenario,),
                seeds=(int(seed),),
            )
            st.json(
                {
                    name: {
                        key: {
                            metric: summary.__dict__
                            for metric, summary in values.items()
                        }
                        for key, values in report.scenarios.items()
                    }
                    for name, report in reports.items()
                }
            )
            return
        if controller_name == "fixed-time":
            controller = FixedTimeController()
        elif controller_name == "random":
            controller = RandomController(seed=int(seed))
        else:
            controller = QLearningController(epsilon=0.0, seed=int(seed))
        trace = record_episode(
            IntersectionEnvironment(config, seed=int(seed)),
            controller,
            seed=int(seed),
        )
        st.pyplot(plot_intersection(trace.states[-1]), clear_figure=True)
        st.pyplot(
            plot_queue_lengths([state.queues for state in trace.states]),
            clear_figure=True,
        )
        st.pyplot(plot_signal_phase(trace.phases), clear_figure=True)
        st.pyplot(plot_waiting_distribution(trace.waiting_times), clear_figure=True)
        st.json(trace.metrics.__dict__)


if __name__ == "__main__":
    main()
