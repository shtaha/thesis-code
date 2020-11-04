import grid2op
import numpy as np

from lib.visualizer import describe_environment, print_dict

if __name__ == "__main__":
    # env_name = "l2rpn_wcci_2020"
    # env_name = "rte_case5_example"
    env_name = "l2rpn_2019"
    env = grid2op.make(dataset=env_name)

    describe_environment(env)
    obs = env.reset()
    action = env.action_space({})

    obs, reward, done, info = env.step(action)

    # print(info)
    # print(done)
    # print(reward)
    #
    # print("year", obs.year)
    # print("month", obs.month)
    # print("day", obs.day)
    # print("hour", obs.hour_of_day)
    # print("minute", obs.minute_of_hour)
    # print("week day", obs.day_of_week)
    # print("prod")
    # print("p", obs.prod_p)
    # print("q", obs.prod_q)
    # print("v", obs.prod_v)
    # print("load")
    # print("p", obs.load_p)
    # print("q", obs.load_q)
    # print("v", obs.load_v)
    # print("line or")
    # print("p", obs.p_or)
    # print("q", obs.q_or)
    # print("v", obs.v_or)
    # print("a", obs.a_or)
    # print("line ex")
    # print("p", obs.p_ex)
    # print("q", obs.q_ex)
    # print("v", obs.v_ex)
    # print("a", obs.a_ex)
    # print("rho", obs.rho)
    # print("topology", obs.topo_vect)
    # print("line status", obs.line_status)
    # print("line in overflow", obs.timestep_overflow)
    # print("line cooldown time", obs.time_before_cooldown_line)
    # print("sub cooldown time", obs.time_before_cooldown_sub)
    # print("next maintenance", obs.time_next_maintenance)
    # print("maintenance duration", obs.duration_next_maintenance)
    # print("dispatch")
    # print("target", obs.target_dispatch)
    # print("actual", obs.actual_dispatch)
    #
    # print("forecast")
    # simobs, simr, simd, siminfo = obs.simulate(env.action_space())
    # prod_p_f, prod_v_f, load_p_f, load_q_f = obs.get_forecasted_inj(time_step=1)
    # print("prod_p", prod_p_f)
    # print("prod_v", prod_v_f)
    # print("load_p", load_p_f)
    # print("load_q", load_q_f)

    print(reward)
    print("backend", env.backend.get_line_flow())
    print("obs", obs.a_or)

    relative_flows = np.divide(np.abs(obs.a_or), env.get_thermal_limit() + 1e-1)
    line_margins = np.maximum(0.0, 1.0 - relative_flows)
    line_scores = 1.0 - np.square(1.0 - line_margins)
    print(line_scores)
    print(line_scores.sum())

    # print_dict(obs.to_dict())
