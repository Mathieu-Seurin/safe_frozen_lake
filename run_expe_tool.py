import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

def test_algo(algo, game, repeat=1):
    count_optimal_policy = 0
    count_goal_reach = 0

    feedback_list = np.zeros(repeat)
    num_steps_list = np.zeros(repeat)

    step_to_reach_goal_list = np.ones(repeat) * game.max_steps

    for test in range(repeat):

        num_steps_this_ep = 0
        num_feedback_this_ep = 0

        state = game.reset()
        done = False

        while not done:
            action = algo.choose_action_greedy(state)
            state, reward, done, info = game.step(action)

            num_steps_this_ep += 1
            num_feedback_this_ep += state['gave_feedback']

        feedback_list[test] = num_feedback_this_ep
        num_steps_list[test] = num_steps_this_ep

        if reward == 1:
            count_goal_reach += 1
            step_to_reach_goal_list[test] = num_steps_this_ep

        # if last reward is 1, we consider that the environment is "solved"
        # if the algo solve the algo N times in a row, '''with the best policy''' => it's okay ! (N = success_to_win)
        if reward == 1 and num_steps_this_ep <= game.shortest_path_length:
            count_optimal_policy += 1

    return count_optimal_policy, count_goal_reach, feedback_list.mean(), num_steps_list.mean(), step_to_reach_goal_list.mean()



def run_expe(create_algo, game, n_episodes_train, n_tests, percentage_to_success, verbose=True):
    first_ep_where_goal_is_reached = n_episodes_train

    algo = create_algo()

    total_training_steps = 0
    total_feedback_list_train = []
    total_feedback_list_test = []
    percentage_goal_reach_list = []
    num_steps_list = []
    steps_to_reach_goal_list = []

    for ep in range(n_episodes_train):

        assert algo.current_eps >= algo.minimum_epsilon, "Failed to change epsilon"
        state = game.reset()
        done = False

        # Init information to save per episode
        cumul_reward = 0
        num_steps_this_ep = 0
        feedback_train_this_ep = 0

        while not done:
            action = algo.choose_action_eps_greedy(state)
            next_state, reward, done, info = game.step(action)
            algo.optimize(state, action, next_state, reward)

            cumul_reward += reward
            state = next_state
            num_steps_this_ep += 1
            feedback_train_this_ep += next_state['gave_feedback']

            # if next_state['gave_feedback']:
            #     print("fais pas ça petit con")

        total_training_steps += num_steps_this_ep
        if reward == 1:
            first_ep_where_goal_is_reached = min(first_ep_where_goal_is_reached, ep)

        if verbose:
            print("End of ep #{:05d} in {:03d} steps, cumul_reward = {}, current_eps = {}".format(
                ep, num_steps_this_ep, cumul_reward, algo.current_eps))

        total_feedback_list_train.append(feedback_train_this_ep)

        count_optimal_policy, count_goal_reach, n_feedback, num_steps, steps_to_reach_goal \
            = test_algo(algo, game, repeat=n_tests)

        total_feedback_list_test.append(n_feedback)
        percentage_goal_reach_list.append(count_goal_reach / n_tests)
        num_steps_list.append(num_steps)
        steps_to_reach_goal_list.append(steps_to_reach_goal)

        if count_optimal_policy / n_tests >= percentage_to_success:
            break

    run_info = dict()
    run_info['episode_first_success'] = first_ep_where_goal_is_reached
    run_info['last_ep_number'] = ep
    run_info['final_epsilon'] = algo.current_eps
    run_info['total_training_steps'] = total_training_steps
    run_info['feedback_number_test_list'] = total_feedback_list_test
    run_info['feedback_number_train_list'] = total_feedback_list_train
    run_info['num_steps_list'] = num_steps_list
    run_info['percentage_goal_reach_list'] = percentage_goal_reach_list
    run_info['steps_to_reach_goal'] = steps_to_reach_goal_list

    return run_info


def run_multiple_expe(create_algo, game, n_expe, n_episodes_train, n_tests, percentage_to_success, show=False):
    mean_num_ep = np.zeros(n_expe)

    num_feedback_per_test_ep = np.zeros((n_expe, n_episodes_train))
    num_feedback_per_train_ep = np.zeros((n_expe, n_episodes_train))

    num_steps_array = np.zeros((n_expe, n_episodes_train))
    steps_to_reach_goal_array = np.zeros((n_expe, n_episodes_train))

    percentage_goal_reach_array = np.zeros((n_expe, n_episodes_train))

    total_steps_array = np.zeros(n_expe)

    for i in range(n_expe):
        run_info = run_expe(create_algo, game, n_episodes_train=n_episodes_train, verbose=False,
                            n_tests=n_tests, percentage_to_success=percentage_to_success)
        mean_num_ep[i] = run_info['last_ep_number']

        num_feedback_test_list = run_info['feedback_number_test_list']
        num_feedback_per_test_ep[i, :len(num_feedback_test_list)] = num_feedback_test_list

        num_feedback_train_list = run_info['feedback_number_train_list']
        num_feedback_per_train_ep[i, :len(num_feedback_train_list)] = num_feedback_train_list

        num_steps_list = run_info['num_steps_list']
        num_steps_array[i, :len(num_steps_list)] = num_steps_list
        num_steps_array[i, len(num_steps_list):] = num_steps_list[-1]

        steps_to_reach_goal_list = run_info['steps_to_reach_goal']
        steps_to_reach_goal_array[i, :len(steps_to_reach_goal_list)] = steps_to_reach_goal_list
        steps_to_reach_goal_array[i, len(steps_to_reach_goal_list):] = steps_to_reach_goal_list[-1]

        percentage_goal_reach_list = run_info['percentage_goal_reach_list']
        percentage_goal_reach_array[i, :len(percentage_goal_reach_list)] = percentage_goal_reach_list
        percentage_goal_reach_array[i, len(percentage_goal_reach_list):] = percentage_goal_reach_list[-1]

        total_steps_array[i] = run_info["total_training_steps"]

    if show:
        print("Total steps to solve environment, mean {}, std {}".format(total_steps_array.mean(),
                                                                         total_steps_array.std()))
        print("Number of episodes mean {}, std {}".format(np.mean(mean_num_ep), np.std(mean_num_ep)))

        print("Number of failure to solve env", np.count_nonzero(mean_num_ep == n_episodes_train - 1))

        episodes_of_non_failure = mean_num_ep[np.where(mean_num_ep < n_episodes_train - 1)]
        print("Number of episodes when not failing", episodes_of_non_failure.mean())

        sns.tsplot(data=num_feedback_per_test_ep)
        sns.tsplot(data=np.zeros(n_episodes_train))
        plt.title("Number of time feedback is being used during TEST")
        plt.show()

        sns.tsplot(data=num_feedback_per_train_ep)
        sns.tsplot(data=np.zeros(n_episodes_train))
        plt.title("Number of time feedback is being used during TRAIN")
        plt.show()

        sns.tsplot(data=steps_to_reach_goal_array)
        sns.tsplot(data=np.zeros(n_episodes_train))
        plt.title("Number of steps to reach goal")
        plt.show()

        sns.tsplot(data=percentage_goal_reach_array)
        sns.tsplot(data=np.zeros(n_episodes_train))
        sns.tsplot(data=np.ones(n_episodes_train))
        plt.title("Percentage of goal reach when testing")
        plt.show()

    res_dict = dict()
    res_dict['total_steps_array'] = total_steps_array
    res_dict['num_feedback_per_train_ep'] = num_feedback_per_train_ep
    res_dict['num_feedback_per_test_ep'] = num_feedback_per_test_ep
    res_dict['num_steps_array'] = num_feedback_per_test_ep
    res_dict['percentage_goal_reach_array'] = percentage_goal_reach_array
    res_dict['steps_to_reach_goal'] = steps_to_reach_goal_array

    return res_dict


def compare_algorithms(list_of_game_algo, n_expe, n_episodes_train, n_tests, percentage_to_success):
    """
    :param list_of_game_algo: a list of triplet (game, algo, name)
    :param n_expe: number of experiment you need to run, to compute algorithm performances, the more expe, the more precise
    :param n_episodes_train: maximum number of episodes to train your algo
    :param n_tests: number of tests PER EPISODES (to see the evolution of your algorithm perf)
    :param percentage_to_success: percentage of goal reach with "optimal policy" to consider the problem as solved
    """

    palette = sns.color_palette('colorblind')
    assert len(list_of_game_algo) <= len(palette)-1, "More game than {} games ({})=> visual are fucked up".format(
        len(palette), len(list_of_game_algo))


    # GATHER EXPE AND RESULTS
    result_dict = OrderedDict()
    for game, algo_creator, name in list_of_game_algo:
        result_dict[name] = run_multiple_expe(algo_creator, game, n_expe, n_episodes_train, n_tests, percentage_to_success, show=False)
        print("Finished {}".format(name))

    expe_dict = dict()
    expe_dict['result_dict'] = result_dict
    expe_dict['minimum_number_of_steps'] = game.shortest_path_length

    plot_result(expe_dict)

    return expe_dict


def plot_result(expe_dict):
    palette = sns.color_palette('colorblind')
    result_dict = expe_dict['result_dict']

    #  PLOT NUMBER OF FEEDBACK GIVEN BY ENV DURING test
    for n_item, (name, result) in enumerate(result_dict.items()):
        sns.tsplot(data=result['num_feedback_per_test_ep'], color=palette[n_item], legend=True)

    plt.legend(handles=plt.gca().get_lines(),
               labels=result_dict.keys())  # Handles is to avoid the blurry effect of snsplot
    plt.title("Num Feedback During Test")
    plt.show()

    #  PLOT % of goal reach during test
    for n_item, (name, result) in enumerate(result_dict.items()):
        sns.tsplot(data=result['percentage_goal_reach_array'], color=palette[n_item], legend=True)

    plt.legend(handles=plt.gca().get_lines(),
               labels=result_dict.keys())  # Handles is to avoid the blurry effect of snsplot
    plt.title("Percentage goal reach during test")
    plt.show()

    # Plot mean number of steps to reach goal
    for n_item, (name, result) in enumerate(result_dict.items()):
        sns.tsplot(data=result['steps_to_reach_goal'], color=palette[n_item], legend=True)

    approx_min_steps_to_reach_goal = np.ones(result['steps_to_reach_goal'].shape[1]) * expe_dict[
        'minimum_number_of_steps']
    sns.tsplot(data=approx_min_steps_to_reach_goal, color=palette[-1])
    legend = list(result_dict.keys())
    legend.append("Estimate minimum number of steps to reach goal")

    h = plt.gca().get_lines()
    plt.legend(handles=h, labels=legend)

    plt.title("Number of steps to reach goal")
    plt.show()










