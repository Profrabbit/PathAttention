# import pygments
# import pygments.lexers

code = "def learn(env,\n          network,\n          seed=None,\n          lr=5e-4,\n          total_timesteps=100000,\n          buffer_size=50000,\n          exploration_fraction=0.1,\n          exploration_final_eps=0.02,\n          train_freq=1,\n          batch_size=32,\n          print_freq=100,\n          checkpoint_freq=10000,\n          checkpoint_path=None,\n          learning_starts=1000,\n          gamma=1.0,\n          target_network_update_freq=500,\n          prioritized_replay=False,\n          prioritized_replay_alpha=0.6,\n          prioritized_replay_beta0=0.4,\n          prioritized_replay_beta_iters=None,\n          prioritized_replay_eps=1e-6,\n          param_noise=False,\n          callback=None,\n          load_path=None,\n          **network_kwargs\n            ):\n    \"\"\"Train a deepq model.\n\n    Parameters\n    -------\n    env: gym.Env\n        environment to train on\n    network: string or a function\n        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models\n        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which\n        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)\n    seed: int or None\n        prng seed. The runs with the same seed \"should\" give the same results. If None, no seeding is used.\n    lr: float\n        learning rate for adam optimizer\n    total_timesteps: int\n        number of env steps to optimizer for\n    buffer_size: int\n        size of the replay buffer\n    exploration_fraction: float\n        fraction of entire training period over which the exploration rate is annealed\n    exploration_final_eps: float\n        final value of random action probability\n    train_freq: int\n        update the model every `train_freq` steps.\n        set to None to disable printing\n    batch_size: int\n        size of a batched sampled from replay buffer for training\n    print_freq: int\n        how often to print out training progress\n        set to None to disable printing\n    checkpoint_freq: int\n        how often to save the model. This is so that the best version is restored\n        at the end of the training. If you do not wish to restore the best version at\n        the end of the training set this variable to None.\n    learning_starts: int\n        how many steps of the model to collect transitions for before learning starts\n    gamma: float\n        discount factor\n    target_network_update_freq: int\n        update the target network every `target_network_update_freq` steps.\n    prioritized_replay: True\n        if True prioritized replay buffer will be used.\n    prioritized_replay_alpha: float\n        alpha parameter for prioritized replay buffer\n    prioritized_replay_beta0: float\n        initial value of beta for prioritized replay buffer\n    prioritized_replay_beta_iters: int\n        number of iterations over which beta will be annealed from initial value\n        to 1.0. If set to None equals to total_timesteps.\n    prioritized_replay_eps: float\n        epsilon to add to the TD errors when updating priorities.\n    param_noise: bool\n        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)\n    callback: (locals, globals) -> None\n        function called at every steps with state of the algorithm.\n        If callback returns true training stops.\n    load_path: str\n        path to load the model from. (default: None)\n    **network_kwargs\n        additional keyword arguments to pass to the network builder.\n\n    Returns\n    -------\n    act: ActWrapper\n        Wrapper over act function. Adds ability to save it and load it.\n        See header of baselines/deepq/categorical.py for details on the act function.\n    \"\"\"\n    # Create all the functions necessary to train the model\n\n    sess = get_session()\n    set_global_seeds(seed)\n\n    q_func = build_q_func(network, **network_kwargs)\n\n    # capture the shape outside the closure so that the env object is not serialized\n    # by cloudpickle when serializing make_obs_ph\n\n    observation_space = env.observation_space\n    def make_obs_ph(name):\n        return ObservationInput(observation_space, name=name)\n\n    act, train, update_target, debug = deepq.build_train(\n        make_obs_ph=make_obs_ph,\n        q_func=q_func,\n        num_actions=env.action_space.n,\n        optimizer=tf.train.AdamOptimizer(learning_rate=lr),\n        gamma=gamma,\n        grad_norm_clipping=10,\n        param_noise=param_noise\n    )\n\n    act_params = {\n        'make_obs_ph': make_obs_ph,\n        'q_func': q_func,\n        'num_actions': env.action_space.n,\n    }\n\n    act = ActWrapper(act, act_params)\n\n    # Create the replay buffer\n    if prioritized_replay:\n        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)\n        if prioritized_replay_beta_iters is None:\n            prioritized_replay_beta_iters = total_timesteps\n        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,\n                                       initial_p=prioritized_replay_beta0,\n                                       final_p=1.0)\n    else:\n        replay_buffer = ReplayBuffer(buffer_size)\n        beta_schedule = None\n    # Create the schedule for exploration starting from 1.\n    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),\n                                 initial_p=1.0,\n                                 final_p=exploration_final_eps)\n\n    # Initialize the parameters and copy them to the target network.\n    U.initialize()\n    update_target()\n\n    episode_rewards = [0.0]\n    saved_mean_reward = None\n    obs = env.reset()\n    reset = True\n\n    with tempfile.TemporaryDirectory() as td:\n        td = checkpoint_path or td\n\n        model_file = os.path.join(td, \"model\")\n        model_saved = False\n\n        if tf.train.latest_checkpoint(td) is not None:\n            load_variables(model_file)\n            logger.log('Loaded model from {}'.format(model_file))\n            model_saved = True\n        elif load_path is not None:\n            load_variables(load_path)\n            logger.log('Loaded model from {}'.format(load_path))\n\n\n        for t in range(total_timesteps):\n            if callback is not None:\n                if callback(locals(), globals()):\n                    break\n            # Take action and update exploration to the newest value\n            kwargs = {}\n            if not param_noise:\n                update_eps = exploration.value(t)\n                update_param_noise_threshold = 0.\n            else:\n                update_eps = 0.\n                # Compute the threshold such that the KL divergence between perturbed and non-perturbed\n                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).\n                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017\n                # for detailed explanation.\n                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))\n                kwargs['reset'] = reset\n                kwargs['update_param_noise_threshold'] = update_param_noise_threshold\n                kwargs['update_param_noise_scale'] = True\n            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]\n            env_action = action\n            reset = False\n            new_obs, rew, done, _ = env.step(env_action)\n            # Store transition in the replay buffer.\n            replay_buffer.add(obs, action, rew, new_obs, float(done))\n            obs = new_obs\n\n            episode_rewards[-1] += rew\n            if done:\n                obs = env.reset()\n                episode_rewards.append(0.0)\n                reset = True\n\n            if t > learning_starts and t % train_freq == 0:\n                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.\n                if prioritized_replay:\n                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))\n                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience\n                else:\n                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)\n                    weights, batch_idxes = np.ones_like(rewards), None\n                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)\n                if prioritized_replay:\n                    new_priorities = np.abs(td_errors) + prioritized_replay_eps\n                    replay_buffer.update_priorities(batch_idxes, new_priorities)\n\n            if t > learning_starts and t % target_network_update_freq == 0:\n                # Update target network periodically.\n                update_target()\n\n            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)\n            num_episodes = len(episode_rewards)\n            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:\n                logger.record_tabular(\"steps\", t)\n                logger.record_tabular(\"episodes\", num_episodes)\n                logger.record_tabular(\"mean 100 episode reward\", mean_100ep_reward)\n                logger.record_tabular(\"% time spent exploring\", int(100 * exploration.value(t)))\n                logger.dump_tabular()\n\n            if (checkpoint_freq is not None and t > learning_starts and\n                    num_episodes > 100 and t % checkpoint_freq == 0):\n                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:\n                    if print_freq is not None:\n                        logger.log(\"Saving model due to mean reward increase: {} -> {}\".format(\n                                   saved_mean_reward, mean_100ep_reward))\n                    save_variables(model_file)\n                    model_saved = True\n                    saved_mean_reward = mean_100ep_reward\n        if model_saved:\n            if print_freq is not None:\n                logger.log(\"Restored model with mean reward: {}\".format(saved_mean_reward))\n            load_variables(model_file)\n\n    return act"
code_l = ["def", "learn", "(", "env", ",", "network", ",", "seed", "=", "None", ",", "lr", "=", "5e-4", ",",
          "total_timesteps", "=", "100000", ",", "buffer_size", "=", "50000", ",", "exploration_fraction", "=", "0.1",
          ",", "exploration_final_eps", "=", "0.02", ",", "train_freq", "=", "1", ",", "batch_size", "=", "32", ",",
          "print_freq", "=", "100", ",", "checkpoint_freq", "=", "10000", ",", "checkpoint_path", "=", "None", ",",
          "learning_starts", "=", "1000", ",", "gamma", "=", "1.0", ",", "target_network_update_freq", "=", "500", ",",
          "prioritized_replay", "=", "False", ",", "prioritized_replay_alpha", "=", "0.6", ",",
          "prioritized_replay_beta0", "=", "0.4", ",", "prioritized_replay_beta_iters", "=", "None", ",",
          "prioritized_replay_eps", "=", "1e-6", ",", "param_noise", "=", "False", ",", "callback", "=", "None", ",",
          "load_path", "=", "None", ",", "*", "*", "network_kwargs", ")", ":",
          "# Create all the functions necessary to train the model", "sess", "=", "get_session", "(", ")",
          "set_global_seeds", "(", "seed", ")", "q_func", "=", "build_q_func", "(", "network", ",", "*", "*",
          "network_kwargs", ")", "# capture the shape outside the closure so that the env object is not serialized",
          "# by cloudpickle when serializing make_obs_ph", "observation_space", "=", "env", ".", "observation_space",
          "def", "make_obs_ph", "(", "name", ")", ":", "return", "ObservationInput", "(", "observation_space", ",",
          "name", "=", "name", ")", "act", ",", "train", ",", "update_target", ",", "debug", "=", "deepq", ".",
          "build_train", "(", "make_obs_ph", "=", "make_obs_ph", ",", "q_func", "=", "q_func", ",", "num_actions", "=",
          "env", ".", "action_space", ".", "n", ",", "optimizer", "=", "tf", ".", "train", ".", "AdamOptimizer", "(",
          "learning_rate", "=", "lr", ")", ",", "gamma", "=", "gamma", ",", "grad_norm_clipping", "=", "10", ",",
          "param_noise", "=", "param_noise", ")", "act_params", "=", "{", "'make_obs_ph'", ":", "make_obs_ph", ",",
          "'q_func'", ":", "q_func", ",", "'num_actions'", ":", "env", ".", "action_space", ".", "n", ",", "}", "act",
          "=", "ActWrapper", "(", "act", ",", "act_params", ")", "# Create the replay buffer", "if",
          "prioritized_replay", ":", "replay_buffer", "=", "PrioritizedReplayBuffer", "(", "buffer_size", ",", "alpha",
          "=", "prioritized_replay_alpha", ")", "if", "prioritized_replay_beta_iters", "is", "None", ":",
          "prioritized_replay_beta_iters", "=", "total_timesteps", "beta_schedule", "=", "LinearSchedule", "(",
          "prioritized_replay_beta_iters", ",", "initial_p", "=", "prioritized_replay_beta0", ",", "final_p", "=",
          "1.0", ")", "else", ":", "replay_buffer", "=", "ReplayBuffer", "(", "buffer_size", ")", "beta_schedule", "=",
          "None", "# Create the schedule for exploration starting from 1.", "exploration", "=", "LinearSchedule", "(",
          "schedule_timesteps", "=", "int", "(", "exploration_fraction", "*", "total_timesteps", ")", ",", "initial_p",
          "=", "1.0", ",", "final_p", "=", "exploration_final_eps", ")",
          "# Initialize the parameters and copy them to the target network.", "U", ".", "initialize", "(", ")",
          "update_target", "(", ")", "episode_rewards", "=", "[", "0.0", "]", "saved_mean_reward", "=", "None", "obs",
          "=", "env", ".", "reset", "(", ")", "reset", "=", "True", "with", "tempfile", ".", "TemporaryDirectory", "(",
          ")", "as", "td", ":", "td", "=", "checkpoint_path", "or", "td", "model_file", "=", "os", ".", "path", ".",
          "join", "(", "td", ",", "\"model\"", ")", "model_saved", "=", "False", "if", "tf", ".", "train", ".",
          "latest_checkpoint", "(", "td", ")", "is", "not", "None", ":", "load_variables", "(", "model_file", ")",
          "logger", ".", "log", "(", "'Loaded model from {}'", ".", "format", "(", "model_file", ")", ")",
          "model_saved", "=", "True", "elif", "load_path", "is", "not", "None", ":", "load_variables", "(", "load_path",
          ")", "logger", ".", "log", "(", "'Loaded model from {}'", ".", "format", "(", "load_path", ")", ")", "for",
          "t", "in", "range", "(", "total_timesteps", ")", ":", "if", "callback", "is", "not", "None", ":", "if",
          "callback", "(", "locals", "(", ")", ",", "globals", "(", ")", ")", ":", "break",
          "# Take action and update exploration to the newest value", "kwargs", "=", "{", "}", "if", "not",
          "param_noise", ":", "update_eps", "=", "exploration", ".", "value", "(", "t", ")",
          "update_param_noise_threshold", "=", "0.", "else", ":", "update_eps", "=", "0.",
          "# Compute the threshold such that the KL divergence between perturbed and non-perturbed",
          "# policy is comparable to eps-greedy exploration with eps = exploration.value(t).",
          "# See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017",
          "# for detailed explanation.", "update_param_noise_threshold", "=", "-", "np", ".", "log", "(", "1.", "-",
          "exploration", ".", "value", "(", "t", ")", "+", "exploration", ".", "value", "(", "t", ")", "/", "float",
          "(", "env", ".", "action_space", ".", "n", ")", ")", "kwargs", "[", "'reset'", "]", "=", "reset", "kwargs",
          "[", "'update_param_noise_threshold'", "]", "=", "update_param_noise_threshold", "kwargs", "[",
          "'update_param_noise_scale'", "]", "=", "True", "action", "=", "act", "(", "np", ".", "array", "(", "obs",
          ")", "[", "None", "]", ",", "update_eps", "=", "update_eps", ",", "*", "*", "kwargs", ")", "[", "0", "]",
          "env_action", "=", "action", "reset", "=", "False", "new_obs", ",", "rew", ",", "done", ",", "_", "=", "env",
          ".", "step", "(", "env_action", ")", "# Store transition in the replay buffer.", "replay_buffer", ".", "add",
          "(", "obs", ",", "action", ",", "rew", ",", "new_obs", ",", "float", "(", "done", ")", ")", "obs", "=",
          "new_obs", "episode_rewards", "[", "-", "1", "]", "+=", "rew", "if", "done", ":", "obs", "=", "env", ".",
          "reset", "(", ")", "episode_rewards", ".", "append", "(", "0.0", ")", "reset", "=", "True", "if", "t", ">",
          "learning_starts", "and", "t", "%", "train_freq", "==", "0", ":",
          "# Minimize the error in Bellman's equation on a batch sampled from replay buffer.", "if",
          "prioritized_replay", ":", "experience", "=", "replay_buffer", ".", "sample", "(", "batch_size", ",", "beta",
          "=", "beta_schedule", ".", "value", "(", "t", ")", ")", "(", "obses_t", ",", "actions", ",", "rewards", ",",
          "obses_tp1", ",", "dones", ",", "weights", ",", "batch_idxes", ")", "=", "experience", "else", ":", "obses_t",
          ",", "actions", ",", "rewards", ",", "obses_tp1", ",", "dones", "=", "replay_buffer", ".", "sample", "(",
          "batch_size", ")", "weights", ",", "batch_idxes", "=", "np", ".", "ones_like", "(", "rewards", ")", ",",
          "None", "td_errors", "=", "train", "(", "obses_t", ",", "actions", ",", "rewards", ",", "obses_tp1", ",",
          "dones", ",", "weights", ")", "if", "prioritized_replay", ":", "new_priorities", "=", "np", ".", "abs", "(",
          "td_errors", ")", "+", "prioritized_replay_eps", "replay_buffer", ".", "update_priorities", "(",
          "batch_idxes", ",", "new_priorities", ")", "if", "t", ">", "learning_starts", "and", "t", "%",
          "target_network_update_freq", "==", "0", ":", "# Update target network periodically.", "update_target", "(",
          ")", "mean_100ep_reward", "=", "round", "(", "np", ".", "mean", "(", "episode_rewards", "[", "-", "101", ":",
          "-", "1", "]", ")", ",", "1", ")", "num_episodes", "=", "len", "(", "episode_rewards", ")", "if", "done",
          "and", "print_freq", "is", "not", "None", "and", "len", "(", "episode_rewards", ")", "%", "print_freq", "==",
          "0", ":", "logger", ".", "record_tabular", "(", "\"steps\"", ",", "t", ")", "logger", ".", "record_tabular",
          "(", "\"episodes\"", ",", "num_episodes", ")", "logger", ".", "record_tabular", "(",
          "\"mean 100 episode reward\"", ",", "mean_100ep_reward", ")", "logger", ".", "record_tabular", "(",
          "\"% time spent exploring\"", ",", "int", "(", "100", "*", "exploration", ".", "value", "(", "t", ")", ")",
          ")", "logger", ".", "dump_tabular", "(", ")", "if", "(", "checkpoint_freq", "is", "not", "None", "and", "t",
          ">", "learning_starts", "and", "num_episodes", ">", "100", "and", "t", "%", "checkpoint_freq", "==", "0", ")",
          ":", "if", "saved_mean_reward", "is", "None", "or", "mean_100ep_reward", ">", "saved_mean_reward", ":", "if",
          "print_freq", "is", "not", "None", ":", "logger", ".", "log", "(",
          "\"Saving model due to mean reward increase: {} -> {}\"", ".", "format", "(", "saved_mean_reward", ",",
          "mean_100ep_reward", ")", ")", "save_variables", "(", "model_file", ")", "model_saved", "=", "True",
          "saved_mean_reward", "=", "mean_100ep_reward", "if", "model_saved", ":", "if", "print_freq", "is", "not",
          "None", ":", "logger", ".", "log", "(", "\"Restored model with mean reward: {}\"", ".", "format", "(",
          "saved_mean_reward", ")", ")", "load_variables", "(", "model_file", ")", "return", "act"]
temp_code = 'None'
# for token in pygments.lex(temp_code, pygments.lexers.PythonLexer()):
#     print(token)
#
# for token in pygments.lex(code, pygments.lexers.PythonLexer()):
#     print(token)
# for token in code_l:
#     for a in pygments.lex(token, pygments.lexers.PythonLexer()):
#         print(a)
#         break
import torch

# print(torch.ones(1, 1).tril_() == 0)

# a = torch.rand(3, 4, 5, 6)
# b = torch.rand(1, 1, 5, 6, 7)
# # x = torch.matmul(a, b)
# # y = torch.einsum('abcd,xyed->abce', a, b)
# # print(x == y)
# # print(x)
# # print(y)
# i = torch.LongTensor([[0, 1, 3], [1, 3, 5], [0, 6, 7]])
# v = torch.FloatTensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
# print(torch.sparse.FloatTensor(i.t(), v, torch.Size([2, 8, 8, 3])).to_dense())
# print(torch.cuda.memory_allocated())

# a = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
# idx = torch.tensor([[0, 1, 2], [2, 3, 4], [3, 4, 5]])
# b = torch.gather(a, -1, idx)
# print(b)
#
# a = torch.rand(5, 1, 3, 4)
# b = torch.rand(5, 4, 3, 4)
# print(torch.matmul(a, b.transpose(-1, -2)).shape)
# a = torch.tensor([1, 2, 3, 4, 5])
# b = torch.tensor([0, 0, 2, 2, 3])
# c = torch.zeros(4).type_as(a)
# # print(a.type(), b.type(), c.type())
# b = c.index_add_(0, b, a)
# print(b)

a = torch.ones(3, 4)
print(a)
idx = torch.randint(low=0, high=5, size=(3, 4))
print(idx)
c = torch.zeros(3, 5).type_as(a)
d = c.scatter_add_(-1, idx, a)
print(d)
a = torch.nn.Embedding(10, 2)
print(isinstance(a, torch.nn.Linear))
a = torch.nn.Linear(10, 2)
print(isinstance(a, torch.nn.Linear))
embedding = torch.nn.Embedding.from_pretrained(torch.randn(2, 3),freeze=False)
print(embedding.weight)
torch.nn.init.xavier_uniform_(embedding.weight)
print(embedding.weight)
