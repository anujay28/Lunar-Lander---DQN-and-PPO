def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--bs', type=int, default=32,help = 'batch size')
    parser.add_argument('--max_episodes', type=int, default=10000,help = 'number of episode to terminate')
    parser.add_argument('--gamma', type=float, default=0.99,help='gamma')
    parser.add_argument('--eps', type=float, default=1.0,help ='epsilon')
    parser.add_argument('--eps_decay_window', type=int, default=1000000,help ='epsilon decaying windows (denominator)')
    parser.add_argument('--eps_min', type=float, default=0.1,help = 'minimum amount of epsilon')
    parser.add_argument('--window', type=int, default=100,help='window size')
    parser.add_argument('--mem_capacity', type=int, default=1000000,help = 'memory size')
    parser.add_argument('--mem_init_size', type=int, default=50000,help='memory initial size')
    parser.add_argument('--sync_period', type=int, default=10000,help = 'Syncing Time (T)')
    parser.add_argument('--learn_freq', type=int, default=4,help='frequency of learning')
    parser.add_argument('--save_freq', type=int, default=100,help='frequency of saving model')
    return parser
