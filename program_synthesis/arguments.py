import argparse
import time


def get_arg_parser(title, mode):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--model_type', type=str, default='karel-lgrl-ref')
    parser.add_argument('--model_dir', type=str, default='models/%d' % int(time.time()))
    parser.add_argument('--dataset', type=str, default='karel')
    parser.add_argument('--dataset_max_size', type=int, default=0)
    parser.add_argument('--dataset_max_code_length', type=int, default=0)
    parser.add_argument('--dataset_filter_code_length', type=int, default=0)
    parser.add_argument('--dataset_bucket', action='store_true', default=False)
    parser.add_argument('--vocab_min_freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)#128
    parser.add_argument('--load-sync', action='store_true')
    parser.add_argument('--iterative-search', type=str, default=None)
    parser.add_argument('--iterative-search-start-with-beams', action='store_true', help="start with the beams from the original model")
    parser.add_argument('--iterative-search-step-limit', type=int, default=5)
    parser.add_argument('--iterative-search-use-overfit-model', help="k=v,k=v... pairs of data to put into the overfit model")
    parser.add_argument('--num_placeholders', type=int, default=0)  # 100
    parser.add_argument('--use_ref_orig', action='store_true') # False
    parser.add_argument('--model_nickname', type=str, default='')


    parser.add_argument(
        '--pretrained', type=str, help='Use format "entire-model:logdirs/path"')
    parser.add_argument(
        '--pretrained-step', type=int)

    if mode == 'train':
        train_group = parser.add_argument_group('train')
        train_group.add_argument('--save_every_n', type=int, default=1000)
        train_group.add_argument('--keep_every_n', type=int, default=10000)
        train_group.add_argument('--debug_every_n', type=int, default=1000)#20
        train_group.add_argument('--eval_every_n', type=int, default=10000000)#10000000
        train_group.add_argument('--eval_n_steps', type=int, default=50)
        train_group.add_argument('--log_interval', type=int, default=100)#20
        train_group.add_argument('--optimizer', type=str, default='sgd') #adam
        train_group.add_argument('--lr', type=float, default=0.1) #.001
        train_group.add_argument('--lr_decay_steps', type=int, default=100000)
        train_group.add_argument('--lr_decay_rate', type=float, default = 0.5)
        train_group.add_argument('--gradient-clip', type=float, default=1)
        train_group.add_argument('--n_warmup_steps', type=int, default=4000)
        train_group.add_argument('--num_epochs', type=int, default=10)
        train_group.add_argument('--num_units', type=int, default=100)
        train_group.add_argument('--num-att-heads', type=int, default=8)
        train_group.add_argument('--bidirectional', action='store_true', default=False)
        train_group.add_argument('--read-code', dest='read_code', action='store_true', default=False)
        train_group.add_argument('--read-text', dest='read_text', action='store_true', default=True)
        train_group.add_argument('--skip-text', dest='read_text', action='store_false')
        train_group.add_argument('--read-io', dest='read_io', action='store_true', default=False)
        train_group.add_argument('--skip-io', dest='read_io', action='store_false')
        train_group.add_argument('--io-count', type=int, default=3)

        train_group.add_argument('--train-policy-gradient-loss', action='store_true')
        train_group.add_argument('--no-baseline', action='store_true')
        train_group.add_argument('--use-held-out-test-for-rl', action='store_true')

        # REINFORCE.
        train_group.add_argument('--reinforce', action='store_true', default=False)
        train_group.add_argument('--max_rollout_length', type=int, default=3)
        train_group.add_argument('--replay_buffer_size', type=int, default=16384)
        train_group.add_argument('--erase_factor', type=float, default=0.01)
        train_group.add_argument('--num_episodes', type=int, default=150)
        train_group.add_argument('--num_training_steps', type=int, default=50)
        train_group.add_argument('--ppo_steps', type=int, default=3)
        train_group.add_argument('--max_grad_norm', type=int, default=0.5)
        train_group.add_argument('--use_clipped_value_loss', default=False)
        train_group.add_argument('--clip_param', type=float, default=0.2)
        train_group.add_argument('--value_loss_coef', type=float, default=0.01)
        train_group.add_argument('--entropy_coef', type=float, default=0.0001)
        train_group.add_argument('--load_sl_model', type=bool, default=True)


        train_group.add_argument(
            '--reinforce-step', type=int, default=0,
            help='Step after which start to use reinforce')
        train_group.add_argument(
            '--reinforce-beam-size', type=int, default=100,
            help='Size of beam to evalutate when using reinforce'
        )

        # REFINE.
        train_group.add_argument('--refine', action='store_true', default=False)
        train_group.add_argument(
            '--refine-beam', type=int, default=10,
            help='Beam size to use while decoding to generate candidate code for training the refinement model.')
        train_group.add_argument(
            '--refine-samples', type=int, default=100000,
            help='# Number of refinement training samples to keep in the buffer.')
        train_group.add_argument('--refine-min-items', type=int, default=128)
        train_group.add_argument(
            '--refine-frac', type=float, default=0.5,
            help='Fraction of time we should sample refinement data for training.')
        train_group.add_argument(
            '--refine-warmup-steps', type=int, default=1000,
            help='Number of steps we should train before we sample any code to generate the refinement dataset.')
        train_group.add_argument(
            '--refine-sample-frac', type=float, default=0.1,
            help='Fraction of batches for which we should sample code to add to the refinement data for training.')

        train_group.add_argument('--karel-hidden-size', type=int, default=256, help="the hidden size to use in LSTM, etc. Changing this value from the default only works for a vanilla model (--model_type karel-lgrl-ref --karel-trace-enc none --karel-refine-dec edit)")
        train_group.add_argument('--karel-trace-enc', default='none') #lstm
        train_group.add_argument('--karel-code-enc', default='default')
        train_group.add_argument('--karel-refine-dec', default='edit') #default
        train_group.add_argument('--karel-trace-usage', default='memory')
        train_group.add_argument('--karel-code-usage', default='memory')

    elif mode == 'eval':
        eval_group = parser.add_argument_group('eval')
        eval_group.add_argument('--tag', type=str, default='')
        eval_group.add_argument('--example-id', type=int, default=None)
        eval_group.add_argument('--step', type=int, default=None)
        eval_group.add_argument('--refine-iters', type=int, default=1)
        eval_group.add_argument('--eval-train', action='store_true', default=False)
        eval_group.add_argument('--eval-segment', default='val')
        eval_group.add_argument('--hide-example-info', action='store_true', default=False)
        eval_group.add_argument('--report-path')
        eval_group.add_argument('--eval-final', action='store_true')
        eval_group.add_argument('--limit', type=int, default=None)

        eval_group.add_argument('--run-predict', action='store_true', default=False)
        eval_group.add_argument('--predict-path')

        eval_group.add_argument('--evaluate-on-all', action='store_true',
                                help="evaluate on all 6 examples, not just the held out one")

    infer_group = parser.add_argument_group('infer')
    infer_group.add_argument('--max_decoder_length', type=int, default=100)
    infer_group.add_argument('--max_beam_trees', type=int, default=1)#100
    infer_group.add_argument('--max_beam_iter', type=int, default=1000)
    infer_group.add_argument('--use_length_penalty', action='store_true', default=False)
    infer_group.add_argument('--length_penalty_factor', type=float, default=0.7)
    infer_group.add_argument('--max_eval_trials', type=int)
    infer_group.add_argument('--min_prob_threshold', type=float, default=1e-5)
    infer_group.add_argument('--search-bfs', action='store_true', default=True)
    infer_group.add_argument('--karel-file-ref-train', help='json file containing a list of dictionaries with keys '
                                                            'is_correct, passes_given_tests, and output. '
                                                            'You can add a colon followed by keyword args '
                                                            'start= and end=, which determine which segment '
                                                            'of the indices to use. For example, '
                                                            '--karel-file-ref-train x.json:start=0.2,end=0.8 '
                                                            'loads the middle 60% of the indices. '
                                                            'The indices are shuffled deterministically '
                                                            'before loading but the ranges do overlap as '
                                                            'expected')
    infer_group.add_argument('--karel-file-ref-val', help='see help for --karel-ref-file-train')
    infer_group.add_argument('--karel-file-ref-train-balancing', choices=['equal-count', 'none'], default='equal-count',
                             help='How to balance the positive/negative examples. Only applies to classification models')
    infer_group.add_argument('--karel-file-ref-train-all-beams', action='store_true',
                             help='Whether to use all the beams individually to train the model.')
    infer_group.add_argument('--karel-mutate-ref', action='store_true')
    infer_group.add_argument('--karel-mutate-n-dist', default='1,2,3')

    infer_group.add_argument('--karel-gold-replace-train',
                             help="dictionary d where d[guid][i] == (model_generating_program, program)")

    infer_group.add_argument('--ensemble-parameters', nargs='*',
                             help="If provided, the character '#' in the model path"
                                  " will be replaced by each of the values provided."
                                  " e.g., --model_dir 'logdirs/hi#' --ensemble-parameters 1 2 3"
                                  " means to use the models logdirs/hi1, logdirs/hi2, logdirs/hi3")
    infer_group.add_argument('--ensemble-mode', choices=['none', 'dovetail'], default='none')

    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument(
        '--restore-map-to-cpu', action='store_true')

    return parser


def backport_default_args(args):
    """Backport default args."""
    backport = {
        "restore_map_to_cpu": False,
        "keep_every_n": 10000000,
        "read_text": True,
        "read_io": False,
        "io_count": 3,
        "refine": False,
        "read_code": False,
        "optimizer": "adam",
        "dataset_filter_code_length": 0,
        "karel_trace_usage": "memory",
        "karel_code_usage": "memory",
        "karel_refine_dec": "edit",
        "karel_hidden_size": 256,
    }
    for key, value in backport.items():
        if not hasattr(args, key):
            setattr(args, key, value)
