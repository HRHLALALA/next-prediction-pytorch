

def get_default_args(parser):
    # inputs and outputs
    parser.add_argument("prepropath", type=str)
    parser.add_argument("outbasepath", type=str,
                        help="full path will be outbasepath/modelname/runId")
    parser.add_argument("modelname", type=str)
    parser.add_argument("--runId", type=int, default=0,
                        help="used for run the same model multiple times")

    # ---- gpu stuff. Now only one gpu is used
    parser.add_argument("--gpuid", default=0, type=int)

    parser.add_argument("--load", action="store_true",
                        default=False, help="whether to load existing model")
    parser.add_argument("--load_best", action="store_true",
                        default=False, help="whether to load the best model")
    # use for pre-trained model
    parser.add_argument("--load_from", type=str, default=None)

    # ------------- experiment settings
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--is_actev", action="store_true",
                        help="is actev/virat dataset, has activity info")

    # ------------------- basic model parameters
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--enc_hidden_size", type=int,
                        default=256, help="hidden size for rnn")
    parser.add_argument("--dec_hidden_size", type=int,
                        default=256, help="hidden size for rnn")
    parser.add_argument("--activation_func", type=str,
                        default="tanh", help="relu/lrelu/tanh")

    # ---- multi decoder
    parser.add_argument("--multi_decoder", action="store_true")

    # ----------- add person appearance features
    parser.add_argument("--person_feat_path", type=str, default=None)
    parser.add_argument("--person_feat_dim", type=int, default=256)
    parser.add_argument("--person_h", type=int, default=9,
                        help="roi align resize to feature size")
    parser.add_argument("--person_w", type=int, default=5,
                        help="roi align resize to feature size")

    # ---------------- other boxes
    parser.add_argument("--random_other", action="store_true",
                        help="randomize top k other boxes")
    parser.add_argument("--max_other", type=int, default=15,
                        help="maximum number of other box")
    parser.add_argument("--box_emb_size", type=int, default=64)

    # ---------- person pose features
    parser.add_argument("--add_kp", action="store_true")
    parser.add_argument("--kp_size", default=17, type=int)

    # --------- scene features
    parser.add_argument("--scene_conv_kernel", default=3, type=int)
    parser.add_argument("--scene_h", default=36, type=int)
    parser.add_argument("--scene_w", default=64, type=int)
    parser.add_argument("--scene_class", default=11, type=int)
    parser.add_argument("--scene_conv_dim", default=64, type=int)
    parser.add_argument("--pool_scale_idx", default=0, type=int)

    #  --------- activity
    parser.add_argument("--add_activity", action="store_true")

    #  --------- loss weight
    parser.add_argument("--act_loss_weight", default=1.0, type=float)
    parser.add_argument("--grid_loss_weight", default=0.1, type=float)
    parser.add_argument("--traj_class_loss_weight", default=1.0, type=float)

    # ---------------------------- training hparam
    parser.add_argument("--save_period", type=int, default=300,
                        help="num steps to save model and eval")
    parser.add_argument("--batch_size", type=int, default=64)
    # num_step will be num_example/batch_size * epoch
    parser.add_argument("--num_epochs", type=int, default=100)
    # drop out rate
    parser.add_argument("--keep_prob", default=0.7, type=float,
                        help="1.0 - drop out rate")
    # l2 weight decay rate
    parser.add_argument("--wd", default=0.0001, type=float,
                        help="l2 weight decay loss")
    parser.add_argument("--clip_gradient_norm", default=10, type=float,
                        help="gradient clipping")
    parser.add_argument("--optimizer", default="adadelta",
                        help="momentum|adadelta|adam")
    parser.add_argument("--learning_rate_decay", default=0.95,
                        type=float, help="learning rate decay")
    parser.add_argument("--num_epoch_per_decay", default=2.0,
                        type=float, help="how epoch after which lr decay")
    parser.add_argument("--init_lr", default=0.2, type=float,
                        help="Start learning rate")
    parser.add_argument("--emb_lr", type=float, default=1.0,
                        help="learning scaling factor for emb variables")

    parser.add_argument("--preload_features", action="store_true")
    parser.add_argument("--embed_traj_label", action="store_true")