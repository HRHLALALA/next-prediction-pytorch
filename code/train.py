# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import wandb

"""Train person prediction model.

See README for running instructions.
"""

import argparse
import math
import sys

import torch
import torch.nn as nn

import os

import models
from tqdm import tqdm
import utils
from torch.profiler import profile, record_function, ProfilerActivity
from default_args import get_default_args
torch.cuda.benchmark =True

parser = argparse.ArgumentParser()
get_default_args(parser)

""""""""""""""""""""""Extra args"""""""""""""""""""""
# Wandb related
parser.add_argument("--message", "-m", default="")
parser.add_argument("--group", default="")
""""""""""""""""""""""""""""""""""""""""""""""""""""""

def main(args):
    """Run training."""
    val_perf = []  # summary of validation performance, and the training loss

    train_data = utils.read_data(args, "train")
    val_data = utils.read_data(args, "val")

    args.train_num_examples = train_data.num_examples

    # construct model under gpu0
    model = models.get_model(args, gpuid=args.gpuid)

    trainer = models.Trainer(model, args)
    tester = models.Tester(model, args)
    saver = utils.Saver(max_to_keep=5)
    bestsaver = utils.Saver(max_to_keep=5)

    save_period = args.save_period  # also the eval period

    utils.initialize(
        load=args.load, load_best=args.load_best, args=args, engine=trainer)

    # the total step (iteration) the model will run
    # total / batchSize  * epoch
    num_steps = int(math.ceil(train_data.num_examples /
                              float(args.batch_size))) * args.num_epochs
    # get_batches is a generator, run on the fly

    print(" batch_size:%s, epoch:%s, %s step every epoch, total step:%s,"
          " eval/save every %s steps" % (args.batch_size,
                                         args.num_epochs,
                                         math.ceil(train_data.num_examples /
                                                   float(args.batch_size)),
                                         num_steps,
                                         args.save_period))

    metric = "ade"  # average displacement error # smaller better
    # remember the best eval acc during training
    best = {metric: 999999, "step": -1}

    finalperf = None
    is_start = True
    loss = -1
    grid_loss = -1
    xyloss = -1
    act_loss = -1
    traj_class_loss = -1
    for batch in tqdm(train_data.get_batches(args.batch_size,
                                             num_steps=num_steps),
                      total=num_steps, ascii=True):

        global_step = trainer.global_step


        # if load from existing model, save if first
        if (global_step % save_period == 0) or \
                (args.load_best and is_start) or \
                (args.load and is_start and (args.ignore_vars is None)):

            tqdm.write("\tsaving model %s..." % global_step)
            saver.save(trainer, args.save_dir_model, global_step=global_step)
            tqdm.write("\tdone")

            evalperf = utils.evaluate(val_data, args, tester)

            tqdm.write(("\tlast loss:%.5f, xyloss:%.5f, traj_class_loss:%.5f,"
                        " grid_loss:%s, act_loss:%.5f, eval on validation:%s,"
                        " (best %s:%s at step %s) ") % (
                           loss, xyloss, traj_class_loss, grid_loss, act_loss,
                           ["%s: %s" % (k, evalperf[k])
                            for k in sorted(evalperf.keys())], metric,
                           best[metric], best["step"]))
            wandb.log({
                "EVAL/" + k: v for k, v in evalperf.items()
            })
            # remember the best acc
            if evalperf[metric] < best[metric]:
                best[metric] = evalperf[metric]
                best["step"] = global_step
                # save the best model
                tqdm.write("\t saving best model...")
                bestsaver.save(trainer, args.save_dir_best_model,
                               global_step=global_step)
                tqdm.write("\t done.")

                finalperf = evalperf
                val_perf.append((loss, evalperf))
                wandb.summary["best " + metric] = best[metric]
                wandb.summary["best_step"] = best['step']
            is_start = False
        # with profile( with_stack=True, profile_memory=True) as prof:
        #     loss, xyloss, act_loss, traj_class_loss, grid_loss = \
        #         trainer.step(batch)
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        loss, xyloss, act_loss, traj_class_loss, grid_loss = \
                    trainer.step(batch)
        wandb.log({
            "loss": loss,
            "xyloss": xyloss,
            "traj_class_loss": traj_class_loss,
            **{"grid_loss/" + str(i): grid_loss[i] for i in range(len(grid_loss))}
        }, step=global_step)

        if math.isnan(loss):
            print("nan loss.")
            print(grid_loss)
            sys.exit()

    if global_step % save_period != 0:
        saver.save(trainer, args.save_dir_model, global_step=global_step)
        wandb.save(args.save_dir_model)
        wandb.save(args.save_dir_best_model)

    print("best eval on val %s: %s at %s step, final step %s %s is %s" % (
        metric, best[metric], best["step"], global_step, metric,
        finalperf[metric]))


if __name__ == "__main__":
    arguments = parser.parse_args()
    arguments.is_train = True
    arguments.is_test = False
    arguments.save_output = None
    arguments = utils.process_args(arguments)
    run = wandb.init(
        project="next-prediction",
        config=arguments,
        name=arguments.modelname,
        notes=arguments.message,
        group=arguments.group,
    )
    wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))
    main(arguments)
