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

"""Test person prediction model.

See README for running instructions.
"""

import argparse
import torch
import os

import models

import utils
from default_args import get_default_args
parser = argparse.ArgumentParser()
get_default_args(parser)

parser.add_argument('--save_output', default=None,
                    help="The path for saving the test results. Mostly used for visualisation")



def main(args):
    """Run testing."""
    test_data = utils.read_data(args, "test")
    print("total test samples:%s" % test_data.num_examples)

    if args.random_other:
        print("warning, testing mode with 'random_other' will result in "
              "different results every run...")

    model = models.get_model(args, gpuid=args.gpuid)

    # load the graph and variables
    tester = models.Tester(model, args)
    utils.initialize(load=True, load_best=args.load_best,
                     args=args, engine=tester)



    perf = utils.evaluate(test_data, args, tester)

    print("performance:")
    numbers = []
    for k in sorted(perf.keys()):
        print("%s, %s" % (k, perf[k]))
        numbers.append("%s" % perf[k])
    print(" ".join(sorted(perf.keys())))
    print(" ".join(numbers))


if __name__ == "__main__":
    arguments = parser.parse_args()
    arguments.is_train = False
    arguments.is_test = True
    arguments = utils.process_args(arguments)

    main(arguments)
