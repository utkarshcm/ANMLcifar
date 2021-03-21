#used for defining arguments and then using in command lines as req. ex --epochs
import argparse

# manual_seed : Sets the seed for generating random numbers
# manual_seed is used so that we get same results while running each time and there is no different results as end
from torch import manual_seed

#checing for CUDA
from torch.cuda import is_available

#importing the train function in anml.py file
from anml import train

#If the python interpreter is running that module (the source file) as the main program, 
# it sets the special __name__ variable to have a value “__main__”. If this file is being
#  imported from another module, __name__ will be set to the module’s name. Module’s name
#  is available as value to __name__ global variable. 
if __name__ == "__main__":

    # Training settings parameters
    parser = argparse.ArgumentParser(description="ANML training")
    parser.add_argument(
        "--rln", type=int, default=256, help="number of channels to use in the RLN"
    )
    parser.add_argument(
        "--nm", type=int, default=112, help="number of channels to use in the NM"
    )
    parser.add_argument(
        "--mask",
        type=int,
        default=2304,
        help="size of the modulatory mask, needs to match extracted features size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30000,
        help="number of epochs to train (default: 30000)",
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=1e-1,
        help="inner learning rate (default: 1e-1)",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=1e-3,
        help="outer learning rate (default: 1e-3)",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    device = "cpu" if not is_available() else args.device
    manual_seed(args.seed)

    train(
        args.rln,
        args.nm,
        args.mask,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=device,
    )
