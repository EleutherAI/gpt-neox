from gpt_neox.utils import get_args, get_params
from gpt_neox import prepare_data

args = get_args()
if args.model == "enwik8":
    prepare_data("enwik8")
else:
    params = get_params(args.model)
    # prepare data
    dset_params = params["dataset"]
    assert dset_params is not None
    prepare_data(dset_params["name"])
