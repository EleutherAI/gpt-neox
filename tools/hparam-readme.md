## 1. Start a run
Navigate to `gpt-neox/` and create a run with the desired config like so `python tools/hparam_search_wandb.py`. To do search on larger models, simply change the run command in `hparam_search_wandb.py`.

## 2. Start a sweep
From the project page, open the sweep tab from the sidebar

![image](https://user-images.githubusercontent.com/42869065/125507795-b7f29376-9cbe-40a2-8601-ea717bcd77d8.png)

Click on the `Create Sweep` button and a config will be generated, like so:

![image](https://user-images.githubusercontent.com/42869065/125508038-a2ced583-f55d-4fa2-8c6e-99cc5d893f56.png)

Edit the config to only keep the desired hparams (neox has over 170!). Then click on initialize sweep and an agent will be created. Remember to change `program` to `tools/hparam_search_wandb.py`

## 3. Launch the agent
You'll be greeted by this screen

![image](https://user-images.githubusercontent.com/42869065/125508559-6dd003e2-9d0e-4da4-8ec0-662a06faf770.png)

You can also create multiple agents to run on multiple nodes; just copy the command and run locally.
