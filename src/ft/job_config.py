from torchtitan.config_manager import JobConfig as TitanConfig


def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


class JobConfig(TitanConfig):
    def __init__(self):
        super().__init__()
        self.parser.add_argument(
            "--training.padding", type=bool, default=False, help="Whether to use padding dataloader."
        )
        remove_argument(self.parser, "--model.flavor")
        remove_argument(self.parser, "--model.tokenizer_path")

    def _validate_config(self) -> None:
        return
