import typing as tp

import torch
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_obj
from exceptions import LengthException
from select_token import TokenSelection
from stop_criteria import StopCriterion, GenState
from callbacks import OnTokenGenCallback, GetUserInputCallback
from stop_criteria import WordStopCriterion


@torch.no_grad()
def generate(model: tp.Any, past_key_values: tp.Optional[Tensor], past_input_len: int, input_ids: Tensor,
             stop_func: StopCriterion, max_gen_len: int, max_model_len: int, select_token_func: TokenSelection):
    for gen_ids_len in range(max_gen_len):
        if past_input_len + input_ids.size(1) > max_model_len:
            raise LengthException()
        model_out = model(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=True,
        )
        past_key_values = model_out.past_key_values
        past_input_len += input_ids.size(1)
        input_ids = select_token_func(model_out.logits[0, -1]).unsqueeze(0)
        gen_id_int = input_ids.item()

        stop_func.on_gen(gen_id_int)
        gen_state = stop_func.get_state()
        yield gen_id_int, past_key_values, past_input_len, gen_state

        if gen_state is GenState.FULL_STOP:
            break


def process_dialog(model, tokenizer, prompt, stop_criterion, max_gen_len, max_model_len, token_selection,
                   user_start_message, bot_start_message, token_gen_callback, get_input_callback):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    past_key_values = None
    past_input_len = 0
    message_ind = 0

    while True:
        tokens_generator = generate(
            model=model,
            past_key_values=past_key_values,
            past_input_len=past_input_len,
            input_ids=input_ids,
            stop_func=stop_criterion,
            max_gen_len=max_gen_len,
            max_model_len=max_model_len,
            select_token_func=token_selection
        )
        for gen_token, past_key_values, past_input_len, gen_state in tokens_generator:
            token_gen_callback.on_token_gen(gen_token, message_ind)
        token_gen_callback.on_gen_end(message_ind)
        message_ind += 1

        model_input_str = user_start_message + get_input_callback.get_user_input() + bot_start_message
        input_ids = tokenizer(model_input_str, return_tensors="pt").input_ids


def load_prompt(cfg: DictConfig) -> str:
    with open(cfg.prompt.path) as f:
        prompt = f.read()
    prompt = prompt.format(bot_name=cfg.names.bot_name, user_name=cfg.names.user_name)
    return prompt


def load_config(cfg_path: str) -> DictConfig:
    return OmegaConf.load(cfg_path)


def chat(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path, torch_dtype=torch.float16, **cfg.model.load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)

    stop_criterion = WordStopCriterion(tokenizer=tokenizer, stop_words=["\n", " \n", "You:", " You:", "<|endoftext|>"])
    token_selection = load_obj(cfg.token_selection.path)(**cfg.token_selection.kwargs)
    token_gen_callback = OnTokenGenCallback(
        tokenizer=tokenizer, first_message_prefix=cfg.messages.first_message_prefix
    )
    get_input_callback = GetUserInputCallback()
    prompt = load_prompt(cfg)
    print(cfg.messages.dialog_start)

    try:
        process_dialog(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            stop_criterion=stop_criterion,
            max_gen_len=cfg.generation.max_gen_len,
            max_model_len=cfg.generation.max_model_len,
            token_selection=token_selection,
            user_start_message=cfg.messages.user_start_message,
            bot_start_message=cfg.messages.bot_start_message,
            token_gen_callback=token_gen_callback,
            get_input_callback=get_input_callback,
        )
    except LengthException:
        print(cfg.exceptions.length)
    except KeyboardInterrupt:
        print(cfg.exceptions.keyboard)


if __name__ == '__main__':
    chat(load_config("conf/chat.yaml"))
