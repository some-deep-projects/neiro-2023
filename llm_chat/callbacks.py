# callbacks.py
import typing as tp


# Is based on https://github.com/huggingface/text-generation-inference/blob/70f485bf9f601b5450b00894f56e20b973d1c2e4/server/text_generation_server/models/model.py#L64
def decode_token(
    tokenizer,
    all_input_ids: tp.List[int],
    prefix_offset: int = 0,
    read_offset: int = 0,
) -> tp.Tuple[str, int, int]:
    """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

    # The prefix text is necessary only to defeat cleanup algorithms in the decode
    # which decide to add a space or not depending on the surrounding ids.
    prefix_text = tokenizer.decode(
        all_input_ids[prefix_offset:read_offset], skip_special_tokens=False
    )
    new_text = tokenizer.decode(
        all_input_ids[prefix_offset:], skip_special_tokens=False
    )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text) :]
        return new_text, read_offset, len(all_input_ids)
    else:
        return "", prefix_offset, read_offset


class OnTokenGenCallback:
    def __init__(self, tokenizer, first_message_prefix: str):
        self.tokenizer = tokenizer
        self.first_message_prefix = first_message_prefix
        self.message_tokens = []
        self.prefix_offset = 0
        self.read_offset = 0

    def on_token_gen(self, token_id, message_ind):
        if message_ind == 0 and len(self.message_tokens) == 0:
            print(self.first_message_prefix, end='')

        self.message_tokens.append(token_id)
        new_text, self.prefix_offset, self.read_offset = decode_token(
            self.tokenizer, self.message_tokens,
            self.prefix_offset, self.read_offset
        )
        print(new_text, end='', flush=True)

    def on_gen_end(self, message_ind):
        self.message_tokens = []
        self.prefix_offset = 0
        self.read_offset = 0


class GetUserInputCallback:
    def get_user_input(self):
        return input(">>> ")
