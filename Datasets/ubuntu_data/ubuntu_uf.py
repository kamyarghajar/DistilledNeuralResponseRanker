
from functools import partial
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer

MAX_RESPONSE_LEN = 60
MAX_CONTEXT_LEN = 200
MAX_CONTEXT_SEQ_LEN = 10

def truncate(x, length):
    if len(x) <= length:
        return x
    else:
        x = x[0 + max(0, len(x) - length):]
        return x


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

tokenizer.add_special_tokens({
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES[-1]: ['[EOU]', '[NUM]', '[PATH]', '[URL]'],
})

def process(length, x):
    temp = tokenizer.convert_tokens_to_ids(['[CLS]'] + truncate(tokenizer.tokenize(x), length) + ['[SEP]'])
    return temp + [0] * max(0, length - len(temp) + 2)

process_context = partial(process, MAX_CONTEXT_LEN)
process_response = partial(process, MAX_RESPONSE_LEN)

def process_utterances(x):
    utterances = [process_response(i) for i in x][:MAX_CONTEXT_SEQ_LEN]
    utterances = utterances + ((MAX_CONTEXT_SEQ_LEN - len(utterances)) * [[0] * (MAX_RESPONSE_LEN + 2)])
    return utterances