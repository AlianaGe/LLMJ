PROMPT_INPUT_SYSTEM: str = (
    "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]"
)

PROMPT_INPUT_WO_SYSTEM: str = "{input}"

RULE = """Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""
pairwise_tie = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

{rule}"""

protocol_mapping = {
    "pairwise_tie": pairwise_tie,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def build_autoj_input(
    prompt, resp1, resp2=None, rule=RULE, protocol="single", sys_msg=None
):
    user_msg = protocol_mapping[protocol].format(
        prompt=prompt, response=resp1, response_another=resp2, rule=rule
    )
    return llama2_wrapper(user_msg, sys_msg)


if __name__ == "__main__":
    t = build_autoj_input("instruction", "resp1", "resp2", "pairwise_tie")
    print(t)
