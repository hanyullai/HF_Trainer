def chatglm2_template(query, history, system):
    if history is None:
        history = []

    prompt = ""
    
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)

    return prompt

def chatglm3_base_template(query, history, system):
    prompt = "Q: {}\n\nA: ".format(query)

    return prompt

def dummy_template(query, history, system):
    return query

build_prompt = {
    'dummy': dummy_template,
    'chatglm2': chatglm2_template,
    'chatglm3-base': chatglm3_base_template
}