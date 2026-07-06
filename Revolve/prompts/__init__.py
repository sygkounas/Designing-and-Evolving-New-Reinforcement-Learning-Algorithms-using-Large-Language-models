system_prompt = open("prompts/system_prompt.txt", "r").read()
mutation_macro_auto_prompt = open("prompts/mutation_macro_auto.txt", "r").read()
mutation_micro_auto_prompt = open("prompts/mutation_micro_auto.txt", "r").read()
crossover_auto_prompt = open("prompts/crossover_auto.txt", "r").read()
env_prompt= open("prompts/env.txt", "r").read()


types = {
    "system_prompt": system_prompt,
    "mutation_macro_auto": mutation_macro_auto_prompt,
    "mutation_micro_auto": mutation_micro_auto_prompt,
    "crossover_auto": crossover_auto_prompt,
    "env_prompt": env_prompt
}

# print("system_prompt",system_prompt)
# print("env_input_prompt",env_input_prompt)
# print("mutation_auto_prompt",mutation_auto_prompt)
# print("crossover_auto_prompt",crossover_auto_prompt)
# print("mutation_prompt",mutation_prompt)
# print("crossover_prompt",crossover_prompt)
