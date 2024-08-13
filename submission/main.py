import itertools
import pathlib
import random
import re
import time

import torch
import transformers
from utils import default, prompt

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def format_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


NF = len(default.first_questions)
KAGGLE_AGENT = pathlib.Path("/kaggle_simulations/agent")

if KAGGLE_AGENT.exists():
    model = "llama-3.1/transformers/8b-instruct/1"
    model_path = KAGGLE_AGENT / "input" / model

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    def call_model(
        messages: list[dict],
        max_new_tokens: int = 16,
        temperature: int = 0.6,
    ) -> dict[dict[str]]:
        outputs = pipeline(
            messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return outputs[0]["generated_text"][-1]["content"]

else:
    import ollama

    model = "llama3.1:8b"

    def call_model(
        messages: list[dict],
        max_new_tokens: int = 16,
        temperature: int = 0.6,
    ) -> dict[dict[str]]:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_predict": max_new_tokens, "temperature": temperature},
        )
        return response["message"]["content"]


class Agent:
    max_new_token: int = 32
    temperature: int = 0.6

    def __init__(
        self,
        sys_prompt: str = "You are an AI agent",
    ) -> None:
        self.sys_prompt = sys_prompt
        self.state = []
        self.default = ["default res"]

    def __call__(self, obs, cfg) -> str:
        try:
            self.config_state(obs)
            return self.result(obs, cfg)
        except:
            return self.default_res()

    def config_state(self, obs):
        raise NotImplementedError

    def update_state(self, role: str, content: str | None) -> None:
        if content is not None and len(content) > 0:
            message = format_message(role, content)
            self.state.append(message)

    def result(self, obs, cfg) -> str:
        response = call_model(self.state, self.max_new_token, self.temperature)
        res = self.check(response)
        return res

    def check(self, response: str) -> str:
        raise NotImplementedError

    def default_res(self) -> str:
        return random.choice(self.default)


class Questioner(Agent):
    max_new_token: int = 128

    def __init__(self) -> None:
        super().__init__(sys_prompt=prompt.questioner)

    def config_state(self, obs):
        self.reset_state(obs)
        self.update_state("user", "Let's play !")
        for q, a, g in itertools.zip_longest(obs.questions, obs.answers, obs.guesses):
            self.update_state("assistant", q)
            self.update_state("user", a)
            self.update_state("assistant", g)
            if g is not None:
                self.update_state("user", "no")

    def reset_state(self, obs) -> None:
        self.state = []
        clues = [
            default.first_questions[i][yn] for i, yn in enumerate(obs.answers[:NF])
        ]
        intial_clues = "a 'thing', " + ", ".join(clues)
        self.update_state("system", self.sys_prompt.format(initial_clues=intial_clues))


class Asker(Questioner):
    temperature: int = 0.6

    def __init__(self) -> None:
        super().__init__()
        self.default = default.questions

    def config_state(self, obs):
        super().config_state(obs)
        i = len(obs.questions) + 1
        if obs.answers[-5:].count("no") == 5:
            ask_prompt = prompt.ask_5no.format(i=i)
        elif obs.answers[-3:].count("no") == 3:
            ask_prompt = prompt.ask_3no.format(i=i)
        else:
            ask_prompt = prompt.ask.format(i=i)
        self.update_state("user", ask_prompt)

    def result(self, obs, cfg) -> str:
        i = len(obs.questions)
        if i < NF:
            res = default.first_questions[i]["question"]
        else:
            res = super().result(obs, cfg)
        return res

    def check(self, response: str) -> str:
        matched = re.match(r".*\?$", response)
        return response if matched else self.default_res()


class Guesser(Questioner):
    temperature: int = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.default = default.guesses

    def config_state(self, obs):
        super().config_state(obs)
        i = len(obs.guesses) + 1
        self.update_state("user", prompt.guess.format(i=i, guesses=obs.guesses))

    def reset_state(self, obs):
        super().reset_state(obs)
        self.temperature = Guesser.temperature

    def result(self, obs, cfg) -> str:
        start_time = time.time()
        while time.time() - start_time < 0.8 * cfg.actTimeout:
            results = super().result(obs, cfg).split("|")
            for res in results:
                res = re.sub(r"^(a|an|the)\s+", "", res.lower().strip())
                if res not in obs.guesses and res.count(" ") < 3:
                    return res
            self.temperature = min(0.2, self.temperature + 0.05)
        return self.default_res()

    def check(self, response: str) -> str:
        guesses = re.findall(
            r"\*{2}([^\*\s].*?[^\*\s])\*{2}", response.replace("\n", "")
        )
        res = "|".join(guesses)
        return res if res else self.default_res()


class Answerer(Agent):
    max_new_token: int = 16
    temperature: int = 0.1

    def __init__(self) -> None:
        super().__init__(sys_prompt=prompt.answerer)
        self.default = ["yes", "no"]

    def config_state(self, obs):
        self.reset_state(obs)
        self.update_state(
            "user",
            prompt.answer.format(
                keyword=obs.keyword, category=obs.category, question=obs.questions[-1]
            ),
        )

    def reset_state(self, obs) -> None:
        self.state = []
        self.update_state("system", self.sys_prompt)

    def check(self, response: str) -> str:
        return "yes" if "yes" in response.lower() else self.default_res()


def agent_fn(obs, cfg):
    match obs.turnType:
        case "ask":
            return Asker()(obs, cfg)
        case "answer":
            return Answerer()(obs, cfg)
        case "guess":
            return Guesser()(obs, cfg)
