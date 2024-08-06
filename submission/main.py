import itertools
import pathlib
import re
import time

import torch
import transformers
from utils import default, prompt

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def format_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


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
        turn_types: list[str] = [],
        sys_prompt: str = "You are an AI agent",
    ) -> None:
        self.turn_types = turn_types
        self.sys_prompt = sys_prompt
        self.state = []

    def __call__(self, obs, cfg) -> str:
        if obs.turnType in self.turn_types:
            self.config_state(obs)
            return self.result(obs, cfg)
        else:
            raise KeyError

    def result(self, obs, cfg) -> str:
        response = call_model(self.state, self.max_new_token, self.temperature)
        res = self.check(response)
        return res

    def config_state(self, obs):
        raise NotImplementedError

    def check(self, response: str) -> str:
        raise NotImplementedError

    def reset_state(self) -> None:
        self.state = []
        self.update_state("system", self.sys_prompt)

    def update_state(self, role: str, content: str | None) -> None:
        if content is not None and len(content) > 0:
            message = format_message(role, content)
            self.state.append(message)


class Questioner(Agent):
    max_new_token: int = 128
    temperature: int = 0.6

    def __init__(self, turn_types: str = ["ask", "guess"]) -> None:
        super().__init__(turn_types=turn_types, sys_prompt=prompt.questioner)

    def config_state(self, obs):
        self.reset_state()
        self.update_state("user", "Let's play ! Ask the first question (1/20).")
        for q, a in itertools.zip_longest(obs.questions, obs.answers):
            self.update_state("assistant", q)
            self.update_state("user", a)


class Asker(Questioner):
    def __init__(self) -> None:
        super().__init__(turn_types=["ask"])
        self.questions = default.questions

    def next_question(self) -> str:
        return self.questions.pop(0)

    def result(self, obs, cfg) -> str:
        if len(obs.questions) == 0:
            res = self.next_question()
        else:
            res = super().result(obs, cfg)
        return res

    def config_state(self, obs):
        super().config_state(obs)
        i = len(obs.questions) + 1
        self.update_state("user", prompt.ask.format(i=i))

    def check(self, response: str) -> str:
        matched = re.match(r".*\?$", response)
        return response if matched else self.next_question()


class Guesser(Questioner):
    def __init__(self) -> None:
        super().__init__(turn_types=["guess"])
        self.guesses = default.guesses

    def next_guess(self) -> str:
        return self.guesses.pop(0)

    def result(self, obs, cfg) -> str:
        start_time = time.time()
        while time.time() - start_time < 0.8 * cfg.actTimeout:
            res = super().result(obs, cfg).lower()
            if res not in obs.guesses:
                return res
        return self.next_guess()

    def config_state(self, obs):
        super().config_state(obs)
        i = len(obs.guesses) + 1
        guesses = ",".join([f"**{g}**" for g in obs.guesses])
        self.update_state("user", prompt.guess.format(i=i, guesses=guesses))

    def check(self, response: str) -> str:
        res = re.findall(r"\*{2,}(.*?)\*{2,}", response.replace("\n", ""))
        return res[-1] if res else self.next_guess()


class Answerer(Agent):
    max_new_token: int = 16
    temperature: int = 0.2

    def __init__(self) -> None:
        super().__init__(turn_types=["answer"], sys_prompt=prompt.answerer)

    def config_state(self, obs):
        self.reset_state()
        self.update_state(
            "user",
            prompt.answer.format(
                keyword=obs.keyword, category=obs.category, question=obs.questions[-1]
            ),
        )

    def check(self, response: str) -> str:
        return "yes" if "yes" in response.lower() else "no"


def agent_fn(obs, cfg):

    match obs.turnType:
        case "ask":
            return Asker()(obs, cfg)
        case "answer":
            return Answerer()(obs, cfg)
        case "guess":
            return Guesser()(obs, cfg)
