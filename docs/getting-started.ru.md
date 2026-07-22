# Начало работы

## Установка

```bash
pip install "glia-agents[anthropic]"   # ядро + провайдер Claude
```

Для локальной разработки из клона репозитория:

```bash
git clone https://github.com/DenisDrobyshev/glia
cd glia
pip install -e ".[anthropic,dev]"
```

Пакет называется `glia-agents`; импорт всегда `import glia`.

## Ваш первый агент

```python
import asyncio
from glia import Agent, tool
from glia.providers import ClaudeLLM

@tool
async def add(a: int, b: int) -> int:
    """Сложить два целых числа."""
    return a + b

async def main():
    agent = Agent(ClaudeLLM(), tools=[add], system="Будь точным.")
    result = await agent.run("Сколько будет 21 + 21?")
    print(result.output)

asyncio.run(main())
```

`ClaudeLLM` по умолчанию использует модель `claude-opus-4-8` и берёт учётные
данные так же, как это делает Anthropic SDK (`ANTHROPIC_API_KEY` или профиль из
`ant auth login`).

## Запуск офлайн — без ключа API

В glia встроен детерминированный провайдер `EchoLLM`, который воспроизводит
заранее заданную последовательность «ходов» модели. Он реализует тот же протокол
`LLM`, что и `ClaudeLLM`, поэтому код агента не меняется:

```python
from glia.providers import EchoLLM, call

# Ход 1: модель вызывает инструмент. Ход 2: даёт ответ.
llm = EchoLLM([call("add", {"a": 21, "b": 21}), "Ответ: 42."])
agent = Agent(llm, tools=[add])
result = await agent.run("Сколько будет 21 + 21?")
assert result.output == "Ответ: 42."
```

Именно это делает glia тестируемой: весь цикл — инструменты, сабагенты,
компакция, подтверждения — выполняется в CI без затрат и без «мигания» тестов.

## Наблюдать за каждым шагом

Агент генерирует событие на каждое своё действие. Подпишитесь на поток событий,
чтобы увидеть «стеклянный ящик» в работе:

```python
async for event in agent.run_events("Сколько будет 21 + 21?"):
    print(event.kind)
# run_started → model_call → model_response → tool_called → tool_returned → ... → run_finished
```

## Включить стриминг

```python
agent = Agent(ClaudeLLM(), tools=[add], stream=True)
async for event in agent.run_events("Кратко объясни сложение."):
    if event.kind == "model_delta":
        print(event.text, end="", flush=True)   # токены по мере поступления
```

## Что дальше

- [Руководство](guide.md) описывает каждый примитив с примерами.
- В папке `examples/` репозитория есть семь запускаемых скриптов (все — офлайн).
