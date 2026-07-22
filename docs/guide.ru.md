# Руководство

Всё, что умеет glia, с примерами кода. Каждая возможность по умолчанию выключена
и отражается в потоке событий.

## Основные понятия

Всю модель несут три объекта:

- **`Agent`** — модель, её инструменты и цикл, который их связывает.
- **`Trajectory`** — полное, сериализуемое в JSON состояние запуска: системный
  промпт, список сообщений, лог событий (только на добавление) и расход токенов.
- **`Event`** — запись об одном действии агента (`ModelCall`, `ModelResponse`,
  `ToolCalled`, `ToolReturned`, `ApprovalResolved`, …).

`agent.run(prompt)` выполняет цикл до конца и возвращает `RunResult`.
`agent.run_events(prompt)` — тот же цикл в виде асинхронного потока событий:
подпишитесь, чтобы наблюдать, или прервите, чтобы вмешаться.

```python
result = await agent.run("привет")
result.output        # финальный текст модели
result.stop_reason   # почему остановились
result.steps         # сколько ходов модели
result.usage          # суммарный расход токенов
result.trajectory     # полная запись, доступная для инспекции
```

## Инструменты

Инструмент — обычная типизированная Python-функция. Декоратор `@tool` читает
её аннотации типов и докстринг и строит JSON-схему — без DSL для схем и без
базового класса.

```python
from typing import Annotated, Literal
from glia import tool

@tool
async def search(
    query: Annotated[str, "Что искать"],
    limit: int = 5,
    sort: Literal["relevance", "date"] = "relevance",
) -> str:
    """Поиск по базе знаний."""
    ...
```

Работают и синхронные, и асинхронные функции. Инструмент, бросивший исключение,
превращается в результат-ошибку, который модель видит и может обработать — цикл
при этом не падает.

## Провайдеры

glia общается с моделями через один небольшой протокол `LLM` с единственным
методом `async generate(request) -> LLMResponse`. Поставляются два адаптера:

- **`ClaudeLLM`** — Claude через Anthropic SDK (опциональный extra `[anthropic]`).
- **`EchoLLM`** — детерминированный, офлайн, для тестов и демо.

Свой адаптер — около 40 строк. См. [Архитектуру](ARCHITECTURE.md).

## Стриминг

Установите `stream=True`. Если провайдер это поддерживает, дельты текста
переизлучаются как события `ModelDelta`; буферизованный `ModelResponse` всё
равно приходит следом, поэтому дальше по коду ничего не меняется. Провайдеры без
стриминга автоматически откатываются к `generate`.

```python
agent = Agent(ClaudeLLM(), stream=True)
async for event in agent.run_events("Напиши хайку."):
    if event.kind == "model_delta":
        print(event.text, end="", flush=True)
```

## Параллельный запуск инструментов

Когда модель запрашивает несколько инструментов за один ход, glia выполняет их
параллельно (`parallel_tools=True`, по умолчанию), сохраняя события
`ToolCalled` / `ToolReturned` и результаты в исходном порядке вызовов.

```python
agent = Agent(llm, tools=[search, fetch], parallel_tools=True)
```

## Подтверждение с участием человека (human-in-the-loop)

Поставьте любой инструмент за проверяемый «шлюз». Политика подтверждения — это
любой вызываемый объект `(ApprovalRequest) -> Decision` (синхронный или
асинхронный; можно вернуть и просто `bool`). Отклонённые вызовы не выполняются и
возвращают результат-ошибку, на который модель может отреагировать.

```python
from glia import Agent
from glia.approval import deny, allow_only, prompt_in_terminal

# Заблокировать конкретный инструмент:
agent = Agent(llm, tools=[search, delete_all], approval=deny("delete_all"))

# Или разрешить только безопасный набор:
agent = Agent(llm, tools=[...], approval=allow_only("search", "read"))

# Или спросить человека в терминале (эталонная политика):
agent = Agent(llm, tools=[...], approval=prompt_in_terminal)

# Или своя логика:
def policy(request):
    return request.name != "delete_all"   # разрешить всё, кроме delete_all
```

Каждое решение порождает события `ApprovalRequested` → `ApprovalResolved`.

## Структурированный вывод

Получите типизированный объект вместо строки, которую надо парсить. glia
заставляет модель вызвать один инструмент, схема которого *и есть* нужная вам
форма, и читает провалидированные аргументы — не зависит от провайдера, работает
даже с `EchoLLM`.

```python
from dataclasses import dataclass
from glia import generate_structured

@dataclass
class Contact:
    name: str
    email: str
    wants_demo: bool

contact = await generate_structured(
    llm, "Извлеки: Ада (ada@x.io) попросила демо.", Contact
)
# -> Contact(name='Ада', email='ada@x.io', wants_demo=True)
```

В качестве `schema` можно передать `dict` с JSON-схемой (вернётся `dict`), тип
`dataclass` или модель Pydantic (вернётся экземпляр). Pydantic импортируется
только если вы им пользуетесь.

## Инженерия контекста (компакция)

Ограниченное окно контекста — самый дефицитный ресурс долгоживущего агента.
`Compactor` решает, когда траектория стала слишком большой и как её сжать; агент
вызывает его один раз за шаг и порождает событие `Compacted`.

```python
from glia import Agent, SummarizingCompactor, TrimmingCompactor

# Свернуть старые ходы в резюме, написанное моделью (сохраняя происхождение):
agent = Agent(llm, compactor=SummarizingCompactor(max_messages=40, keep_last=12))

# Или просто отбросить самые старые ходы (дёшево, без вызова модели):
agent = Agent(llm, compactor=TrimmingCompactor(max_messages=40, keep_last=20))
```

Оба сохраняют последние ходы дословно и никогда не разрывают вызов инструмента и
его результаты.

## Устойчивое выполнение (чек-пойнт и возобновление)

Запуск — это JSON-документ. Сохраните его и продолжите позже — хоть в новом
процессе.

```python
from glia import Agent, Trajectory
from glia.checkpoint import checkpointer, save, load

traj = Trajectory.new()
agent = Agent(llm, hooks=[checkpointer(traj, "run.json")])   # снапшот на каждом шаге
await agent.run("начать задачу", trajectory=traj)

resumed = load("run.json")
await agent.run("продолжить", trajectory=resumed)            # продолжит с места остановки
```

## Guardrails

Guardrail — любой вызываемый объект `(text) -> None`, который бросает
`GuardrailTripped` при отклонении. Входные guardrails применяются к каждому
промпту, выходные — к финальному ответу.

```python
from glia import Agent
from glia.guardrails import max_length, block_pattern, no_secrets

agent = Agent(
    llm,
    input_guardrails=[max_length(4000)],
    output_guardrails=[block_pattern("confidential"), no_secrets()],
)
```

## Сабагенты

Любой агент можно выставить как инструмент, который вызывает другой агент — это и
есть «сабагенты» в glia целиком.

```python
researcher = Agent(llm, tools=[web_search], name="researcher")
lead = Agent(llm, tools=[researcher.as_tool("research", "Искать в интернете")])
```

## Эвалы как тесты

Относитесь к эвалам как к юнит-тестам: набор промптов плюс проверки в стиле
pytest, запускаемые в CI и блокирующие деплой.

```python
from glia.evals import Case, evaluate, contains, used_tool, did_not_error

suite = [
    Case("отвечает", "сколько 2+2?", [contains("4"), did_not_error]),
    Case("вызывает инструмент", "сколько 2+2?", [used_tool("add")]),
]
report = await evaluate(suite, lambda: Agent(llm, tools=[add]))
assert report.ok, report
```

## Наблюдаемость через hooks

Hook — любой вызываемый объект, получающий каждое `Event` (синхронный или
асинхронный). Через него проходят все события — логируйте, трассируйте или
управляйте UI.

```python
def log(event):
    print(event.kind, event.to_dict())

agent = Agent(llm, hooks=[log])
```
