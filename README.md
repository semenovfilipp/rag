# Vanilla RAG на корпоративной базе знаний

Практика сравнивает Groq zero-shot, Qwen zero-shot в vLLM и Qwen с контекстом из LangChain. Все три режима отвечают на один набор из 50 вопросов. Эффект retrieval оцениваем по разнице между двумя режимами Qwen.

[Открыть в Colab](https://colab.research.google.com/github/semenovfilipp/rag/blob/feature/vanilla-rag/colab.ipynb)

## Запуск

1. Откройте ноутбук с GPU и нажмите «Выполнить всё». Разрешите доступ к существующему секрету `GROQ_API_KEY`.
2. В первой ячейке выберите одновременно `articles.json`, `questions.json` и `ground_truth.json` с компьютера. Файлы хранятся во временной среде Colab.
3. Ноутбук скачает код и зависимости, поднимет vLLM, проверит три режима на двух вопросах, продолжит до 50 и посчитает метрики.
4. В конце ноутбук скачает `rag-results.zip` с `summary.json`, ответами, идентификаторами найденных статей и оценками. Если браузер задержит скачивание, повторно запустите последнюю ячейку. Прогресс сохраняется до сброса среды Colab.

## Схема эксперимента

- Groq zero-shot использует `openai/gpt-oss-20b`.
- Оба локальных режима используют `Qwen/Qwen2.5-1.5B-Instruct` с температурой 0 и одинаковым лимитом генерации.
- RAG индексирует 10 статей целиком. LangChain хранит документы и их метаданные в `InMemoryVectorStore`. Мультиязычная `intfloat/multilingual-e5-small` кодирует вопрос и статьи с префиксами query/passage. Ретривер возвращает три статьи по косинусному сходству. Тексты статей добавляются в запрос к Qwen.
- Эталоны сопоставляются по `id` и используются для оценки. BLEU выводится на шкале 0–100, ROUGE-1 и ROUGE-L F1 на шкале 0–1. Токенизация поддерживает кириллицу. Судья `openai/gpt-oss-120b` оценивает смысловую корректность по шкале 0–2. Он видит вопрос, эталон и ответ. Оценка передаёт эти три поля в Groq.

## Команды Python

```bash
python scripts/experiment.py --data data --results results validate
python scripts/experiment.py --data data --results results run groq
python scripts/experiment.py --data data --results results run local
python scripts/experiment.py --data data --results results run rag
python scripts/experiment.py --data data --results results evaluate
```

Локальные режимы ожидают сервер vLLM на `127.0.0.1:8000`. Для Groq установите `GROQ_API_KEY` в окружении. У режима `run groq` есть флаг `--groq-model`, у `evaluate` — `--judge-model`.

## Результаты

Таблица и выводы заполняются после полного прогона в Colab. Для разбора сравните ответы Qwen по одному и тому же `id` и проверьте, присутствовала ли нужная статья среди сохранённых `article_ids`. Если поиск пропускает статью из-за формулировки вопроса, следующим экспериментом может быть query rewriting. Если нужные документы регулярно уступают менее релевантным, можно проверить reranker.
