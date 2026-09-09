# AWS FAQ Chatbot — NLP + Amazon Lex

Intent classification and semantic FAQ retrieval over the Bitext customer-support
corpus, served through Amazon Lex V2 and a Lambda code hook.

```
User → Lex V2 → Lambda code hook → intent classifier → semantic retrieval → answer
                       ↓                                       ↓
              DynamoDB (interactions)                 fallback detection
                       ↓
              CloudWatch EMF metrics → alarms → analytics
```

## Quick start

```bash
pip install -r requirements.txt

# Kaggle → data/bitext.csv (columns: flags, instruction, category, intent, response)
python -m src.train    --data data/bitext.csv --backend lsa --out artifacts
python -m src.evaluate --artifacts artifacts --out reports --ood data/ood_questions.txt
python -m src.cli      --artifacts artifacts
```

No dataset yet? `python scripts/make_sample_data.py data/sample_bitext.csv` writes a
1,120-row, 8-intent stand-in with the same schema so the pipeline runs end to end.

> The sample data is generated from disjoint templates, so it scores 1.0 on every
> metric. **Those are not real numbers** — they only prove the wiring works. Expect
> roughly 0.85–0.95 macro-F1 on the real 27-intent Bitext corpus, where intents
> like `change_order` / `cancel_order` and `track_order` / `track_refund` genuinely
> overlap.

## Design decisions worth defending in a write-up

**Two-stage answering.** Classify the intent, then retrieve *within* it. Bitext has
~1,000 paraphrases per intent sharing a handful of responses, so a flat semantic
search over all FAQ entries confuses intents with shared vocabulary. Restricting by
predicted intent lifts top-1 accuracy — but only when the classifier is confident.
Below a confidence floor or a thin top-2 margin, `pipeline.py` drops the filter and
searches globally rather than trusting a shaky label.

**Fallback is not a similarity threshold.** This is the part most implementations get
wrong. With a low-rank index, cosine similarity is *not* a reliable out-of-domain
signal: unknown text gets projected onto whichever direction is nearest and can score
0.85+. In testing, "what is the airspeed velocity of an unladen swallow" scored 0.858
against an account-creation FAQ. `retriever.fallback_reason()` therefore uses three
independent gates — lexical vocabulary coverage, absolute similarity, and intent
confidence — and returns *which* gate fired. That matters operationally:
`out_of_vocabulary` means a content gap in the KB, `ambiguous_intent` means two
intents need better training utterances. Different tickets, different owners.

**Calibrated SVM, not raw decision scores.** `CalibratedClassifierCV` wraps LinearSVC
because an uncalibrated margin isn't comparable across intents, and the fallback logic
needs a probability it can threshold.

**Pluggable embeddings.** `lsa` (TF-IDF + SVD) ships in a Lambda layer at a few MB and
runs in ~3 ms. `sbert` gives better paraphrase handling offline. `bedrock` (Titan v2)
removes the artifact entirely at the cost of a network hop. Same interface, so
`evaluate.py` can benchmark all three and you can report the trade-off.

**Lex utterances are sampled for diversity, not randomly.** `lex/build_lex_bot.py`
greedily picks maximally dissimilar paraphrases, because Lex trains on those ~15
utterances and near-duplicates waste the budget.

## Evaluation output (`reports/`)

| File | Contents |
|---|---|
| `metrics.json` | accuracy, macro/weighted P/R/F1, retrieval top-1/top-3/MRR, fallback P/R/F1, latency mean + p95 |
| `intent_classification_report.csv` | per-intent precision, recall, F1, support |
| `confusion_matrix.csv` / `.png` | row-normalised, for spotting which intent pairs actually collide |
| `threshold_sweep.csv` | coverage vs. accuracy-on-answered — how you *pick* the threshold instead of guessing |
| `error_analysis.csv` | answered-but-wrong cases, the ones that hurt users |
| `predictions.csv` | per-query record feeding `scripts/analytics.py` |

Fallback quality is scored as a binary task with "should refuse" as the positive
class, against `data/ood_questions.txt`. Report both recall and false-alarm rate — a
bot that refuses everything gets perfect recall and is useless.

## Analytics

```bash
python scripts/analytics.py --source reports/predictions.csv           # offline
python scripts/analytics.py --source dynamodb --table faq-chatbot-interactions --days 7
```

Covers all six tracked metrics: top FAQs, intent distribution, failed/unknown queries
(broken down by fallback reason), response accuracy, latency percentiles, and
satisfaction rate. Satisfaction requires the client to write a `helpful` flag back to
the interaction row — that thumbs-up/down plumbing is the one piece you have to add
outside this repo.

## Deployment

```bash
# 1. Model artifacts as a Lambda layer
mkdir -p layer/artifacts && cp artifacts/*.joblib artifacts/manifest.json layer/artifacts/
cd layer && zip -r ../faq-artifacts-layer.zip . && cd ..
aws lambda publish-layer-version --layer-name faq-artifacts \
  --zip-file fileb://faq-artifacts-layer.zip --compatible-runtimes python3.12

# 2. Function code (src/ + lambda_function.py, with scikit-learn from a public layer
#    or your own build — do not pip install sklearn into the function zip)
zip -r function.zip src lambda/lambda_function.py

# 3. Lex bot
python lex/build_lex_bot.py --data data/bitext.csv --out lex/bot_definition.json
# then create the bot/locale/intents via `aws lexv2-models` or the console import,
# attach the Lambda as the fulfillment code hook, and build the locale.

# 4. Storage + telemetry
aws dynamodb create-table --cli-input-json file://infra/dynamodb_table.json
# alarms: infra/cloudwatch_alarms.json
```

Set `INTERACTIONS_TABLE`, `METRIC_NAMESPACE`, and either `ARTIFACT_DIR=/opt/artifacts`
(layer) or `ARTIFACT_BUCKET` + `ARTIFACT_PREFIX` (S3, for artifacts over the 250 MB
layer limit). The Lambda role needs `dynamodb:PutItem`, `logs:*`, and
`bedrock:InvokeModel` only if you use the Bedrock backend.

Set Lex's `nluIntentConfidenceThreshold` to 0.40 so ambiguous utterances land in
`AMAZON.FallbackIntent` and reach your semantic retrieval, which is better at
paraphrases than Lex's own NLU.

## Layout

```
src/preprocess.py    Bitext loader, placeholder handling, normalisation, FAQ corpus
src/intent_model.py  TF-IDF (word+char) → calibrated LinearSVC
src/retriever.py     lsa / sbert / bedrock backends, cosine index, fallback gates
src/entities.py      slot regexes + Bitext placeholder → Lex slot mapping
src/pipeline.py      FaqBot: classify → retrieve → extract → fallback → timing
src/train.py         fits and saves artifacts + manifest
src/evaluate.py      all metrics, confusion matrix, threshold sweep, error analysis
src/cli.py           local REPL / batch tester
lambda/              Lex V2 fulfillment handler, EMF metrics, DynamoDB logging
lex/                 bot definition generator (intents, slots, sample utterances)
infra/               DynamoDB table + CloudWatch alarm definitions
scripts/             sample data generator, analytics report
```

## Extending it

- **RAG via Bedrock.** When `fallback_reason == "out_of_vocabulary"`, pass the top-k
  retrieved entries to a Bedrock model as grounding context instead of returning a
  canned refusal. Keep the retrieval score in the prompt so the model can hedge.
- **Comprehend.** Sentiment on the input is a cheap escalation signal — negative
  sentiment plus a fallback is a strong "route to a human" trigger.
- **Active learning.** `error_analysis.csv` plus DynamoDB fallbacks are your labelling
  queue; the highest-value rows are confidently-wrong ones, not low-confidence ones.
