# AWS FAQ Chatbot

Intent classification and semantic FAQ retrieval over customer-support text,
served from AWS Lambda with out-of-domain fallback detection, interaction
logging and an optional RAG layer.

```
Browser ──▶ Lambda ──▶ intent classifier ──▶ FAQ retrieval ──▶ answer
                │                │                    │
                │          numpy runtime       fallback gates
                ▼                                     │
         DynamoDB + CloudWatch ◀────────── rating ◀────┘
```

**Status:** deployed and answering on AWS Lambda at **35.9 ms** per request.
The RAG path is implemented and tested but has no generation metrics — see
[Known gaps](#known-gaps).

---

## Results

Held-out test set of 1,635 utterances, 27 intents.

| Intent classification | |
|---|---|
| Accuracy | 0.9988 |
| Macro precision / recall / F1 | 0.9988 / 0.9988 / 0.9988 |
| Weighted F1 | 0.9988 |

| Retrieval | |
|---|---|
| Top-1 accuracy | 0.9584 |
| Top-3 accuracy | 0.9694 |
| MRR | 0.9624 |

| Out-of-domain fallback | at threshold 0.45 | at 0.70 |
|---|---|---|
| Precision | 1.000 | 0.909 |
| Recall | 1.000 | 1.000 |
| False-alarm rate | 0.000 | 0.033 |
| Fallback rate | 0.002 | 0.083 |

| Serving | |
|---|---|
| Lambda response | 35.9 ms |
| Local inference | 0.13 ms/query |
| Artifact load (cold start) | 12.5 ms |
| Deployment package | 1.6 MB |

> The Bitext sample generates utterances from templates, so train and test share
> phrasings real users would not produce. Treat 0.9988 as an upper bound on this
> corpus, not as expected production accuracy. A hand-labelled set of realistic
> phrasings is the missing piece.

---

## Three findings

### A similarity threshold cannot detect out-of-domain questions

"What is the airspeed velocity of an unladen swallow" scored **0.858 cosine
similarity** against an account-creation FAQ. This is not a tuning problem: a
low-rank index projects unknown text onto whatever direction is nearest, so
nonsense can outscore legitimate paraphrases. No single threshold separates them.

`retriever.fallback_reason()` therefore uses three orthogonal gates — lexical
vocabulary coverage, absolute similarity, and intent confidence — and returns
*which* gate fired. `out_of_vocabulary` means a content gap in the knowledge
base; `ambiguous_intent` means two intents need better training utterances.
Different remediation, so the reason is logged, not just the verdict.

### One config value was silently doing two jobs

Raising the fallback threshold from 0.45 to 0.70 *reduced* top-1 retrieval
accuracy by 4.1 points. The same constant gated both "refuse to answer" and
"abandon the intent-filtered search and re-search the whole index", so raising
the refusal bar made the pipeline discard correct intent-filtered hits in favour
of global matches from the wrong intent.

| `--threshold 0.70` | coupled | decoupled |
|---|---|---|
| Top-1 accuracy | 0.9174 | **0.9584** |
| Fallback rate | 0.064 | 0.083 |
| p95 latency | 23.3 ms | 13.7 ms |

Splitting it into `threshold` and `retry_threshold` recovered the accuracy and
cut p95 latency 41%, because the pipeline stopped running a second unfiltered
search on nearly every query. Regression test:
`test_raising_fallback_threshold_does_not_reroute_retrieval`.

### The remaining errors are semantic overlap, not model error

| Gold intent | Errors | Most often confused with |
|---|---|---|
| `cancel_order` | 9 | `track_order`, `delivery_period` |
| `change_order` | 7 | `cancel_order`, `track_order` |
| `place_order` | 5 | `change_order` |

Half the errors sit in the order lifecycle; most of the rest are in the account
group. "I want to change my order" and "cancel my order and place a new one"
describe overlapping user situations — a human agent would ask which one rather
than guess. More training data will not fix this. The fix is a clarifying turn,
which the frontend implements: when the top two intents fall within 15 points,
it asks instead of answering.

---

## Quick start

Requires **Python 3.10+**.

```bash
make setup        # deps; on Windows use .\make.ps1 setup
make smoke        # generates stand-in data, trains, evaluates, runs tests
```

With the real dataset — [Bitext customer-support sample](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants)
at `data/bitext.csv`:

```bash
python scripts/adapt_dataset.py --in data/bitext.csv --out data/bitext_ready.csv
python -m src.train    --data data/bitext_ready.csv --out artifacts
python -m src.evaluate --artifacts artifacts --out reports --ood data/ood_questions.txt
python -m src.export_runtime --artifacts artifacts --out artifacts/runtime.npz
python -m src.cli --artifacts artifacts          # interactive prompt
```

### The dataset has no answers

The free Bitext sample ships `flags, utterance, category, intent` — utterances
and labels, no responses. There is nothing to retrieve.

`scripts/adapt_dataset.py` writes `data/faq_answers.csv`, one editable row per
intent, seeded with defaults. Edit it and re-run; your edits are preserved.

This is the right shape rather than a workaround. Support answers are authored
by the business and version-controlled; they are not a by-product of the
utterance corpus. The model learns *which question is being asked*, the company
decides *what the answer is*, and `faq_answers.csv` is that boundary made
explicit.

---

## How it answers

**Two-stage retrieval.** Classify the intent, then retrieve *within* it. Bitext
has ~300 paraphrases per intent sharing a handful of responses, so flat semantic
search over all FAQ entries confuses intents with overlapping vocabulary.
Restricting by predicted intent lifts top-1 accuracy — but only when the
classifier is confident. Below a confidence floor or a thin top-2 margin, the
pipeline drops the filter and searches globally rather than trusting a shaky
label.

**Calibrated confidence.** `CalibratedClassifierCV` wraps `LinearSVC` because an
uncalibrated SVM margin is not comparable across intents, and the fallback logic
needs a probability it can threshold.

**Pluggable embeddings.** `lsa` (TF-IDF + SVD) ships in the Lambda package at
1.6 MB and scores in 0.13 ms. `sbert` gives better paraphrase handling offline.
`bedrock` (Titan v2) removes the artifact at the cost of a network hop. Same
interface, so `evaluate.py` benchmarks all three.

---

## Deployment

The inference path is TF-IDF, a linear decision function, three sigmoids and a
matrix product — no scikit-learn needed at runtime. `src/export_runtime.py`
writes the fitted weights to numpy arrays and `src/runtime.py` scores them:

| | sklearn | numpy runtime |
|---|---|---|
| Unpacked dependencies | 193.5 MB | 39.8 MB |
| Model artifact | joblib pickles | 1.6 MB `.npz` |
| Docker required | yes (container image) | no |

Equivalence is asserted numerically, not assumed: across 657 queries the maximum
probability difference is **2.2e-16** with identical top-5 rankings
(`tests/test_runtime.py`). Building this caught a real bug — both sklearn paths
call `normalize()` *before* the vectorizer, and the first numpy version only
lowercased inside the analyzer, so `"I'd like to cancel; can't do it!!"`
retrieved a different FAQ.

```bash
./scripts/deploy_aws.ps1 deploy -Region us-east-1    # Windows
```

Creates the DynamoDB table (GSI + 90-day TTL), a least-privilege IAM role, and
the Lambda with numpy from the public AWS SDK for pandas layer. Verify:

```bash
aws lambda invoke --function-name faq-chatbot --payload fileb://build/ev.json out.json
```

### Frontend

```bash
python scripts/serve.py      # http://localhost:8000
```

Serves the chat UI and proxies to Lambda through the AWS SDK. Every answer shows
the predicted intent, classifier confidence, retrieval similarity, vocabulary
match, extracted entities, runner-up intents and Lambda time — the model's
decision is the product, so it is not hidden behind a "details" toggle.

Thumbs ratings write to DynamoDB, which is what gives `analytics.py` a
satisfaction rate to average.

---

## Monitoring and analytics

Each turn emits CloudWatch Embedded Metric Format (metrics from a log line, no
extra API call): response time, retrieval score, intent confidence, fallback
count, generated count, and the fallback reason as a dimension.

```bash
python scripts/analytics.py --source dynamodb --table faq-chatbot-interactions --days 7
```

Covers top FAQs, intent distribution, failed queries by fallback reason, response
accuracy, latency percentiles and satisfaction rate.

Alarms in `infra/cloudwatch_alarms.json`. The one worth watching is mean
retrieval similarity declining over hours — that is user language drifting away
from the training corpus, and it is the retrain trigger.

---

## RAG

Generation is a layer *on top of* retrieval, not a replacement. The retriever
still decides relevance and whether to answer; the generator only rewrites
already-trusted context.

An LLM given weak context does not say so — it writes a fluent wrong answer,
which for a support bot is worse than a refusal. So generation sits behind four
gates, any of which discards the output and falls back to the canned FAQ:

| Gate | Catches |
|---|---|
| `weak_retrieval` | context too thin — the model is never called, which is also the cost control |
| `invalid_citation` | a fabricated FAQ id, i.e. not reading the context |
| `no_citation` | unattributable claims |
| `low_groundedness` | answer tokens absent from retrieved text |

```bash
python -m src.evaluate_rag --artifacts artifacts --out reports    # offline baseline
python -m src.evaluate_rag --generator bedrock --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0
```

`ood_hallucination_rate` is the number that decides whether this path is
shippable. It is **0.0** — generation never fired on out-of-domain input,
because the fallback gates run first.

---

## Known gaps

**No generation quality metrics.** Bedrock returns
`AccessDenied: INVALID_PAYMENT_INSTRUMENT` — the account is Free Tier with no
payment method, and Bedrock is not covered by Free Tier. The RAG code,
guardrails and evaluation harness are complete and tested against an offline
generator; groundedness and lift over extractive on a real LLM are unmeasured.

**Public Function URL returns 403.** `AuthType: NONE` with a matching resource
policy, no AWS Organization, and both `curl` and PowerShell get Forbidden.
Unresolved. Not on the critical path — the SDK proxy reaches the same function.

**Typos defeat the vocabulary gate.** "for the patment" falls back with 0%
vocabulary match despite 0.611 similarity. Character n-grams in the classifier
handle this; the coverage gate does not. Fuzzy vocabulary matching would fix it.

**Templated evaluation data.** See the note under Results.

**Ratings are separate items.** Feedback rows share the session partition key but
are not joined to their specific turn. `analytics.py` averages any row carrying
`helpful`, which is adequate; per-turn attribution would need a `request_id` join.

---

## Layout

```
src/preprocess.py     Bitext loader, placeholder handling, FAQ corpus
src/intent_model.py   TF-IDF (word + char) → calibrated LinearSVC
src/retriever.py      lsa / sbert / bedrock backends, cosine index, fallback gates
src/entities.py       slot regexes, Bitext placeholder → Lex slot mapping
src/pipeline.py       classify → retrieve → extract → fallback → timing
src/rag.py            grounded generation: citations, groundedness, abstention
src/runtime.py        numpy-only inference (no sklearn at serve time)
src/export_runtime.py sklearn artifacts → numpy bundle
src/evaluate.py       metrics, confusion matrix, threshold sweep, error analysis
src/evaluate_rag.py   groundedness, citation validity, OOD hallucination rate
lambda/handler.py     serves both Lex V2 and HTTP events, EMF metrics, DynamoDB
lex/                  Lex V2 bot definition generator
scripts/serve.py      local frontend + SDK proxy
scripts/analytics.py  the six tracked metrics, from DynamoDB or CSV
infra/                DynamoDB table, CloudWatch alarms, IAM policies
tests/                53 tests
```

```bash
make test    # 53 tests, ~3s
```

Tests train a real (small) model rather than mocking one, because the behaviour
worth protecting — the fallback gates, two-stage routing, numpy/sklearn
equivalence — is a property of a fitted model.
`test_out_of_domain_triggers_fallback` pins the four nonsense queries that score
0.75–0.86 against unrelated FAQs, so it fails loudly if anyone simplifies the
fallback back to a bare similarity threshold.

---

## Licence and data

Code is yours to use. The Bitext dataset is not redistributed here — download it
from Kaggle under its own licence. `data/faq_answers.csv` contains illustrative
support policies, not any real company's terms.
