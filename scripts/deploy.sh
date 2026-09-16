#!/usr/bin/env bash
# Package and deploy the FAQ chatbot to AWS.
#
#   ./scripts/deploy.sh package          # build zips only (no AWS calls)
#   ./scripts/deploy.sh deploy           # package + publish layer + update function
#   ./scripts/deploy.sh infra            # create DynamoDB table + CloudWatch alarms
#
# Prereqs: awscli v2 configured, artifacts/ already built by `make train`.
set -euo pipefail

FUNCTION_NAME="${FUNCTION_NAME:-faq-chatbot-handler}"
LAYER_NAME="${LAYER_NAME:-faq-chatbot-artifacts}"
TABLE_NAME="${TABLE_NAME:-faq-chatbot-interactions}"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME="python3.12"

cd "$(dirname "$0")/.."

package() {
  echo "==> Building artifact layer"
  rm -rf build && mkdir -p build/layer/artifacts
  cp artifacts/intent_model.joblib artifacts/faq_index.joblib artifacts/manifest.json \
     build/layer/artifacts/
  (cd build/layer && zip -qr ../faq-artifacts-layer.zip .)

  echo "==> Building function package"
  rm -rf build/function && mkdir -p build/function
  cp -r src build/function/
  cp lambda/lambda_function.py build/function/
  find build/function -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  (cd build/function && zip -qr ../function.zip .)

  ls -lh build/*.zip
  echo
  echo "NOTE: scikit-learn/numpy/scipy are NOT in function.zip — they are too large."
  echo "Attach a scikit-learn Lambda layer (AWS SDK for pandas layer includes numpy/"
  echo "scipy, or build your own with 'sam build --use-container')."
}

deploy() {
  package
  echo "==> Publishing layer"
  LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --zip-file fileb://build/faq-artifacts-layer.zip \
    --compatible-runtimes "$RUNTIME" \
    --region "$REGION" \
    --query 'LayerVersionArn' --output text)
  echo "    $LAYER_ARN"

  echo "==> Updating function code"
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file fileb://build/function.zip \
    --region "$REGION" >/dev/null

  aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION"

  echo "==> Updating configuration"
  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --handler lambda_function.lambda_handler \
    --timeout 30 --memory-size 1024 \
    --environment "Variables={ARTIFACT_DIR=/opt/artifacts,INTERACTIONS_TABLE=$TABLE_NAME,METRIC_NAMESPACE=FaqChatbot}" \
    --region "$REGION" >/dev/null
  echo "    done. Attach $LAYER_ARN plus a scikit-learn layer to $FUNCTION_NAME."
}

infra() {
  echo "==> Creating DynamoDB table"
  aws dynamodb create-table --cli-input-json file://infra/dynamodb_table.json \
    --region "$REGION" >/dev/null 2>&1 || echo "    table exists, skipping"

  echo "==> Creating CloudWatch alarms"
  python - <<'PY'
import json, subprocess, os
region = os.environ.get("AWS_REGION", "us-east-1")
for a in json.load(open("infra/cloudwatch_alarms.json"))["Alarms"]:
    a.pop("_comment", None)
    cmd = ["aws", "cloudwatch", "put-metric-alarm", "--region", region]
    for k, v in a.items():
        cmd += [f"--{k[0].lower() + k[1:]}", str(v)]
    subprocess.run(cmd, check=False)
    print("   ", a["AlarmName"])
PY
}

case "${1:-package}" in
  package) package ;;
  deploy)  deploy ;;
  infra)   infra ;;
  *) echo "usage: $0 {package|deploy|infra}" && exit 1 ;;
esac
