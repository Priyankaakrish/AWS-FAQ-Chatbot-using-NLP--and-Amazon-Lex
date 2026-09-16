<#
.SYNOPSIS
    Deploy the FAQ chatbot to AWS: DynamoDB table, IAM role, Lambda function
    (numpy runtime), and a public Function URL for the web frontend.

.DESCRIPTION
    Uses a **Lambda Function URL** rather than API Gateway. For a single POST
    endpoint the Function URL gives the same thing — HTTPS, CORS, throttling —
    with one fewer service to create, pay for and debug. API Gateway earns its
    place when you need custom domains, usage plans, request validation or
    multiple routes; none of which apply here.

    numpy comes from the public AWS SDK for pandas layer, so there is no
    Docker build and no 150 MB scikit-learn dependency (see src/runtime.py).

        .\scripts\deploy_aws.ps1 package    # build the zip only, no AWS calls
        .\scripts\deploy_aws.ps1 deploy     # create/update everything
        .\scripts\deploy_aws.ps1 test       # invoke the deployed endpoint
        .\scripts\deploy_aws.ps1 destroy    # tear it all down

.PARAMETER Action
    package | deploy | test | destroy
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('package', 'deploy', 'test', 'destroy')]
    [string]$Action = 'package',

    [string]$Region       = $(if ($env:AWS_REGION) { $env:AWS_REGION } else { 'us-east-1' }),
    [string]$FunctionName = 'faq-chatbot',
    [string]$TableName    = 'faq-chatbot-interactions',
    [string]$RoleName     = 'faq-chatbot-role',
    [string]$RagGenerator = 'none',        # 'bedrock' once billing allows it
    [string]$RagModelId   = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$Runtime = 'python3.12'

function Write-Step { param([string]$m) Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "    $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "    $m" -ForegroundColor Yellow }

function Assert-Prereqs {
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        throw "AWS CLI not found. Install AWS CLI v2 and run 'aws configure'."
    }
    if (-not (Test-Path 'artifacts\runtime.npz')) {
        throw "artifacts\runtime.npz missing. Run: python -m src.export_runtime --artifacts artifacts --out artifacts\runtime.npz"
    }
}

# ---------------------------------------------------------------- packaging --
function Invoke-Package {
    Write-Step "Building deployment package"
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path build\pkg\src, build\pkg\artifacts | Out-Null

    Copy-Item src\*.py build\pkg\src\
    Copy-Item lambda\handler.py build\pkg\
    Copy-Item artifacts\runtime.npz build\pkg\artifacts\

    # Training-only modules pull in sklearn/pandas, which are NOT in the
    # package. Remove them so an accidental import fails at build time here
    # rather than as a cold-start ImportError in production.
    foreach ($f in 'train.py','evaluate.py','evaluate_rag.py','export_runtime.py',
                   'intent_model.py','retriever.py','cli.py','artifacts.py') {
        Remove-Item "build\pkg\src\$f" -ErrorAction SilentlyContinue
    }
    # No stubs needed: the AWS SDK for pandas layer supplies pandas AND numpy.

    Compress-Archive -Path build\pkg\* -DestinationPath build\function.zip -Force
    $size = (Get-Item build\function.zip).Length / 1KB
    Write-Ok ("function.zip  {0:N0} KB" -f $size)
    if ($size -gt 50000) { Write-Warn "Package over 50 MB — direct upload will fail; use S3." }
}

# ------------------------------------------------------------------- deploy --
function Get-NumpyLayerArn {
    # Public AWS SDK for pandas layer ships numpy/pyarrow; we only need numpy.
    $base = "arn:aws:lambda:$Region`:336392948345:layer:AWSSDKPandas-Python312"
    $v = aws lambda list-layer-versions --layer-name $base --region $Region `
         --query 'LayerVersions[0].LayerVersionArn' --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v -or $v -eq 'None') {
        throw "Could not resolve the AWS SDK for pandas layer in $Region. Check the region supports it."
    }
    return $v
}

function New-Table {
    Write-Step "DynamoDB table: $TableName"
    $exists = aws dynamodb describe-table --table-name $TableName --region $Region 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Ok "already exists"; return }

    [System.IO.File]::WriteAllText("$PWD\build\gsi.json", '[{"IndexName":"intent-time-index","KeySchema":[{"AttributeName":"gsi1pk","KeyType":"HASH"},{"AttributeName":"gsi1sk","KeyType":"RANGE"}],"Projection":{"ProjectionType":"ALL"}}]')
    aws dynamodb create-table --region $Region `
        --table-name $TableName `
        --billing-mode PAY_PER_REQUEST `
        --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S `
                                AttributeName=gsi1pk,AttributeType=S AttributeName=gsi1sk,AttributeType=S `
        --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE `
        --global-secondary-indexes file://build/gsi.json | Out-Null
    aws dynamodb wait table-exists --table-name $TableName --region $Region
    aws dynamodb update-time-to-live --table-name $TableName --region $Region `
        --time-to-live-specification "Enabled=true,AttributeName=ttl" 2>$null | Out-Null
    Write-Ok "created"
}

function New-Role {
    Write-Step "IAM role: $RoleName"
    $arn = aws iam get-role --role-name $RoleName --query 'Role.Arn' --output text 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Ok "already exists"; return $arn }

    $trust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    [System.IO.File]::WriteAllText("$PWD\build\trust.json", $trust)
    $arn = aws iam create-role --role-name $RoleName `
            --assume-role-policy-document file://build/trust.json `
            --query 'Role.Arn' --output text
    aws iam attach-role-policy --role-name $RoleName `
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole | Out-Null

    $inline = '{"Version":"2012-10-17","Statement":[' +
      '{"Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:Query","dynamodb:Scan"],"Resource":"*"},' +
      '{"Effect":"Allow","Action":["bedrock:InvokeModel"],"Resource":"*"}]}'
    [System.IO.File]::WriteAllText("$PWD\build\inline.json", $inline)
    aws iam put-role-policy --role-name $RoleName --policy-name faq-chatbot-access `
        --policy-document file://build/inline.json | Out-Null

    Write-Ok "created — waiting 10s for IAM propagation"
    Start-Sleep -Seconds 10   # role creation is eventually consistent
    return $arn
}

function New-Function {
    param([string]$RoleArn, [string]$LayerArn)
    Write-Step "Lambda function: $FunctionName"
    $envVars = "Variables={INTERACTIONS_TABLE=$TableName,METRIC_NAMESPACE=FaqChatbot,ARTIFACT_DIR=/var/task/artifacts,RAG_GENERATOR=$RagGenerator,RAG_MODEL_ID=$RagModelId,ALLOWED_ORIGIN=*}"

    aws lambda get-function --function-name $FunctionName --region $Region 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        aws lambda update-function-code --function-name $FunctionName --region $Region `
            --zip-file fileb://build/function.zip | Out-Null
        aws lambda wait function-updated --function-name $FunctionName --region $Region
        aws lambda update-function-configuration --function-name $FunctionName --region $Region `
            --handler handler.lambda_handler --timeout 30 --memory-size 512 `
            --layers $LayerArn --environment $envVars | Out-Null
        Write-Ok "updated"
    } else {
        aws lambda create-function --function-name $FunctionName --region $Region `
            --runtime $Runtime --role $RoleArn --handler handler.lambda_handler `
            --zip-file fileb://build/function.zip --timeout 30 --memory-size 512 `
            --layers $LayerArn --environment $envVars | Out-Null
        Write-Ok "created"
    }
    aws lambda wait function-updated --function-name $FunctionName --region $Region
}

function New-FunctionUrl {
    Write-Step "Function URL"
    $url = aws lambda get-function-url-config --function-name $FunctionName --region $Region `
           --query 'FunctionUrl' --output text 2>$null
    if ($LASTEXITCODE -ne 0) {
        $cors = '{"AllowOrigins":["*"],"AllowMethods":["POST","OPTIONS"],"AllowHeaders":["content-type"],"MaxAge":300}'
        [System.IO.File]::WriteAllText("$PWD\build\cors.json", $cors)
        $url = aws lambda create-function-url-config --function-name $FunctionName --region $Region `
               --auth-type NONE --cors file://build/cors.json --query 'FunctionUrl' --output text
        # A Function URL with auth NONE still needs an explicit resource policy.
        aws lambda add-permission --function-name $FunctionName --region $Region `
            --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl `
            --principal "*" --function-url-auth-type NONE | Out-Null
    }
    Write-Ok $url
    [System.IO.File]::WriteAllText("$PWD\build\endpoint.txt", $url.Trim())
    return $url.Trim()
}

function Invoke-Deploy {
    Assert-Prereqs
    Invoke-Package
    $layer = Get-NumpyLayerArn
    Write-Ok "numpy layer: $layer"
    New-Table
    $roleArn = New-Role
    New-Function -RoleArn $roleArn -LayerArn $layer
    $url = New-FunctionUrl

    Write-Host "`nDeployed." -ForegroundColor Green
    Write-Host "Endpoint: $url" -ForegroundColor White
    Write-Host "Test it:  .\scripts\deploy_aws.ps1 test`n"
}

# --------------------------------------------------------------------- test --
function Invoke-Test {
    if (-not (Test-Path build\endpoint.txt)) { throw "Deploy first." }
    $url = (Get-Content build\endpoint.txt -Raw).Trim()
    foreach ($q in @('how do I reset my password',
                     'where is my order 4562781',
                     'who won the world cup in 1998')) {
        $body = @{ message = $q } | ConvertTo-Json -Compress
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType 'application/json'
        $sw.Stop()
        $tag = if ($r.is_fallback) { "FALLBACK($($r.fallback_reason))" } else { $r.intent }
        Write-Host "`nQ: $q" -ForegroundColor White
        Write-Host "   [$tag] conf=$($r.intent_confidence) sim=$($r.retrieval_score) " -NoNewline
        Write-Host "lambda=$($r.latency_ms)ms roundtrip=$($sw.ElapsedMilliseconds)ms" -ForegroundColor DarkGray
        Write-Host "A: $($r.answer)"
    }
    Write-Host ""
}

function Invoke-Destroy {
    Write-Step "Deleting resources"
    aws lambda delete-function-url-config --function-name $FunctionName --region $Region 2>$null | Out-Null
    aws lambda delete-function --function-name $FunctionName --region $Region 2>$null | Out-Null
    aws dynamodb delete-table --table-name $TableName --region $Region 2>$null | Out-Null
    aws iam delete-role-policy --role-name $RoleName --policy-name faq-chatbot-access 2>$null | Out-Null
    aws iam detach-role-policy --role-name $RoleName `
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>$null | Out-Null
    aws iam delete-role --role-name $RoleName 2>$null | Out-Null
    Write-Ok "done"
}

try {
    New-Item -ItemType Directory -Force -Path build | Out-Null
    switch ($Action) {
        'package' { Invoke-Package }
        'deploy'  { Invoke-Deploy }
        'test'    { Invoke-Test }
        'destroy' { Invoke-Destroy }
    }
} catch {
    Write-Host "`nFAILED: $($_.Exception.Message)`n" -ForegroundColor Red
    exit 1
}
