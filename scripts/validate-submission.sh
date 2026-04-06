#!/bin/bash

# Pre-submission validation script for the OpenEnv Hackathon.
#
# Usage:
#   validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url: The URL of your deployed HF Space (e.g., https://user-space.hf.space)
#   repo_dir: Path to your repository (default: current directory)

set -e

PING_URL=$1
REPO_DIR=${2:-"."}

if [ -z "$PING_URL" ]; then
    echo "Usage: $0 <ping_url> [repo_dir]"
    exit 1
fi

echo "--- VALIDATING OPENENV SUBMISSION ---"
echo "PING_URL: $PING_URL"
echo "REPO_DIR: $REPO_DIR"

# 1. HF Space Deployment Check
echo "1. Pinging Space URL..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/health" || echo "000")
if [ "$STATUS" -eq 200 ]; then
    echo "   [PASS] Space is live and responds to /health"
else
    echo "   [FAIL] Space at $PING_URL/health returned $STATUS (expected 200)"
    exit 1
fi

# 2. Reset Endpoint Check
echo "2. Testing /reset endpoint..."
STATUS=$(curl -s -X POST -o /dev/null -w "%{http_code}" "$PING_URL/reset")
if [ "$STATUS" -eq 200 ]; then
    echo "   [PASS] /reset returned 200"
else
    echo "   [FAIL] /reset returned $STATUS (expected 200)"
    exit 1
fi

# 3. File Structure Check
echo "3. Checking required files in $REPO_DIR..."
REQUIRED_FILES=("inference.py" "openenv.yaml" "Dockerfile" "README.md" "models.py" "tasks.py")
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$REPO_DIR/$f" ]; then
        echo "   [PASS] Found $f"
    else
        echo "   [FAIL] Missing $f"
        exit 1
    fi
done

# 4. OpenEnv Spec Compliance
echo "4. Validating openenv.yaml..."
if grep -q "tasks:" "$REPO_DIR/openenv.yaml" && grep -q "reward_range:" "$REPO_DIR/openenv.yaml"; then
    echo "   [PASS] openenv.yaml contains tasks and reward_range"
else
    echo "   [FAIL] openenv.yaml is malformed or missing key fields"
    exit 1
fi

# 5. Inference Script Check
echo "5. Checking inference.py..."
if grep -q "OpenAI(" "$REPO_DIR/inference.py" || grep -q "OpenAIclient" "$REPO_DIR/inference.py"; then
    echo "   [PASS] inference.py appears to use OpenAI client"
else
    echo "   [FAIL] inference.py must use OpenAI client"
    exit 1
fi

if grep -q "\[START\]" "$REPO_DIR/inference.py" && \
   grep -q "\[STEP\]" "$REPO_DIR/inference.py" && \
   grep -q "\[END\]" "$REPO_DIR/inference.py"; then
    echo "   [PASS] inference.py emits [START], [STEP], and [END] logs"
else
    echo "   [FAIL] inference.py must emit mandatory stdout logs"
    exit 1
fi

echo "--- VALIDATION SUCCESSFUL - READY TO SUBMIT ---"
