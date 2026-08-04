#!/usr/bin/env bash
set -euo pipefail

aws_profile="${AWS_PROFILE:-personal}"
stack_name="${DOCS_STACK_NAME:-relational-transformers-docs}"
aws_region="${AWS_REGION:-us-east-1}"

python -m sphinx -E -W -c docs -b html . docs/_build/html

aws cloudformation deploy \
  --profile "$aws_profile" \
  --region "$aws_region" \
  --stack-name "$stack_name" \
  --template-file infra/docs-cloudformation.yml \
  --no-fail-on-empty-changeset

bucket_name="$(aws cloudformation describe-stacks \
  --profile "$aws_profile" \
  --region "$aws_region" \
  --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
  --output text)"
distribution_id="$(aws cloudformation describe-stacks \
  --profile "$aws_profile" \
  --region "$aws_region" \
  --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs[?OutputKey==`DistributionId`].OutputValue' \
  --output text)"

aws s3 sync docs/_build/html "s3://${bucket_name}" \
  --profile "$aws_profile" \
  --delete \
  --cache-control 'public,max-age=300'
aws cloudfront create-invalidation \
  --profile "$aws_profile" \
  --distribution-id "$distribution_id" \
  --paths '/*' >/dev/null

echo "Deployed documentation to https://relationaltransformers.com"
