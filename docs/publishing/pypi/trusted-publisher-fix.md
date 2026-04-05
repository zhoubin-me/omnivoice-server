# PyPI Publishing - Configuration Required

## Issue

The publish workflow failed with:
```
invalid-publisher: valid token, but no corresponding publisher
```

This means **Trusted Publisher is not configured on PyPI yet**.

## Solution: Choose One Option

### Option 1: Configure Trusted Publisher on PyPI (Recommended)

This is more secure (no API tokens needed).

**Steps:**

1. Go to PyPI: https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name**: `omnivoice-server`
   - **Owner**: `maemreyo`
   - **Repository name**: `omnivoice-server`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave blank)
4. Click "Add"

**Then re-run the workflow:**
```bash
# Delete and recreate the release to trigger workflow
gh release delete v0.1.0 --yes
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes-file <(cat <<'EOF'
## Initial Release

OpenAI-compatible HTTP server for OmniVoice TTS.

### Installation
\`\`\`bash
pip install omnivoice-server
\`\`\`

See [README](https://github.com/maemreyo/omnivoice-server#readme) for full documentation.
EOF
)
```

### Option 2: Use API Token Instead

If you prefer using an API token:

**Steps:**

1. Get your PyPI API token (you mentioned you already created one)
2. Add it to GitHub Secrets:
   - Go to: https://github.com/maemreyo/omnivoice-server/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...` (your token)

3. Update `.github/workflows/publish.yml`:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
    verbose: true
```

Remove the OIDC comments and use the `password` parameter.

**Then push the updated workflow and re-trigger:**
```bash
git add .github/workflows/publish.yml
git commit -m "fix: use API token for PyPI publish"
git push origin main

# Delete and recreate release
gh release delete v0.1.0 --yes
gh release create v0.1.0 --title "v0.1.0 - Initial Release" --notes "..."
```

## Which Option to Choose?

- **Option 1 (Trusted Publisher)**: More secure, no secrets to manage, recommended by PyPI
- **Option 2 (API Token)**: Simpler setup, works immediately

I recommend **Option 1** for better security.

## Current Status

- ✅ Package builds successfully
- ✅ Workflow is configured correctly
- ❌ PyPI publisher configuration missing

Once you configure the publisher (Option 1) or add the API token (Option 2), the publish will work.
