# PyPI Publishing Checklist

## Pre-Publish Verification ✅

All items below have been verified and are ready:

- [x] **License consistency** - MIT across all files (LICENSE, pyproject.toml, README)
- [x] **GitHub Release v0.1.0** - Published with audio samples
- [x] **CI/CD workflow** - `.github/workflows/publish.yml` exists and configured
- [x] **Documentation complete** - README, CHANGELOG, SECURITY.md, issue templates
- [x] **Package metadata** - pyproject.toml has correct URLs, keywords, classifiers
- [x] **Default values consistent** - num_step=32 in config and examples
- [x] **Installation instructions** - README includes pip, uv, and GitHub install methods

## Publishing to PyPI

### Option 1: Automated via GitHub Actions (Recommended)

The publish workflow is already configured in `.github/workflows/publish.yml`.

**To trigger automatic publish:**

1. Create and push a new git tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. The workflow will automatically:
   - Build the package
   - Run tests
   - Publish to PyPI (requires `PYPI_API_TOKEN` secret)

**Setup PyPI token (one-time):**

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with scope: "Entire account" or specific to `omnivoice-server`
3. Add to GitHub Secrets:
   - Go to: https://github.com/maemreyo/omnivoice-server/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...` (your token)

### Option 2: Manual Publish

If you prefer to publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the built package
twine check dist/*

# Upload to TestPyPI first (optional, for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

**Note:** You'll need a PyPI account and API token. Set it up:
```bash
# Create ~/.pypirc
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

## Post-Publish Tasks

After successful PyPI publish:

- [ ] **Verify package on PyPI** - Check https://pypi.org/project/omnivoice-server/
- [ ] **Test installation** - `pip install omnivoice-server` in fresh virtualenv
- [ ] **Update README badge** - PyPI badge should show correct version
- [ ] **Announce release**:
  - [ ] Post to OmniVoice upstream discussions
  - [ ] Submit to awesome-tts lists
  - [ ] Share on relevant communities
- [ ] **GitHub Discussions** - Create "Announcements" category
- [ ] **Social preview image** - Add to GitHub repo settings
- [ ] **HuggingFace Space** - Consider creating demo (optional)

## Current Status

**Ready to publish:** ✅ All pre-publish checks passed

**Next step:** Choose Option 1 (automated) or Option 2 (manual) above to publish to PyPI.

## Notes

- Package version is `0.1.0` (defined in `pyproject.toml`)
- Package name is `omnivoice-server` (will be available as `pip install omnivoice-server`)
- CLI command will be `omnivoice-server` after installation
- GitHub Release v0.1.0 already exists with audio samples

## Troubleshooting

**If publish fails:**

1. Check PyPI token is valid and has correct permissions
2. Verify package name is not taken: https://pypi.org/project/omnivoice-server/
3. Check build artifacts in `dist/` directory
4. Review workflow logs in GitHub Actions

**If installation fails after publish:**

1. Wait 5-10 minutes for PyPI CDN to propagate
2. Try with `--no-cache-dir`: `pip install --no-cache-dir omnivoice-server`
3. Check PyPI project page for any warnings
