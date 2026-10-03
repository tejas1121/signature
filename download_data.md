# Downloading public signature data

The loader expects genuine images below a folder for each identity:

```text
data/raw/<person_id>/<image>.png
```

It also accepts deeper folders such as
`data/raw/<person_id>/genuine/<image>.png`. Folder names containing `forg`,
`forged`, `forgery`, `forgeries`, or `negative`, and identity folder names
ending in `_forg` or `_forged`, are excluded. Only genuine signatures are used
for this identity-classification experiment; forged samples are not used as
training examples.

## Kaggle CLI

1. Install and configure the Kaggle CLI using the official instructions at
   <https://github.com/Kaggle/kaggle-api>. Accept the dataset's terms on
   Kaggle first if required; do not put API tokens in this repository.
2. From this project's root, download the public Signature Verification
   Dataset (Kaggle slug `robinreni/signature-verification-dataset`):

   ```powershell
   kaggle datasets download -d robinreni/signature-verification-dataset --unzip -p data/kaggle-download
   ```

3. Inspect the extracted folders. Copy or reorganize **genuine** samples into
   `data/raw/<person_id>/genuine/`. For datasets with `*_forg` sibling folders,
   do not copy those folders. Keep identities as folder names. Avoid adding
   wrapper folders such as `train` between `data/raw` and identity folders;
   instead set `SIGNATURE_DATA_DIR` to the directory whose immediate children
   are identities.
4. On PowerShell, you can set an alternate data root for subsequent commands:

   ```powershell
   $env:SIGNATURE_DATA_DIR = "C:\path\to\dataset"
   ```

   In Command Prompt, use `set SIGNATURE_DATA_DIR=C:\path\to\dataset`.

The same loader works with CEDAR, ICDAR SigComp, BHSig260, or another
per-identity image dataset after arranging the folders as above. Do not use a
verification pair list as labels; this project's target is identity.

## Manual download

1. Open the Kaggle dataset page at
   <https://www.kaggle.com/datasets/robinreni/signature-verification-dataset>
   (or obtain CEDAR, ICDAR SigComp, or BHSig260 from its official/public
   distribution page), sign in, and accept any data-use terms.
2. Choose **Download**, then extract the archive locally.
3. Create `data/raw/<person_id>/genuine/` for each writer and place only
   genuine signature image files there. Preserve a stable writer ID as the
   folder name. Ensure each identity has at least three images; substantially
   more examples per writer are recommended.
4. Run the loader/training commands from the project root. If the folder lives
   elsewhere, set `SIGNATURE_DATA_DIR` as described above.

The bundled `python make_demo_data.py` is the no-download alternative. Those
rendered samples exercise the code path only and must not be presented as a
real-world accuracy benchmark.
