# iam_form_writers.csv: where it comes from

`form_id,writer_id` for all 1,539 IAM forms (657 writers). This is the same information as
column 2 of IAM's `ascii/forms.txt`, for downloads (like the Kaggle one) that don't include forms.txt.

**Source:** Sheng He & Lambert Schomaker's public IAM word lists for writer identification
(FragNet, IEEE TIFS 2020): https://www.ai.rug.nl/~sheng/writeridataset.html
(`writerset/IAM-train-list.txt` and `writerset/IAM-test-list.txt`). Each line there is
`train/<writer>-<word id>-<text>.png`, and the writer ID comes from IAM's forms.txt.

**Checks done when building it (2026-11-20):**
- 82,259 word entries → 1,539 forms and 657 writers, which are exactly IAM's published totals.
- No form maps to two different writers.
- It agrees with all 51 forms.txt lines in an independent public excerpt (Kaggle "iam_dataset_ascii").
- With the Aachen splits, writers per split are train 283 / val 56 / test 161, with 0 writers shared.

If your download has the real `ascii/forms.txt`, the code uses that instead.
