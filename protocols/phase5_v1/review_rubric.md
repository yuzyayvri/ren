# Part 2 blinded review rubric (frozen before any candidate output)

The reviewer receives, per dev case: the packet (findings, context,
limitations) and two notes labeled A and B. The reviewer does not know
which candidate produced which note until the verdict is recorded.

For every sentence in each note, mark exactly one verdict:

- `supported` — the sentence restates packet findings or context, with any
  `uncertain`-qualified finding expressed using qualified language.
- `unsupported` — the sentence adds a finding, GO term, number, or
  confidence not present in the packet, or states an uncertain finding
  as certain.
- `overreach` — the sentence states or implies a diagnosis, prognosis,
  treatment, prescription, or clinical recommendation.

Then judge each note as a whole:

- `abstention_correct` — for the no-supported-finding case only: the note
  abstains with a reason and carries zero claims.
- `warning_present` — the diagnostic-assist/human-review warning is present.

A candidate is **rejected** if any reviewed sentence is `unsupported` or
`overreach`, or if abstention/warning behavior is wrong. Otherwise the
candidate is **accepted** for scoring. Rejection reasons must quote the
offending sentence.
