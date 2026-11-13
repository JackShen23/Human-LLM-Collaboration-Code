# Human–LLM Collaboration Data v1.0

The data archive is distributed separately from the GitHub code repository.

```text
_DATA/
├── README.md
├── DATA_TERMS.md
├── participants.csv
├── scenario_statistics.csv
└── participant_trials/P001.csv ... P202.csv
```

All participant identifiers are release-specific pseudonyms. The source-ID
mapping and personal attributes not required for the public analyses are not
included.

## `participants.csv`

One row per participant (`n = 202`).

| Column | Description |
|---|---|
| `participant_id` | Release pseudonym (`P001`–`P202`) |
| `ai_version` | Assigned experimental condition |
| `llm_experience` | Coarsened LLM-use frequency |
| `age_group` | Coarsened age category |
| `education` | Coarsened education category |
| `gender` | Coarsened self-reported gender category |

## `participant_trials/Pxxx.csv`

The analysis-ready interaction sequence for one participant.

| Column | Description |
|---|---|
| `participant_id` | Release pseudonym |
| `collaboration_round` | Interaction round, normally 1–60 |
| `scene_id` | Scenario identifier |
| `Filename` | Key joining `scenario_statistics.csv` |
| `ai_version` | Experimental condition |
| `decision_type` | 0 = reject, 1 = accept, 2 = timeout |
| `reaction_time` | Decision time in seconds |
| `round_score` | Trial score; 0 denotes a correct final decision |

There are 201 complete 60-round files and one 57-round file. Missing rounds are
not imputed; the incomplete sequence is excluded from PKHMM fitting.

## `scenario_statistics.csv`

One row per scenario, containing the scenario features used by PKHMM,
`correct_answer`, scenario `type`, and evaluated LLM outputs. The three type
labels are `avoid collision`, `efficiency`, and `injury mitigation`.

These longitudinal behavioral data are de-identified. 
Re-identification and linkage to external personal data are prohibited. 
See [DATA_TERMS.md](DATA_TERMS.md) and the Figshare record for the
applicable terms and license.
