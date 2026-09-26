# Designing a GenAI Tool to Support Neurodivergent Learners in Studying Computer Science More Effectively

This repository contains supplementary materials associated with the master's thesis **Designing a GenAI Tool to Support Neurodivergent Learners in Studying Computer Science More Effectively**.

- **Author:** Ying Jiang
- **University:** University of Copenhagen
- **Submitted:** August 2026

## Related publication

Earlier research: Ying Jiang and Boris Düdder (2026). **A Trait-Based Prioritization Framework: Teaching Practices to Support Neurodivergent Learners in Computer Science Education.** In *Proceedings of the 18th International Conference on Computer Supported Education (CSEDU 2026)*, Volume 2, pp. 1877–1885. SciTePress. [Publisher record](https://www.scitepress.org/PublishedPapers/2026/145846/) · [DOI: 10.5220/0014584600004021](https://doi.org/10.5220/0014584600004021).

This conference paper is related earlier work, distinct from the master's thesis for which this repository provides supplementary materials.

## Public materials

- English survey instrument
- Chinese survey instrument
- Analysis code used to produce the reported thesis results
- Dependency information
- Selected aggregated outputs, where available

## Not publicly released

- Participant-level survey responses
- Raw survey datasets
- Item-level source-audit record
- Private research notes

Participant-level data are not publicly released, and this repository should not be interpreted as containing the complete research dataset.

The master's thesis reports a source audit; the item-level source-audit record is not included in this public repository.

## Repository structure

- `survey/`: English and Chinese questionnaire instruments
- `analysis/`: archived analysis scripts and dependency information
- `outputs/`: selected non-sensitive aggregated figures and tables
- `data/`: data-availability statement only; no row-level response data

The analysis directory contains a thesis-final supplementary snapshot prepared against the final submitted PDF. Selected aggregated outputs are included only where they match the final thesis and do not expose participant-level records. The underlying Excel exports, row-level intermediate CSV files, and participant-level PCA projection are excluded.

## Re-running the analysis

The analysis scripts require the non-public raw Excel files `chinese_survey.xlsx` and `english_survey.xlsx` and cannot be rerun from this public repository alone. Installing the dependencies does not supply these data. The files in `outputs/` are archived aggregate results corresponding to the final submitted thesis, not results regenerated when the repository is cloned. See [analysis notes](analysis/README.md) and [data availability](data/README.md) for the scope and limitations.

## Licensing

Source code in this repository is licensed under the MIT License; see `LICENSE`.

The questionnaires and other textual research materials are **not** licensed under the MIT License. No permission for reuse, redistribution, adaptation, or commercial use of those materials is granted unless an explicit license is added later.
