# ai-hiring-bias-causal-analysis

A causal investigation into an AI hiring tool: does it reflect genuine qualifications, or does it encode bias against protected attributes — and if bias is present, how much of it flows through the AI score itself versus somewhere else in the process?

AI screening tools now make (or heavily influence) hiring decisions before a human ever reviews an application. If a tool like that is quietly weighting gender or a proxy for socioeconomic background, that's not a modeling footnote — it's a decision that can end someone's shot at a job before they're ever seen. Understanding *where* bias enters an automated pipeline, not just whether it exists, is what makes it possible to actually fix it.

## Why this project

Most bias audits stop at "here's a correlation between a protected attribute and an outcome." This project goes a step further and asks: **is that relationship causal, and where in the pipeline does it happen** — directly, or through the AI's own score? That distinction matters, because the fix is different depending on the answer: direct discrimination points to the hiring process itself, while an AI-mediated gap points to the scoring tool.

This project is built around three questions, worked in this order:

1. **What is the AI's bias score actually weighting** — legitimate qualification signals, or attributes that double as demographic proxies (or the protected attributes directly)?
2. **Across protected attributes** (gender, university tier, education, age), **which one has the largest hiring gap that flows specifically through the AI score**, and does that finding hold up under stress-testing?
3. **At the individual level**, how many candidates sitting right at the hire/no-hire boundary would have gotten a different outcome if their AI score had simply been average instead of their actual score?

## A note on method

This project deliberately uses two different tools that answer two different kinds of questions, and keeps them from getting conflated:

- **SHAP** (Question 1) explains what a model is *paying attention to* — it's an explainability tool, not a causal one. It can tell you a feature matters to a prediction; it can't tell you that changing the feature would change the outcome in the real world.
- **DoWhy** (Question 2) is built for actual causal effect estimation — it requires stating a causal graph explicitly, checking whether the effect is identifiable, estimating it, and then stress-testing the estimate with refutation tests (e.g. swapping in a fake treatment and checking the effect collapses to zero, which it should if the original estimate is real).

Using SHAP to explain what's inside the AI's scoring model, and DoWhy to estimate what actually causes the hiring gap, is more honest than treating either tool as if it answers both questions.

## Data

[AI Hiring Bias and Fairness Benchmark](https://www.kaggle.com/datasets/sridipbasu/ai-hiring-bias-and-fairness-benchmark) — 5,000 candidates, no missing values, covering demographics (age, gender, education, university tier), qualifications (technical/coding/aptitude scores, experience, projects, certifications), two AI-generated scores (`ai_resume_score`, `ai_bias_score`), and the final hiring outcome.

**A limitation worth stating honestly:** the causal analysis in Question 2 only controls for qualification variables as confounders — it doesn't control for the *other* protected attributes when analyzing one of them (e.g. gender's estimated effect isn't adjusted for university tier). Some of what's attributed to one protected attribute could overlap with another. This is a real limitation of the current analysis, not something the notebook claims to have solved.

## Key findings

| Question | Finding |
|---|---|
| **What drives the AI score** | `technical_skill_score` is the top driver — expected from a legitimate scoring tool. But `university_tier` is a close second, ahead of `coding_test_score`, `aptitude_test_score`, and `years_experience`. `is_female` ranks 4th — meaning gender itself, not just a proxy for it, appears to directly influence the AI's score. |
| **Largest algorithmically-mediated gap** | University tier: 61.5% of its total effect on hiring flows through the AI score. Gender is close behind at 60.0%. Education barely mediates (13.8%). Age is an anomaly — its direct and indirect effects point in opposite directions, flagged rather than smoothed over. The university tier finding held up under a placebo refutation test (effect collapsed to ~0 when a fake treatment was substituted, p = 0.90). |
| **Individual boundary cases** | Of 249 candidates sitting right at the hire/no-hire line, 111 (44.6%) would have gotten a different outcome if their AI score had simply been average. The flipped group doesn't skew by gender or university tier compared to the boundary group overall — the mechanism is real and large, but this particular cut doesn't show it concentrated in one demographic group. |

## Repo structure

```
ai-hiring-bias-causal-analysis/
├── hiring_bias_analysis.ipynb    # full notebook
├── hiring_bias_analysis.py       # matching script version
├── AI_Hiring_Bias_Dataset.csv
└── README.md
```

The `.py` and `.ipynb` versions are kept in sync — the notebook is generated directly from the script.

## How to run

1. Clone the repo and make sure the CSV is in the same folder as the notebook.
2. Install the required packages:
   ```
   pip install pandas scikit-learn dowhy shap
   ```
3. Open `hiring_bias_analysis.ipynb` in Jupyter and run all cells top to bottom.

## Tools

- **pandas** — data loading, grouping, aggregation
- **scikit-learn** — logistic/linear regression, used both as a modeling tool and as the estimator underneath DoWhy
- **DoWhy** — causal graph specification, effect identification, estimation, and refutation testing
- **SHAP** — explainability for the model predicting the AI's bias score

## What's next

- Bring in visuals: a path/mediation diagram for the direct-vs-indirect effect split (Question 2), and a Sankey diagram tracing candidates through gender/AI score tier/hiring outcome.
- Extend the causal graph to control for multiple protected attributes simultaneously, addressing the overlap limitation noted above.
- Investigate the age anomaly from Question 2 further — the opposite-signed direct and indirect effects suggest a different mechanism than the one found for gender and university tier.

## Author

Keshia Neal, Ph.D.
