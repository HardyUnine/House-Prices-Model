# Steps to Implement DOE



Define your objective (screening vs. optimization).

Select factors & levels based on domain knowledge.

Choose a design (full vs. fractional factorial, response-surface, etc.).

Generate the design matrix (e.g., with Python’s pyDOE2 or simple Cartesian product for small k).

Run the experiments in randomized order, collecting your response.

Analyze via regression/ANOVA to estimate main and interaction effects.

Validate & verify by running confirmation experiments at the “optimal” settings.