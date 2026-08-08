# P4.2 — IBKR Reference-Cost Profile Workflow

## Purpose

This document defines the manual maintenance and
activation process for the broker-disconnected IBKR
reference-cost model used by the BSAVCI Smart Investment
Bot.

The reference model exists only to estimate realistic
transaction economics for the paper portfolio. It does
not connect to IBKR, place broker orders, read an IBKR
account, or require credentials.

Current pinned profile:

- Provider: IBKR
- Profile path: `config/ibkr_reference_costs_v1.json`
- Profile version: `ibkr-reference-2026-08-08-v1`
- API connection: disabled
- Cost gate: disabled
- Active pricing plan: unresolved
- Active FX mode: unresolved

## Safety rule

The cost gate remains disabled until the actual account
pricing plan and FX mode have been explicitly confirmed,
encoded in a reviewed configuration change, and covered
by deterministic tests.

No code may infer Fixed versus Tiered pricing from trade
size, geography, account currency, or any other proxy.

No code may infer SPOT_FX versus AUTO_CONVERSION from the
portfolio currency.

## When IBKR pricing changes

Use this workflow whenever a commission, minimum,
regulatory fee, exchange fee, clearing fee, fractional
share rule, minimum notional, or FX conversion rule used
by the model changes.

### 1. Verify the source

Review official IBKR pricing material.

Use the official IBKR source references stored in the
current profile as the starting point. If the relevant
official page has moved or a new official page is needed,
record that source in the replacement profile.

Do not use forum posts, summaries, advertisements, or
third-party calculators as the authoritative pricing
source.

### 2. Create a new version

Do not silently rewrite historical economics.

Create a new versioned profile rather than changing the
meaning of an already released profile without a version
change.

For example:

`config/ibkr_reference_costs_v2.json`

The new profile must have a new `profile_version` and a
fresh verification date.

The prior profile remains in Git history and should
remain reproducible for historical analysis.

### 3. Update pricing rules

Update only rules supported by verified official IBKR
material.

Review at minimum:

- stock commission rates;
- minimum commission;
- maximum commission where relevant;
- regulatory fees;
- clearing fees;
- exchange or route-dependent charges;
- fractional-share rules;
- minimum-notional rules;
- spot FX commission;
- automatic currency-conversion economics.

Market-specific European fees must remain market-specific.
Do not turn a reference example for one venue into a
universal Europe rule.

### 4. Recalculate hand examples

For each modified rule, add or update hand-calculated
examples in automated tests.

At minimum validate:

- a small whole-share trade;
- a fractional-share case where supported;
- a sell-side regulatory-fee example;
- FX conversion;
- round-trip reward-path cost;
- round-trip risk-path cost;
- net reward-to-risk;
- a trade close to EUR 100 that becomes uneconomic after
  costs when the numbers warrant rejection.

The hand-calculated values and the implementation must
match before the profile can be adopted.

### 5. Run regression

Run the focused reference-cost tests first.

Then run the execution-engine tests, P4.1 sizing/FX
regressions, and the full regression suite.

`git diff --check` must be clean.

The operational paper database must not be modified by a
reference-profile maintenance change.

### 6. Update the product-policy pin

Only after the replacement profile is validated, update:

- `reference_profile_path`;
- `reference_profile_version`;
- product `policy_version`.

The product policy and reference profile must identify the
same provider and profile version.

### 7. Resolve account-specific pricing separately

Profile maintenance and account activation are different
decisions.

Before enabling the cost gate, explicitly establish the
actual IBKR account:

- pricing plan: Fixed or Tiered;
- FX mode used for intended transactions;
- whether FX conversion is modelled on entry;
- whether FX conversion is modelled on exit;
- relevant market or routing assumptions;
- fractional-share eligibility when fractional execution
  is later enabled.

If the pricing plan is Tiered and required route-dependent
cost information is unavailable, the model must fail
closed rather than treat the estimate as complete.

### 8. Activation review

Activation requires a separate reviewed change.

The change must explicitly set the intended pricing plan,
FX mode, and conversion assumptions. It must demonstrate
that representative trades receive the expected
cost-adjusted acceptance or rejection result.

Until that review is complete:

- `ibkr_cost_gate_enabled` remains `false`;
- `ibkr_pricing_plan` remains `null`;
- `ibkr_fx_mode` remains `null`;
- entry FX conversion remains `false`;
- exit FX conversion remains `false`.

### 9. Credentials and connectivity prohibition

The P4.2 reference-cost model requires no credentials.

Do not add:

- broker usernames or passwords;
- API keys;
- API tokens;
- session tokens;
- account secrets;
- private keys;
- IBKR API connection code.

P4.2 is a reference economics layer only. Manual copy to
IBKR may be allowed by product policy, but broker
connectivity is outside this phase.

## Release evidence

A profile or activation update is acceptable only when
the repository records:

- exact profile version;
- official source references;
- hand-calculated test evidence;
- focused tests;
- full regression;
- clean diff integrity;
- secret-isolation validation;
- unchanged operational database when the change is
  configuration/model-only.

Any unresolved pricing assumption remains explicit and
fail-closed.
