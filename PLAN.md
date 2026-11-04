# nanoCVXPY: design plan (v0, draft for iteration)

## Context

cvxpy is about 45k lines of Python, plus a C++ backend. Most of that is machinery the basics don't need:
- DPP parameter tensors
- multiple canon backends
- the `ParamConeProg` and `ConeMatrixStuffing` reordering
- DGP, DQCP, complex numbers, PSD, ND arrays
- the solver zoo
- atom logic spread over four places: the atom class, the Dcp2Cone canon, the DNLP smooth canon, and the diff-engine registry

The DNLP branch (`DNLP/cvxpy`, `feat/diffengine-dcp-backend`) already shows that SparseDiffPy can do two jobs:
1. Act as the Ipopt oracle: values, sparse Jacobian, and the lower triangle of the Lagrangian Hessian.
2. Extract the conic data `q, P, A, b` for Clarabel by evaluating at x = 0. See `diffengine_cone_program.py:308`.

So one converter can serve both paths, and every LinOp, cvxcore and coefficient-extractor layer can go.

**Goal:** the smallest readable library that behaves like cvxpy for the basics.
- **Default mode, DNLP:** solved with Ipopt via cyipopt.
- **Opt-in mode, `dcp=True`:** solved with Clarabel.
- **Runtime dependencies:** numpy, scipy, sparsediffpy, cyipopt, clarabel. Nothing else.

**Target size:** about 1.5–2k lines total.

## Explicit non-goals

- DQCP, DGP, DPP rules, or any grammar besides DCP and DNLP
- PSD/SDP, quantum or exotic atoms, and complex numbers
- ND arrays (above 2-D), and solver choice beyond Ipopt and Clarabel
- Reduction chains, `ParamConeProg`, cone reordering, warm-start tensors, `get_problem_data`
- Variable attributes beyond `nonneg`, `nonpos` and `bounds`; no symmetric, PSD, integer or boolean variables

## Core design ideas

### 1. One composition rule for both grammars

Each expression gets a `(convex, concave)` pair, computed by a single function:

```python
def curvature(e, grammar):                 # -> (cvx, ccv)
    if e.is_constant() or isinstance(e, Leaf): return (True, True)
    a_cvx, a_ccv = e.atom_curvature(grammar)
    cvx, ccv = a_cvx, a_ccv
    for i, arg in enumerate(e.args):
        c, v = curvature(arg, grammar)
        base = c and v                     # affine (DCP) / smooth (DNLP)
        cvx &= base or (c and e.incr(i)) or (v and e.decr(i))
        ccv &= base or (v and e.incr(i)) or (c and e.decr(i))
    return cvx, ccv
```

The only difference between the grammars is the atom flags:

| Grammar | `atom_curvature` returns | Effect |
|---|---|---|
| DCP | `(spec.convex, spec.concave)` | |
| DNLP | `(spec.convex or spec.smooth, spec.concave or spec.smooth)` | "Both true" now means smooth, not affine |

This reproduces `is_linearizable_convex`/`is_linearizable_concave` from DNLP `atom.py:267-284` exactly. The problem-level checks are shared too:
- `Minimize` needs cvx, `Maximize` needs ccv.
- `==` needs both.
- `<=` needs cvx of `lhs - rhs`.

### 2. One declarative spec per atom, everything in one place

An atom is a small spec object, not a subclass with six or more overridden methods. It holds:
- name
- `numeric`: a NumPy fallback, only for nonsmooth atoms the engine lacks; values otherwise come from a SparseDiffPy forward pass
- `shape` rule (elementwise broadcast, reduce, or custom)
- `sign` rule
- `convex` / `concave` / `smooth`
- `incr` / `decr`, either bools or `lambda args, i: ...` for sign-dependent cases like `abs`
- `lower`: how to build the SparseDiffPy node
- `cone`: DCP canon, returning `(expr, [cone constraints])`
- `smoothen`: DNLP epigraph canon, needed only for nonsmooth atoms

Most atoms take one line each:

```python
exp  = unary("exp",  np.exp,  de.make_exp,  convex=True, incr=True, sign=NONNEG, smooth=True, cone=exp_cone)
log  = unary("log",  np.log,  de.make_log,  concave=True, incr=True, smooth=True, cone=log_cone)
sin  = unary("sin",  np.sin,  de.make_sin,  smooth=True)      # DNLP only (no cone canon)
abs_ = unary("abs",  np.abs,  None, convex=True, incr=pos_arg, decr=neg_arg, sign=NONNEG,
             cone=abs_epi, smoothen=abs_epi)                  # LP epigraph serves both modes
```

Many canons serve both modes. The epigraphs for `abs`, `max`, `norm1` and `norm_inf` work as DCP canons and as DNLP smoothings, because they introduce only linear constraints.

### 3. One converter: Expression → SparseDiffPy capsule

`lower.py` walks the tree once:
- A variable becomes `make_variable(d1, d2, offset, n)`.
- A constant, or a constant subtree folded by an engine forward pass, becomes `make_parameter(..., -1, vals)`.
- Every other atom calls `spec.lower(*children)`.
- Matmul and multiply pick dense or sparse constant forms, following `helpers.py` in DNLP.
- The converter checks shapes in Python, because the engine uses `assert`/`exit(1)`.

Both solver paths call it. The patterns to port are DNLP `diff_engine/converters.py`, `registry.py` and `helpers.py`.

### 4. DNLP path (default) → Ipopt

1. Check DNLP.
2. Run `smoothen` bottom-up. It replaces nonsmooth atoms with new variables plus linear constraints, and sets each new variable's `.value` for the initial point.
3. Flatten the constraints:
   - `==` gets `cl = cu = 0`.
   - `<=` becomes `g = lhs - rhs` with `cl = -inf`, `cu = 0`.
   - Variable bounds become Ipopt `lb`/`ub`.
4. Build one capsule with `make_problem(obj, cons)`. A ~60-line `Oracles` class handles caching: it tracks the last x and re-runs forward when it changes.
5. Initial point: the user's value, else the midpoint of the bounds, else `lb+1` or `ub-1`, else 0. Same as DNLP `nlp_solving_chain.py:67`.
6. Unpack x by offsets. `mult_g` slices become the constraint duals.

### 5. DCP path (`dcp=True`) → Clarabel, with no reordering layer

1. Check DCP.
2. Canonicalize with `cone` bottom-up into constraints of the form `affine ∈ K`, with K one of Zero, NonNeg, SOC or Exp.
   - Quadratic atoms stay in the objective when every ancestor is affine (cvxpy's `affine_above` trick). These are `sum_squares`, `quad_form`, `square`/`power(·,2)` and `quad_over_lin` with a constant denominator.
3. Build one capsule. The objective is the possibly-quadratic objective. The constraints are each cone block's affine expression, in the order created.
4. Evaluate at x = 0 to get the data:
   - `d`, `q` from the objective value and gradient
   - `P` from the Hessian with zero duals, upper triangle only
   - `g0`, `J` from the constraint values and Jacobian
5. Pass Clarabel `A = -J`, `b = g0`, cones = `[ZeroConeT(k1), NonnegativeConeT(k2), SecondOrderConeT(..), ...]` **in creation order**.
   - **This is the big simplification:** Clarabel accepts any cone sequence, so there is no grouping by cone type and no row permutation.
   - The exp cone needs `(x, y, z)` interleaved per element. One `make_index` over a `vstack` does that.
6. Duals: each user constraint remembers its row slice in z.

### 6. Expressions and shapes

- Shapes are 0-D, 1-D or 2-D with NumPy broadcasting. Internally everything is 2-D, column-major, with `(n,)` mapped to `(n, 1)`.
- `Variable(shape, nonneg=, nonpos=, bounds=, name=)`, `Constant`, `Parameter`.
- Operators: `+ - * @ / ** [] .T`, `==`, `<=`, `>=`.

## Proposed atom set (v1)

| Kind | Atoms |
|---|---|
| Affine | add, neg, matmul (const@expr, expr@const), multiply (by const), divide by const, index/slice, transpose, reshape, vec, sum(axis), hstack, vstack, diag, trace, cumsum, promote/broadcast |
| Smooth convex / concave | exp, log, entr, logistic, power(p), square, sqrt, inv_pos, quad_form, sum_squares, quad_over_lin, log_sum_exp, rel_entr |
| Nonsmooth convex / concave | abs, pos, neg, max, min, maximum, minimum, norm1, norm2, norm_inf, huber |
| DNLP only (smooth, neither convex nor concave) | sin, cos, tan, sinh, tanh, asinh, atanh, xexp, normcdf, prod, multiply(expr, expr), matmul(expr, expr) |

The cones needed are LP, SOC and Exp. In DCP mode, `power` is restricted to p in {1, 2, 0.5, -1}, each handled with one SOC.

## Package layout and line budget

```
nanocvxpy/
  __init__.py        public API                                           ~40
  expression.py      Expression, Variable, Constant, Parameter, ops       ~300
  atoms.py           spec class + declarative atom table                  ~450
  curvature.py       the one composition rule, sign helpers               ~60
  constraints.py     Equality, Inequality (+ internal Zero/NonNeg/SOC/Exp) ~80
  canon.py           cone canons + smooth canons (shared epigraphs)       ~300
  lower.py           Expression → sparsediffpy                            ~200
  problem.py         Minimize/Maximize/Problem.solve(dcp=False)           ~150
  ipopt.py           oracles + cyipopt call                               ~120
  clarabel.py        data extraction + clarabel call                      ~100
tests/               compare against DNLP-cvxpy (test-only dependency)
```

## Reuse sources

| What | Where |
|---|---|
| Converter patterns | `DNLP/cvxpy/reductions/solvers/nlp_solvers/diff_engine/{converters,registry,helpers}.py` |
| Ipopt oracles and bounds | `DNLP/.../nlp_solvers/nlp_solver.py:80-240`, `ipopt_nlpif.py:121-200` |
| Conic extraction via the engine | `DNLP/cvxpy/reductions/dcp2cone/diffengine_cone_program.py:308-373` |
| Smooth canons | `DNLP/cvxpy/reductions/dnlp2smooth/canonicalizers/` |
| Cone canons | `cvxpy/reductions/dcp2cone/canonicalizers/` and `eliminate_pwl/canonicalizers/` |
| `gm_constrs` | `cvxpy/utilities/power_tools.py` |

## Verification

- A pytest suite of about 30 small problems: LP, QP, SOCP, exp-cone, and DNLP problems (trig, bilinear, nonsmooth with smoothing). Each one is solved with nanoCVXPY and with DNLP-cvxpy, and the optimal values and solutions must match to 1e-5.
- Grammar unit tests: a table of expressions with expected DCP and DNLP verdicts, mirroring the DNLP branch's tests.
- A line-count check (`cloc nanocvxpy`) to keep within budget.

## Step 0: repo setup (done)

- Cloned `Transurgeon/nanoCVXPY` (empty) as `origin`; `SparseDifferentiation/nanoCVXPY` added as `upstream`.

## Decisions so far

| Question | Decision |
|---|---|
| Parameters | Yes, with no DPP. Parameters are engine leaves updated with `problem_update_params`, and the compiled capsule plus Clarabel/Ipopt data are cached on the Problem. Canons that depend on values, such as the `quad_form` eigendecomposition in a DCP constraint, raise an error if given a parameter. For the DCP check, parameters count as constants with sign attributes. |
| Duals | Yes. Each user constraint records its row slice in Ipopt's `mult_g` or Clarabel's `z`. |
| API | Drop-in for the basics: same names as cvxpy (`Variable`, `Parameter`, `Problem`, `Minimize`, `Maximize`, `sum_squares`, `norm1`, …). This lets DNLP-cvxpy act as the test oracle. |
| Power | In DCP mode, only p in {1, 2, 0.5, -1}, giving `square`, `sqrt` and `inv_pos` via a single SOC each. Any p is allowed in DNLP mode. Do not port `gm_constrs`. |
| Values | `.value` and constant folding use a SparseDiffPy forward pass: lower the (sub)expression, then run forward at the current variable and parameter values. Atoms carry no NumPy code, except nonsmooth atoms the engine lacks (abs, max, min, maximum, minimum, norm1, norm_inf, norm2, huber, pos, neg). Those get a one-line `numeric` fallback that is used only for `.value`. |
| DCP check | `solve(dcp=True)` on a non-DCP problem raises `DCPError`. There is no fallback. |
| SparseDiffPy | Target 0.7.x. |
| best_of | Keep `best_of=N` random restarts for Ipopt: sample within variable bounds (or `sample_bounds`), keep the best objective, and record all objectives in `solver_stats`. About 30 lines. |
| Atom set | Keep all three proposed groups (smooth convex/concave, nonsmooth convex/concave, DNLP-only smooth) plus the affine atoms. No additions for v1. |

## Open questions (to iterate on)

None right now.
