# Netlib MPS Support Notes

This project's Netlib path currently targets a practical LP subset and converts it to standard form before calling `DualSimplex`.

## Supported MPS Sections

- `NAME`
- `OBJSENSE` (`MIN` / `MAX`)
- `ROWS` (`N`, `E`, `L`, `G`)
- `COLUMNS`
- `RHS`
- `BOUNDS` (`LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `LI`, `UI`, `BV`)
- `RANGES`
- `ENDATA`

## Objective Direction

- Internally, the solver minimizes.
- If `OBJSENSE MAX` is present, coefficients are negated during parse so the model is converted to minimization.

## `.mps.gz` Handling

- The reader does **not** parse compressed streams directly.
- Decompress `.mps.gz` files to plain `.mps` before running tests.
- The batch script and runner both assume plain `.mps` input files.

## Standardization Strategy

- All constraints are converted to inequality bounds and then to equalities by adding one slack/surplus variable per produced row.
- Every standardized row receives one dedicated slack column, which is used to build an explicit initial basis.
- Variable bounds are normalized to nonnegative variables (`x >= 0`) via shifts/splitting:
  - finite lower: shift (`x = l + y`)
  - finite upper only: flip (`x = u - y`)
  - free: split (`x = y_plus - y_minus`)
  - finite range: add upper-bound inequality for shifted variable
