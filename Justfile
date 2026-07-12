# SpectralSVR task runner. Requires `just` (https://github.com/casey/just) and
# `uv`. Recipes run inside uv-managed environments (Python 3.13).

# list recipes
default:
    @just --list

# create/refresh the full dev environment
sync:
    uv sync --extra test --extra lint --extra notebook

# run the test suite (pass extra pytest args, e.g. `just test -k fourier`)
test *args:
    uv run --extra test pytest {{args}}

# run tests with coverage and print a missing-lines report
cov *args:
    # COVERAGE_CORE=sysmon avoids a torch+coverage C-tracer clash
    COVERAGE_CORE=sysmon uv run --extra test coverage run -m pytest {{args}}
    uv run --extra test coverage report -m

# lint + type-check (ruff, mypy, ty)
lint:
    uv run --extra lint ruff check SpectralSVR tests
    uv run --extra lint mypy SpectralSVR
    uv run --extra lint ty check SpectralSVR

# auto-format and apply ruff fixes
fmt:
    uv run --extra lint ruff format SpectralSVR tests
    uv run --extra lint ruff check --fix SpectralSVR tests

# individual linters
ruff:
    uv run --extra lint ruff check SpectralSVR tests

mypy:
    uv run --extra lint mypy SpectralSVR

ty:
    uv run --extra lint ty check SpectralSVR

# everything CI should gate on
check: lint test
