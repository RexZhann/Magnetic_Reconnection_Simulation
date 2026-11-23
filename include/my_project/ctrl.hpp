#pragma once
// Control-deck (ctrl.txt) reader.
//
// Format (Gorgon-style namelist):
//   ! comment            -- everything after '!' is ignored (outside quotes)
//   &group ... &end      -- group markers, used only for readability
//   key = value          -- keys are case-insensitive; strings may be quoted;
//                           booleans: .true./.false./true/false/1/0
//   key = default        -- keep the test's built-in default (same as omitting)
//
// `test` is required. All other keys are optional: the run starts from
// make_config_for_test(test, nx, ny, divb, solver) and each key present in the
// deck overrides one field. Unknown keys are a hard error (catches typos).

#include "my_project/types.hpp"

#include <string>

namespace my_project {

// Parse the deck and return the fully resolved RunConfig.
RunConfig load_ctrl(const std::string& path);

// Recompute parameters derived from the physical inputs (test-28 Dirichlet
// y-boundary scalars from B1/B2/rho1/rho2). Called after any overrides.
void sync_derived_params(RunConfig& cfg);

// Apply env-var switches RESUME=1, IC_CHECK=1, L2_STRIDE=n (OR-ed with deck).
void apply_env_overrides(RunConfig& cfg);

// Expand a leading '~' and $VAR / ${VAR} references, drop trailing '/'.
std::string expand_path(const std::string& path);

// Final directory a run writes to: <output_dir>/<label>, or <output_dir> when
// label is empty.
std::string run_output_dir(const RunConfig& cfg);

// Human-readable dump of the resolved configuration.
void print_config(const RunConfig& cfg);

} // namespace my_project
