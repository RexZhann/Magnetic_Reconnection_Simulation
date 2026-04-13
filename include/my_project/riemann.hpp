#pragma once

#include "my_project/types.hpp"

namespace my_project {

Vec force_flux(const Vec& uL, const Vec& uR, double dt, double dx,
               double gamma, bool glm_on, double ch);
Vec hlld_flux(const Vec& uL, const Vec& uR, double gamma,
              bool glm_on, double ch);

} // namespace my_project
