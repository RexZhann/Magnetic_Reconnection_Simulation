#pragma once

#include "my_project/types.hpp"

namespace my_project {

double minbee(double r);
Vec pri2con(const Vec& w, double gamma);
Vec con2pri(const Vec& u, double gamma);
Vec phys_flux(const Vec& u, double gamma, bool glm_on, double ch);
double calc_cf(double rho, double p, double Bx, double By, double Bz, double gamma);

} // namespace my_project
