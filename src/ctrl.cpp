#include "my_project/ctrl.hpp"
#include "my_project/solver.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace my_project {

namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Strip a trailing '!' comment that is not inside quotes.
std::string strip_comment(const std::string& line) {
    char q = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (q) { if (c == q) q = 0; }
        else if (c == '\'' || c == '"') q = c;
        else if (c == '!') return line.substr(0, i);
    }
    return line;
}

std::string unquote(const std::string& v) {
    if (v.size() >= 2 && (v.front() == '\'' || v.front() == '"') && v.back() == v.front())
        return v.substr(1, v.size() - 2);
    return v;
}

struct Entry { std::string value; int line; };

double to_double(const std::string& key, const Entry& e) {
    // Accept Fortran-style exponents (1d-3).
    std::string v = e.value;
    for (char& c : v) if (c == 'd' || c == 'D') c = 'e';
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || trim(end) != "")
        throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                                 + "' expects a number, got '" + e.value + "'");
    return x;
}

int to_int(const std::string& key, const Entry& e) {
    const double x = to_double(key, e);
    if (x != static_cast<double>(static_cast<long long>(x)))
        throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                                 + "' expects an integer, got '" + e.value + "'");
    return static_cast<int>(x);
}

bool to_bool(const std::string& key, const Entry& e) {
    const std::string v = lower(e.value);
    if (v == ".true." || v == "true" || v == "t" || v == "1" || v == "yes") return true;
    if (v == ".false." || v == "false" || v == "f" || v == "0" || v == "no") return false;
    throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                             + "' expects .true./.false., got '" + e.value + "'");
}

DivBCleaningKind to_divb(const std::string& key, const Entry& e) {
    const std::string v = lower(e.value);
    if (v == "ct" || v == "2") return DivBCleaningKind::CT;
    if (v == "glm" || v == "1") return DivBCleaningKind::GLM;
    if (v == "none" || v == "0") return DivBCleaningKind::None;
    throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                             + "' must be 'CT', 'GLM' or 'none'");
}

SolverKind to_solver(const std::string& key, const Entry& e) {
    const std::string v = lower(e.value);
    if (v == "hlld" || v == "1") return SolverKind::HLLD;
    if (v == "force" || v == "0") return SolverKind::FORCE;
    throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                             + "' must be 'HLLD' or 'FORCE'");
}

HallStabKind to_hall_stab(const std::string& key, const Entry& e) {
    const std::string v = lower(e.value);
    if (v == "none" || v == "0") return HallStabKind::NONE;
    if (v == "hyper_res" || v == "1") return HallStabKind::HYPER_RES;
    if (v == "hall_hll" || v == "2") return HallStabKind::HALL_HLL;
    throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                             + "' must be 'none', 'hyper_res' or 'hall_hll'");
}

BC to_bc(const std::string& key, const Entry& e) {
    const std::string v = lower(e.value);
    if (v == "periodic" || v == "p") return BC::Periodic;
    if (v == "transmissive" || v == "outflow" || v == "t") return BC::Transmissive;
    throw std::runtime_error("ctrl line " + std::to_string(e.line) + ": '" + key
                             + "' must be 'periodic' or 'transmissive'");
}

const char* bc_str(BC b) { return b == BC::Periodic ? "periodic" : "transmissive"; }

const char* divb_str(DivBCleaningKind d) {
    switch (d) {
        case DivBCleaningKind::CT: return "CT";
        case DivBCleaningKind::GLM: return "GLM";
        default: return "none";
    }
}
const char* hall_stab_str(HallStabKind h) {
    switch (h) {
        case HallStabKind::HYPER_RES: return "hyper_res";
        case HallStabKind::HALL_HLL: return "hall_hll";
        default: return "none";
    }
}

bool uses_asym_profile(int test) {
    return test == 25 || test == 26 || test == 27 || test == 28 || test == 30;
}

} // namespace

RunConfig load_ctrl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open control deck '" + path + "'");

    std::map<std::string, Entry> kv;
    std::string raw;
    int lineno = 0;
    while (std::getline(in, raw)) {
        ++lineno;
        const std::string line = trim(strip_comment(raw));
        if (line.empty() || line[0] == '&' || line[0] == '/') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("ctrl line " + std::to_string(lineno)
                                     + ": expected 'key = value', got '" + line + "'");
        const std::string key = lower(trim(line.substr(0, eq)));
        std::string val = trim(line.substr(eq + 1));
        if (!val.empty() && val.back() == ',') val.pop_back();   // namelist style
        val = unquote(trim(val));
        if (lower(val) == "default") continue;
        if (kv.count(key))
            throw std::runtime_error("ctrl line " + std::to_string(lineno)
                                     + ": duplicate key '" + key + "'");
        kv[key] = Entry{val, lineno};
    }

    auto take = [&](const std::string& k, auto conv, auto dflt) {
        auto it = kv.find(k);
        if (it == kv.end()) return dflt;
        auto v = conv(k, it->second);
        kv.erase(it);
        return v;
    };

    // --- base configuration -------------------------------------------------
    if (!kv.count("test")) throw std::runtime_error("ctrl: 'test' is required");
    const int test = take("test", to_int, 0);
    const int nx   = take("nx", to_int, 200);
    const int ny   = take("ny", to_int, 200);
    const auto divb   = take("divb", to_divb, DivBCleaningKind::CT);
    const auto solver = take("solver", to_solver, SolverKind::HLLD);
    if (nx <= 0 || ny <= 0) throw std::runtime_error("ctrl: nx, ny must be > 0");

    RunConfig cfg = make_config_for_test(test, nx, ny, divb, solver);

    // --- field overrides (one entry per ctrl key) ----------------------------
    using Setter = std::function<void(const std::string&, const Entry&)>;
    const std::map<std::string, Setter> setters = {
        // &problem
        {"label",      [&](auto&, auto& e) { cfg.label = e.value; }},
        {"output_dir", [&](auto&, auto& e) { cfg.output_dir = expand_path(e.value); }},
        {"resume",     [&](auto& k, auto& e) { cfg.resume = to_bool(k, e); }},
        {"ic_check",   [&](auto& k, auto& e) { cfg.ic_check = to_bool(k, e); }},
        // &numerics
        {"cfl",        [&](auto& k, auto& e) { cfg.cfl = to_double(k, e); }},
        {"gamma",      [&](auto& k, auto& e) { cfg.gamma = to_double(k, e); }},
        {"rho_floor",  [&](auto& k, auto& e) { cfg.rho_floor = to_double(k, e); }},
        {"p_floor",    [&](auto& k, auto& e) { cfg.p_floor = to_double(k, e); }},
        {"subcycle_nonideal", [&](auto& k, auto& e) { cfg.subcycle_nonideal = to_bool(k, e); }},
        {"n_subcycle_max",    [&](auto& k, auto& e) { cfg.n_subcycle_max = to_int(k, e); }},
        {"bc_x",       [&](auto& k, auto& e) { cfg.bcx = to_bc(k, e); }},
        {"bc_y",       [&](auto& k, auto& e) { cfg.bcy = to_bc(k, e); }},
        // &time_control
        {"t_start",    [&](auto& k, auto& e) { cfg.t_start = to_double(k, e); }},
        {"t_stop",     [&](auto& k, auto& e) { cfg.t_end = to_double(k, e); }},
        {"output_dt",  [&](auto& k, auto& e) { cfg.output_dt = to_double(k, e); }},
        {"l1_dt",      [&](auto& k, auto& e) { cfg.l1_dt = to_double(k, e); }},
        {"ckpt_dt",    [&](auto& k, auto& e) { cfg.ckpt_dt = to_double(k, e); }},
        {"l2_stride",  [&](auto& k, auto& e) { cfg.l2_stride = to_int(k, e); }},
        // &physics
        {"eta",        [&](auto& k, auto& e) { cfg.eta = to_double(k, e); }},
        {"eta_h",      [&](auto& k, auto& e) { cfg.eta_H = to_double(k, e); }},
        {"hall_di",    [&](auto& k, auto& e) { cfg.hall_di = to_double(k, e); }},
        {"hall_stab",  [&](auto& k, auto& e) { cfg.hall_stab = to_hall_stab(k, e); }},
        {"psi0",       [&](auto& k, auto& e) { cfg.psi0 = to_double(k, e); }},
        {"driven_ez",  [&](auto& k, auto& e) { cfg.driven_ez = to_double(k, e); }},
        // &profile: B1/B2/rho1/rho2 feed both IC families. Asymmetric single
        // sheet (tests 25-28, 30): 1 = y>0 side, 2 = y<0 side. Double sheet
        // (test 31): 1 = outer sheet (B01, rho01), 2 = inner sheet (B02, rho02).
        {"b1",   [&](auto& k, auto& e) { cfg.asym_B1   = cfg.dh_B01   = to_double(k, e); }},
        {"b2",   [&](auto& k, auto& e) { cfg.asym_B2   = cfg.dh_B02   = to_double(k, e); }},
        {"rho1", [&](auto& k, auto& e) { cfg.asym_rho1 = cfg.dh_rho01 = to_double(k, e); }},
        {"rho2", [&](auto& k, auto& e) { cfg.asym_rho2 = cfg.dh_rho02 = to_double(k, e); }},
    };

    const BC bcx_default = cfg.bcx, bcy_default = cfg.bcy;
    bool profile_given = false;
    for (const auto& [key, entry] : kv) {
        auto it = setters.find(key);
        if (it == setters.end())
            throw std::runtime_error("ctrl line " + std::to_string(entry.line)
                                     + ": unknown key '" + key + "'");
        it->second(key, entry);
        if (key == "b1" || key == "b2" || key == "rho1" || key == "rho2") profile_given = true;
    }

    // --- validation -----------------------------------------------------------
    for (double v : {cfg.asym_B1, cfg.asym_B2, cfg.asym_rho1, cfg.asym_rho2})
        if (!(v > 0.0)) throw std::runtime_error("ctrl: B1, B2, rho1, rho2 must be > 0");
    if (!(cfg.t_end > cfg.t_start) && !cfg.ic_check)
        throw std::runtime_error("ctrl: t_stop must be greater than t_start");
    if (cfg.l2_stride < 1) throw std::runtime_error("ctrl: l2_stride must be >= 1");
    if (profile_given && !uses_asym_profile(cfg.test) && cfg.test != 31)
        std::cerr << "[ctrl] warning: B1/B2/rho1/rho2 are only used by tests "
                     "25-28, 30, 31; ignored for test " << cfg.test << "\n";
    // Boundary-condition consistency.
    if (cfg.bcy == BC::Periodic && cfg.driven_ez != 0.0)
        throw std::runtime_error("ctrl: driven_ez drives the y boundaries and needs "
                                 "bc_y = 'transmissive'");
    if (cfg.bcy == BC::Periodic && cfg.dirichlet_y_scalars)
        throw std::runtime_error("ctrl: test 28 Dirichlet y boundaries need "
                                 "bc_y = 'transmissive'");
    if (cfg.bcx != bcx_default || cfg.bcy != bcy_default) {
        std::cerr << "[ctrl] warning: boundary conditions changed from the test " << cfg.test
                  << " default (x=" << bc_str(bcx_default) << ", y=" << bc_str(bcy_default)
                  << ") to x=" << bc_str(cfg.bcx) << ", y=" << bc_str(cfg.bcy)
                  << "; the initial condition was built for the default.\n";
        if (cfg.test == 31 || cfg.test == 29)
            std::cerr << "[ctrl] warning: the double Harris sheet is constructed for a "
                         "doubly periodic box; non-periodic boundaries cut the outer "
                         "sheet at y = +/-Ly/2.\n";
    }

    sync_derived_params(cfg);
    return cfg;
}

std::string expand_path(const std::string& path) {
    std::string out;
    std::size_t i = 0;
    if (!path.empty() && path[0] == '~' && (path.size() == 1 || path[1] == '/')) {
        const char* home = std::getenv("HOME");
        if (!home) throw std::runtime_error("output_dir uses '~' but $HOME is not set");
        out = home;
        i = 1;
    }
    while (i < path.size()) {
        if (path[i] == '$') {
            std::size_t j = i + 1;
            std::string name;
            if (j < path.size() && path[j] == '{') {
                const auto close = path.find('}', j);
                if (close == std::string::npos)
                    throw std::runtime_error("output_dir: unterminated ${ in '" + path + "'");
                name = path.substr(j + 1, close - j - 1);
                j = close + 1;
            } else {
                while (j < path.size() && (std::isalnum(static_cast<unsigned char>(path[j])) || path[j] == '_'))
                    name += path[j++];
            }
            if (name.empty()) { out += path[i++]; continue; }
            const char* v = std::getenv(name.c_str());
            if (!v) throw std::runtime_error("output_dir: environment variable $" + name + " is not set");
            out += v;
            i = j;
        } else {
            out += path[i++];
        }
    }
    while (out.size() > 1 && out.back() == '/') out.pop_back();
    return out;
}

std::string run_output_dir(const RunConfig& cfg) {
    const std::string root = cfg.output_dir.empty() ? "output" : cfg.output_dir;
    return cfg.label.empty() ? root : root + "/" + cfg.label;
}

void sync_derived_params(RunConfig& cfg) {
    // test 28: Dirichlet y-boundary scalars = asymptotic IC state.
    // P_total as in asym_scan_params: max(B1², B2²)/2 + 1.
    if (cfg.dirichlet_y_scalars) {
        const double B1 = cfg.asym_B1, B2 = cfg.asym_B2;
        const double Pt = 0.5 * std::max(B1 * B1, B2 * B2) + 1.0;
        cfg.bc_rho_top = cfg.asym_rho1;  cfg.bc_p_top = Pt - 0.5 * B1 * B1;
        cfg.bc_rho_bot = cfg.asym_rho2;  cfg.bc_p_bot = Pt - 0.5 * B2 * B2;
    }
}

void apply_env_overrides(RunConfig& cfg) {
    if (const char* s = std::getenv("RESUME"); s && s[0] == '1') cfg.resume = true;
    if (const char* s = std::getenv("IC_CHECK"); s && s[0] == '1') cfg.ic_check = true;
    if (const char* s = std::getenv("L2_STRIDE")) {
        const int v = std::atoi(s);
        if (v > 1) cfg.l2_stride = v;
    }
}

void print_config(const RunConfig& cfg) {
    std::cout << "=== Run configuration ===\n"
              << "  test=" << cfg.test << "  grid=" << cfg.nx << "x" << cfg.ny
              << "  solver=" << (cfg.solver == SolverKind::FORCE ? "FORCE" : "HLLD")
              << "  divB=" << divb_str(cfg.divb) << "  cfl=" << cfg.cfl
              << "  gamma=" << cfg.gamma << "\n"
              << "  bc x=" << bc_str(cfg.bcx) << "  y=" << bc_str(cfg.bcy) << "\n"
              << "  domain x=[" << cfg.x0 << ", " << cfg.x1 << "]  y=[" << cfg.y0
              << ", " << cfg.y1 << "]\n"
              << "  t_start=" << cfg.t_start << "  t_stop=" << cfg.t_end
              << "  output_dt=" << cfg.output_dt << "\n"
              << "  eta=" << cfg.eta << "  eta_H=" << cfg.eta_H << "  hall_di=" << cfg.hall_di
              << "  hall_stab=" << hall_stab_str(cfg.hall_stab) << "  psi0=" << cfg.psi0
              << "  driven_ez=" << cfg.driven_ez << "\n";
    if (uses_asym_profile(cfg.test))
        std::cout << "  profile: B1=" << cfg.asym_B1 << " B2=" << cfg.asym_B2
                  << " rho1=" << cfg.asym_rho1 << " rho2=" << cfg.asym_rho2 << "\n";
    if (cfg.test == 31)
        std::cout << "  profile: B1(B01)=" << cfg.dh_B01 << " B2(B02)=" << cfg.dh_B02
                  << " rho1(rho01)=" << cfg.dh_rho01 << " rho2(rho02)=" << cfg.dh_rho02
                  << "  l1_dt=" << cfg.l1_dt << " ckpt_dt=" << cfg.ckpt_dt
                  << " l2_stride=" << cfg.l2_stride << "\n";
    if (cfg.dirichlet_y_scalars)
        std::cout << "  dirichlet y: top rho=" << cfg.bc_rho_top << " p=" << cfg.bc_p_top
                  << "  bot rho=" << cfg.bc_rho_bot << " p=" << cfg.bc_p_bot << "\n";
    std::cout << "  output -> " << run_output_dir(cfg)
              << "  resume=" << (cfg.resume ? "yes" : "no")
              << "  ic_check=" << (cfg.ic_check ? "yes" : "no") << "\n"
              << "=========================\n";
}

} // namespace my_project
