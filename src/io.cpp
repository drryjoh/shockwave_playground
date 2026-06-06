#include "splay/io.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <filesystem>

#ifdef SPLAY_ENABLE_MPI
#include <mpi.h>
#endif

namespace splay {

namespace fs = std::filesystem;

// ─── Helpers ─────────────────────────────────────────────────────────────────
static std::string step_str(int step, int width = 8) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << step;
    return oss.str();
}

static void ensure_dir(const std::string& path) {
    fs::create_directories(path);
}

// ─── CSV snapshot ─────────────────────────────────────────────────────────────
void write_csv(const State&          s,
               const Mesh&           m,
               const TransportModel& tm,
               const GasModel&       gas,
               const MPIDecomp&      decomp,
               const std::string&    output_dir,
               const std::string&    case_name,
               int                   step,
               double                time)
{
    const int ib = m.interior_begin();
    const int ie = m.interior_end();

    // Each rank writes its own file.
    std::string rank_dir = output_dir + "/" + case_name + "/snapshots";
    ensure_dir(rank_dir);

    std::string filename = rank_dir + "/step_" + step_str(step)
                         + "_rank" + std::to_string(decomp.rank) + ".csv";

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open output file: " + filename);

    out << std::scientific << std::setprecision(10);
    out << "# time=" << time << " step=" << step << "\n";
    out << "x,rho,u,p,T,Mach,mu,kappa\n";

    for (int i = ib; i < ie; ++i) {
        const double a    = gas.sound_speed(s.p[i], s.rho[i]);
        const double Mach = s.u[i] / a;
        const double mu_v = tm.viscosity(s.T[i]);
        const double k_v  = tm.conductivity(s.T[i]);
        out << m.x_cell[i] << ","
            << s.rho[i]    << ","
            << s.u[i]      << ","
            << s.p[i]      << ","
            << s.T[i]      << ","
            << Mach        << ","
            << mu_v        << ","
            << k_v         << "\n";
    }
    out.close();

    // Rank 0 gathers and writes a combined file.
#ifdef SPLAY_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (decomp.rank == 0) {
        std::string combined = rank_dir + "/step_" + step_str(step) + "_combined.csv";
        std::ofstream comb(combined);
        if (!comb) throw std::runtime_error("Cannot open combined file: " + combined);
        comb << "# time=" << time << " step=" << step << " nranks=" << decomp.nranks << "\n";
        comb << "x,rho,u,p,T,Mach,mu,kappa\n";
        for (int r = 0; r < decomp.nranks; ++r) {
            std::string rfile = rank_dir + "/step_" + step_str(step)
                              + "_rank" + std::to_string(r) + ".csv";
            std::ifstream rf(rfile);
            if (!rf) continue;
            std::string line;
            std::getline(rf, line);  // skip time comment
            std::getline(rf, line);  // skip header
            while (std::getline(rf, line)) comb << line << "\n";
        }
    }
}

// ─── Restart write ────────────────────────────────────────────────────────────
void write_restart(const State&       s,
                   const Mesh&        m,
                   const Config&      cfg,
                   const MPIDecomp&   decomp,
                   const std::string& restart_dir,
                   int                step,
                   double             time)
{
    ensure_dir(restart_dir);

    // Write per-rank data CSV.
    std::string data_file = restart_dir + "/rank" + std::to_string(decomp.rank) + "_data.csv";
    std::ofstream df(data_file);
    if (!df) throw std::runtime_error("Cannot write restart data: " + data_file);
    df << std::scientific << std::setprecision(15);
    df << "x,rho,rhou,rhoE,u,p,T\n";
    const int ib = m.interior_begin();
    const int ie = m.interior_end();
    for (int i = ib; i < ie; ++i) {
        df << m.x_cell[i] << ","
           << s.rho[i]    << ","
           << s.rhou[i]   << ","
           << s.rhoE[i]   << ","
           << s.u[i]      << ","
           << s.p[i]      << ","
           << s.T[i]      << "\n";
    }
    df.close();

    // Rank 0 writes YAML metadata.
    if (decomp.rank == 0) {
        std::string meta_file = restart_dir + "/metadata.yml";
        std::ofstream mf(meta_file);
        if (!mf) throw std::runtime_error("Cannot write restart metadata: " + meta_file);
        mf << "step: "    << step    << "\n";
        mf << "time: "    << std::scientific << std::setprecision(15) << time << "\n";
        mf << "nranks: "  << decomp.nranks   << "\n";
        mf << "gas: "     << cfg.gas         << "\n";
        mf << "case_name: " << cfg.case_name << "\n";
        mf << "n_global: " << m.n_global     << "\n";
        mf << "x_min: "   << m.x_min_global  << "\n";
        mf << "x_max: "   << m.x_max_global  << "\n";
        mf << "dx: "      << m.dx            << "\n";
        mf << "inviscid_scheme: ";
        switch (cfg.solver.inviscid_scheme) {
            case InviscidScheme::Central: mf << "central\n"; break;
            case InviscidScheme::MUSCL:   mf << "muscl\n";   break;
            case InviscidScheme::PPM:     mf << "ppm\n";     break;
        }
        mf << "viscous_terms: " << (cfg.solver.viscous_terms ? "true" : "false") << "\n";
    }
}

// ─── Restart read ─────────────────────────────────────────────────────────────
void read_restart(State&             s,
                  const Mesh&        m,
                  const MPIDecomp&   decomp,
                  const std::string& restart_path,
                  double&            time_out,
                  int&               step_out)
{
    // Read metadata from rank 0.
    std::string meta_file = restart_path + "/metadata.yml";
    {
        std::ifstream mf(meta_file);
        if (!mf) throw std::runtime_error("Cannot read restart metadata: " + meta_file);
        std::string key;
        while (mf >> key) {
            if (key == "step:") mf >> step_out;
            else if (key == "time:") mf >> time_out;
            else { std::string dummy; mf >> dummy; }
        }
    }

    // Read per-rank data.
    std::string data_file = restart_path + "/rank" + std::to_string(decomp.rank) + "_data.csv";
    std::ifstream df(data_file);
    if (!df) throw std::runtime_error("Cannot read restart data: " + data_file);

    std::string header;
    std::getline(df, header);  // skip header

    const int ib = m.interior_begin();
    int i = ib;
    std::string line;
    while (std::getline(df, line) && i < m.interior_end()) {
        std::istringstream ss(line);
        char comma;
        double x;
        ss >> x >> comma
           >> s.rho[i]  >> comma
           >> s.rhou[i] >> comma
           >> s.rhoE[i] >> comma
           >> s.u[i]    >> comma
           >> s.p[i]    >> comma
           >> s.T[i];
        ++i;
    }
}

} // namespace splay
