#pragma once
#include <string>
#include "splay/state.hpp"
#include "splay/mesh.hpp"
#include "splay/config.hpp"
#include "splay/transport.hpp"
#include "splay/mpi_decomp.hpp"

namespace splay {

/// Write a CSV solution snapshot.
/// Each MPI rank writes its own file; rank 0 also writes a combined file.
///
/// Columns: x, rho, u, p, T, Mach, mu, kappa
void write_csv(const State&          s,
               const Mesh&           m,
               const TransportModel& tm,
               const GasModel&       gas,
               const MPIDecomp&      decomp,
               const std::string&    output_dir,
               const std::string&    case_name,
               int                   step,
               double                time);

/// Write a restart file (one per rank).
/// Format: YAML metadata + CSV data.
void write_restart(const State&       s,
                   const Mesh&        m,
                   const Config&      cfg,
                   const MPIDecomp&   decomp,
                   const std::string& restart_dir,
                   int                step,
                   double             time);

/// Read a restart file and populate the state.
/// Returns the time and step stored in the restart.
void read_restart(State&             s,
                  const Mesh&        m,
                  const MPIDecomp&   decomp,
                  const std::string& restart_path,
                  double&            time_out,
                  int&               step_out);

} // namespace splay
