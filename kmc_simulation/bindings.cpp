#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "lattice.h"
#include "monte_carlo.h"
#include "rate_calculator.h"
#include "site_info.h"
#include "variable_field.h"
#include "utilities.h"
#include "simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(kmc_lattice_gas, m) {
  m.doc() = "This is a module for simulating the dynamics of a 3D lattice gas model for nanoparticle self-assembly.";

  py::class_<Cluster>(m, "Cluster")
    .def(py::init<>())
    .def_readwrite("size", &Cluster::size)
    .def_readwrite("sites", &Cluster::sites);

  py::enum_<CellType>(m, "CellType")
    .value("SURFACE", CellType::SURFACE)
    .value("FLUID", CellType::FLUID)
    .value("NANOPARTICLE", CellType::NANOPARTICLE)
    .export_values();

  py::class_<Lattice>(m, "Lattice")
    .def(py::init<const int, const int, const int>())
    .def("pos2idx", &Lattice::pos2idx)
    .def("select_interface_site", &Lattice::select_interface_site)
    .def("get_neighboring_indices", &Lattice::get_neighboring_indices)
    .def("update_lattice_with_move", &Lattice::update_lattice_with_move)
    .def("get_site_info", &Lattice::get_site_info)
    .def("find_clusters", &Lattice::find_clusters)
    .def("get_cluster_size_distribution", &Lattice::get_cluster_size_distribution)
    .def_readwrite("volume", &Lattice::volume)
    .def_readwrite("n_side", &Lattice::n_side)
    .def_readwrite("height", &Lattice::height)
    .def_readwrite("num_np", &Lattice::num_np)
    .def_readwrite("n_fluid", &Lattice::n_fluid)
    .def_readwrite("np_lat_sides", &Lattice::np_lat_sides)
    .def_readwrite("sim_box", &Lattice::sim_box)
    .def_readwrite("orientations", &Lattice::orientations)
    .def_readwrite("nl", &Lattice::nl)
    .def_readwrite("np_list", &Lattice::np_list);

  py::class_<SiteInfo>(m, "SiteInfo")
    .def(py::init<>())
    .def(py::init<int, int, int, int, int>())
    .def("to_string", &SiteInfo::to_string)
    .def("get_total_counts", &SiteInfo::get_total_counts)
    .def_readwrite("central_type", &SiteInfo::central_type)
    .def_readwrite("bottom_neighbor", &SiteInfo::bottom_neighbor)
    .def_readwrite("lateral_f_count", &SiteInfo::lateral_f_count)
    .def_readwrite("lateral_n_count", &SiteInfo::lateral_n_count)
    .def_readwrite("top_neighbor", &SiteInfo::top_neighbor)
    .def_readwrite("total_n_count", &SiteInfo::total_n_count)
    .def_readwrite("total_f_count", &SiteInfo::total_f_count)
    .def_readwrite("total_s_count", &SiteInfo::total_s_count)
    .def_readwrite("axial_n_count", &SiteInfo::axial_n_count)
    .def_readwrite("site_id", &SiteInfo::site_id)
    .def_readwrite("total_counts", &SiteInfo::total_counts);

  py::class_<MC>(m, "MC")
    .def(py::init<>())
    .def("get_possible_moves", &MC::get_possible_moves)
    .def("sample_move", py::overload_cast<Lattice&, RateCalculator&, VariableField&, const int>(&MC::sample_move))
    .def("sample_move", py::overload_cast<Lattice&, RateCalculator&, const int>(&MC::sample_move))
    .def_readwrite("time", &MC::time)
    .def_readwrite("max_move_rate", &MC::max_move_rate)
    .def_readwrite("move_set", &MC::move_set)
    .def_readwrite("move_rates", &MC::move_rates);

  py::class_<RateCalculator>(m, "RateCalculator")
    .def(py::init<Lattice&, double, double, double, double, double, double, double, double, double, double, double>())
    .def("calculate_fluc_rate", &RateCalculator::calculate_fluc_rate)
    .def("calculate_diff_rate", &RateCalculator::calculate_diff_rate)
    .def("calculate_rot_rate", &RateCalculator::calculate_rot_rate)
    .def("calculate_total_lattice_energy", &RateCalculator::calculate_total_lattice_energy)
    .def("get_alignment_factor", &RateCalculator::get_alignment_factor)
    .def("get_rates", &RateCalculator::get_rates)
    .def_readwrite("temperature", &RateCalculator::temperature)
    .def_readwrite("total_energy", &RateCalculator::total_energy)
    .def_readwrite("ens", &RateCalculator::ens)
    .def_readwrite("enn", &RateCalculator::enn);
  
  py::class_<VariableField>(m, "VariableField")
    .def(py::init<RateCalculator&, Lattice&, double, double, double, double, double>())
    .def("set_amplitude", &VariableField::set_amplitude)
    .def("set_frequency", &VariableField::set_frequency)
    .def("update_ens_", py::overload_cast<RateCalculator&, double>(&VariableField::update_ens_))
    .def("update_ens_", py::overload_cast<RateCalculator&, double, double>(&VariableField::update_ens_))
    .def("update_ens", &VariableField::update_ens)
    .def("update_rate_parameter", &VariableField::update_rate_parameter)
    .def_readwrite("lower", &VariableField::lower)
    .def_readwrite("delay", &VariableField::delay)
    .def_readwrite("amplitude", &VariableField::amplitude)
    .def_readwrite("frequency", &VariableField::frequency);

  py::class_<Utilities>(m, "Utilities")
    .def(py::init<>())
    .def("save_lattice_traj_xyz", &Utilities::save_lattice_traj_xyz)
    .def("final_print", &Utilities::final_print);

  py::class_<Simulation>(m, "Simulation")
    .def(py::init<int, double, double, double, double, double, double, double, double, double, double, double, double, int, int>(),
    R"doc(Run kinetic Monte Carlo simulation of nanoparticle lattice gas. 
    
    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducibility
    enn : float, default=1.0
        Nanoparticle-nanoparticle interaction energy
    ens : float, default=1.0
        Nanoparticle-substrate interaction energy
    enf : float, default=1.0
        Nanoparticle-fluid interaction energy
    esf : float, default=1.0
        Substrate-fluid interaction energy
    eff : float, default=1.0
        Fluid-fluid interaction energy
    muf : float, default=1.0
        Chemical potential of fluid
    mun : float, default=5.0
        Chemical potential of nanoparticles
    fug : float, default=0.0
        Fugacity
    temperature : float, default=0.1
        System temperature in reduced units
    diffusion_prefactor : float, default=1.0
        Prefactor for diffusion rates
    fluctuation_prefactor : float, default=1.0
        Prefactor for fluctuation rates
    rotational_prefactor : float, default=10.0
        Prefactor for rotational rates
    height : int, default=10
        Height of simulation box in lattice units
    n_side : int, default=50
        Length of simulation box side in lattice units

    Returns
    -------
    None
        Results are saved to trajectory file if save_lattice_traj=True

    Examples
    --------
    >>> import kmc_lattice_gas
    >>> # Run with default parameters
    >>> kmc_lattice_gas.run_simulation()
    >>> 
    >>> # Run with custom parameters
    >>> kmc_lattice_gas.run_simulation(
    ...     seed=123,
    ...     enn=2.0,
    ...     temperature=0.2
    ... )
  )doc",
    py::arg("seed"),
    py::arg("enn"), 
    py::arg("ens"),
    py::arg("enf")=1.0,
    py::arg("esf")=1.0,
    py::arg("eff")=1.0,
    py::arg("muf")=1.0,
    py::arg("mun")=5.0,
    py::arg("fug")=0.0,
    py::arg("temperature")=0.1,
    py::arg("diffusion_prefactor")=1.0,
    py::arg("fluctuation_prefactor")=1.0,
    py::arg("rotational_prefactor")=10.0,
    py::arg("height")=10,
    py::arg("n_side")=50
  )
  .def("step", &Simulation::step,
    R"doc(Perform a single Monte Carlo step in the simulation.

    Parameters
    ----------
    num_steps : int
        Number of steps to run

    Returns
    -------
    None
        Results are saved to trajectory file if save_lattice_traj=True

    Examples
    --------
    >>> import kmc_lattice_gas
    >>> kmc_lattice_gas.step(1000)
  )doc", py::arg("num_steps")
  )
  .def("reset", &Simulation::reset,
  R"doc(Reset the simulation to initial conditions.
  )doc"
  )
  .def("take_action", py::overload_cast<const std::vector<double>&>(&Simulation::take_action),
    R"doc(Update the simulation with new rates.

    Parameters
    ----------
    ens_update : list
        List of new ens values at each site.

    Returns
    -------
    None
        Results are saved to trajectory file if save_lattice_traj=True
  )doc", 
    py::arg("ens_update")
  )
  .def("take_action", py::overload_cast<const double&, bool>(&Simulation::take_action),
    R"doc(Update the simulation with new rates.

    Parameters
    ----------
    ens_update : float
        New parameter value (globally).
    update_temp : bool
        Type of update to perform. If True, update temperature. If False, update ens values with same value.

    Returns
    -------
    None
        Results are saved to trajectory file if save_lattice_traj=True
  )doc",
    py::arg("update_value"),
    py::arg("update_temp")
  )
  .def("save_traj", &Simulation::save_traj,
    R"doc(Save the trajectory to an XYZ file.

    Parameters
    ----------
    filename : str
        Name of the file to save the trajectory to.

    Returns
    -------
    None
        Results are saved to trajectory file if save_lattice_traj=True
  )doc",
    py::arg("filename")
  )
  .def("get_state", [](Simulation& sim) {
    std::vector<int> cpp_state = sim.get_state();
    return py::array(py::cast(cpp_state));
  })
  .def("get_box", [](Simulation& sim) {
    std::vector<int> cpp_box = sim.get_box();
    return py::array(py::cast(cpp_box));
  })
  .def("print_state", &Simulation::print_state)
  .def_readwrite("time", &Simulation::time)
  .def_readwrite("seed_value", &Simulation::seed_value)
  .def_property("ens_grid",
    [](Simulation& sim) -> std::vector<double>& { return sim.ens_grid; },
    [](Simulation& sim, const std::vector<double>& value) { sim.ens_grid = value; }
  )
  .def_property("num_np",
    [](Simulation& sim) -> int& { return sim.num_np; },
    [](Simulation& sim, int value) { sim.num_np = value; }
  )
  .def_property("temperature",
    [](Simulation& sim) -> double& { return sim.temperature; },
    [](Simulation& sim, double value) { sim.temperature = value; }
  );
}